from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
import threading
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")


class VideoDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.split = split
        self.data_root = Path(cfg.data_root)
        self.metadata_path = Path(cfg.metadata_path)
        self.auto_download = cfg.auto_download
        self.force_download = cfg.force_download
        self.test_percentage = cfg.test_percentage
        self.id_token = cfg.id_token or ""
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.fps = cfg.fps
        self.trim_mode = cfg.trim_mode
        self.pad_mode = cfg.pad_mode
        self.filtering = cfg.filtering
        self.load_video_latent = cfg.load_video_latent
        self.load_prompt_embed = cfg.load_prompt_embed
        self.augmentation = cfg.augmentation
        self.image_to_video = cfg.image_to_video
        self.max_text_tokens = cfg.max_text_tokens

        # trigger auto-download if not already downloaded
        trigger_download = False
        if not self.data_root.is_dir():
            print(f"Dataset root folder {self.data_root} does not exist.")
            if not self.auto_download:
                raise ValueError(
                    f"Attempting to automatically download the dataset since dataset root folder {self.data_root} does not exist. "
                    "If this is the intended behavior, append `dataset.auto_download=True` in your command to pass this check."
                )
            trigger_download = True
        if self.force_download:
            trigger_download = True
        if trigger_download:
            # if threading.current_thread() is not threading.main_thread():
            if torch.distributed.is_initialized():
                raise ValueError(
                    "Download must be called from the main thread with single-process training. Did you call this inside a multi-worker dataloader?"
                )
            print(f"Attempting to download dataset to {self.data_root}...")
            self.download()

        self.records = self._load_records()  # a list of dictionaries
        self.augment_transforms = self._build_video_transforms(augment=True)
        self.no_augment_transforms = self._build_video_transforms(augment=False)
        self.img_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
        )

        if self.trim_mode not in ["speedup", "random_cut"]:
            raise ValueError(
                f"Invalid trim_mode: {self.trim_mode}. Must be one of ['speedup', 'random_cut']."
            )
        if self.pad_mode not in ["slowdown", "pad_last", "discard"]:
            raise ValueError(
                f"Invalid pad_mode: {self.pad_mode}. Must be one of ['slowdown', 'pad_last', 'discard']."
            )

    def _build_video_transforms(self, augment: bool = True):
        trans = []
        if augment and self.augmentation.random_flip is not None:
            trans.append(transforms.RandomHorizontalFlip(self.augmentation.random_flip))

        augment = self.augmentation.is_augment if hasattr(self.augmentation, 'is_augment') else augment

        if augment:

            aspect_ratio = self.width / self.height
            aspect_ratio = [aspect_ratio, aspect_ratio]
            if augment and self.augmentation.ratio is not None:
                aspect_ratio[0] *= self.augmentation.ratio[0]
                aspect_ratio[1] *= self.augmentation.ratio[1]

            scale = [1.0, 1.0]
            if augment and self.augmentation.scale is not None:
                scale[0] *= self.augmentation.scale[0]
                scale[1] *= self.augmentation.scale[1]

            trans.append(
                transforms.RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=scale,
                    ratio=aspect_ratio,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
        else:
            # Validation uses deterministic resize and center crop.
            trans.append(transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC))
            trans.append(transforms.CenterCrop((self.height, self.width)))

        return transforms.Compose(trans)

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # a hook to modify the original record on the fly
        return record

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: Any) -> Dict[str, Any]:
        # Handle tuple index from DistributedKRepeatSampler for GRPO deterministic sampling
        seed = None
        if isinstance(idx, (tuple, list)):
            seed, idx = idx
        
        record = self.records[idx]

        # Context manager for temporary random seed
        rng_state = None
        if seed is not None:
             # Save current random state
             rng_state =  np.random.get_state()
             # Set deterministic seed for this sample group
             try:
                np.random.seed((int(seed) + int(idx)) % (2**32))
             except ValueError:
                # Fallback if seed + idx overflows or is invalid
                np.random.seed(int(seed) % (2**32))

        try:
            # Load video data - either raw or preprocessed latents
            videos = self._load_video(record)
            # images = videos[:1].clone() if self.image_to_video else None
            image_latents, video_latents = None, None
            video_metadata = {
                "num_frames": videos.shape[0],
                "height": videos.shape[2],
                "width": videos.shape[3],
            }

            if self.load_video_latent:
                image_latents, video_latents = self._load_video_latent(record)
                # This is hardcoded for now.
                # The VAE's temporal compression ratio is 4.
                # The VAE's spatial compression ratio is 8.
                latent_num_frames = video_latents.size(1)
                if latent_num_frames % 2 == 0:
                    n_frames = latent_num_frames * 4
                else:
                    n_frames = (latent_num_frames - 1) * 4 + 1

                height = video_latents.size(2) * 8
                width = video_latents.size(3) * 8

                assert video_metadata["num_frames"] == n_frames, "num_frames changed"
                assert video_metadata["height"] == height, "height changed"
                assert video_metadata["width"] == width, "width changed"

            # Load prompt data - either raw or preprocessed embeddings
            caption = record.get("caption", "")
            video_metadata["has_caption"] = caption != ""
            prompts = self.id_token + caption
            prompt_embeds = None
            prompt_embed_len = None
            if self.load_prompt_embed:
                prompt_embeds, prompt_embed_len = self._load_prompt_embed(record)

            has_bbox, bbox_render = self._render_bbox(record)

            output = {
                "videos": videos,
                "video_metadata": video_metadata,
                "bbox_render": bbox_render,
                "has_bbox": has_bbox,
                "video_path": str(self.data_root / record["video_path"]),
            }

            if prompts is not None:
                output["prompts"] = prompts
            # if images is not None:
            #     output["images"] = images
            if prompt_embeds is not None:
                output["prompt_embeds"] = prompt_embeds
                output["prompt_embed_len"] = prompt_embed_len
            if image_latents is not None:
                output["image_latents"] = image_latents
            if video_latents is not None:
                output["video_latents"] = video_latents
        finally:
            # Restore random state
            if rng_state is not None:
                np.random.set_state(rng_state)

        return output

    def _n_frames_in_src(self, src_fps):
        """
        Given the fps of the source video, return the number of frames in it we shall
        use in order to generate a target video of self.n_frames frames at self.fps.

        Note the definition of fps of the source video is described in README.md as,
        for a real-world task that requires 1 second to finish, how many frames does it
        take this source video to capture? This is usually just the fps of the source
        video, but if the source video is already a slow motion video, this may be
        different.
        """
        return round(self.n_frames / self.fps * src_fps)

    def _temporal_sample(self, n_frames: int, fps: int) -> torch.Tensor:
        """
        Given number of frames and fps, return a sequence of frame indices to downsample / upsample the video temporally.
        This shall consider self.n_frames and fps.
        """

        # target_len is the number of frames in the source video that we shall use to generate a target video of self.n_frames frames at self.fps
        target_len = self._n_frames_in_src(fps)

        if n_frames < target_len:
            if self.pad_mode == "pad_last":
                indices = np.linspace(0, target_len - 1, self.n_frames)
                indices = np.clip(indices, 0, n_frames - 1)
            elif self.pad_mode == "slowdown":
                indices = np.linspace(0, n_frames - 1, self.n_frames)
            elif self.pad_mode == "discard":
                raise ValueError(
                    "pad_mode is set to 'discard', but this short video is not filtered out."
                )
            else:
                raise ValueError(f"Invalid pad_mode: {self.pad_mode}")
        elif n_frames > target_len:
            if self.trim_mode == "random_cut":
                start = np.random.randint(0, n_frames - target_len)
                indices = start + np.linspace(0, target_len - 1, self.n_frames)
            elif self.trim_mode == "speedup":
                indices = np.linspace(0, n_frames - 1, self.n_frames)
            elif self.trim_mode == "discard":
                raise ValueError(
                    "trim_mode is set to 'discard', but this long video is not filtered out."
                )
            else:
                raise ValueError(f"Invalid trim_mode: {self.trim_mode}")
        else:
            indices = np.linspace(0, n_frames - 1, self.n_frames)

        indices = np.round(indices).astype(int)
        return indices

    def _load_video(self, record: Dict[str, Any]) -> torch.Tensor:
        """
        Given a record, return a tensor of shape (n_frames, 3, H, W)
        """

        video_path = self.data_root / record["video_path"]
        video_reader = decord.VideoReader(uri=video_path.as_posix())
        n_frames = len(video_reader)
        start = record.get("trim_start", 0)
        end = record.get("trim_end", n_frames)
        indices = self._temporal_sample(end - start, record["fps"])
        indices = list(start + indices)
        frames = video_reader.get_batch(indices)

        # do some padding
        if len(frames) != self.n_frames:
            raise ValueError(
                f"Expected {len(frames)=} to be equal to {self.n_frames=}."
            )

        # crop if specified in the record
        if "crop_top" in record and "crop_bottom" in record:
            frames = frames[:, record["crop_top"] : record["crop_bottom"]]
        if "crop_left" in record and "crop_right" in record:
            frames = frames[:, :, record["crop_left"] : record["crop_right"]]

        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0

        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
        else:
            frames = self.augment_transforms(frames)
        frames = self.img_normalize(frames)

        return frames

    def _render_bbox(self, record: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a record, return a tensor of shape (H, W)
        """

        # first frame and last frame forms 2 channels
        bbox_render = torch.zeros(2, record["height"], record["width"])
        has_bbox = torch.zeros(2, dtype=torch.bool)
        # if "first_frame_has_bbox" in record and record["first_frame_has_bbox"]:
        #     has_bbox[0] = True
        #     bbox_top = int(record["first_frame_bbox_top"])
        #     bbox_bottom = int(record["first_frame_bbox_bottom"])
        #     bbox_left = int(record["first_frame_bbox_left"])
        #     bbox_right = int(record["first_frame_bbox_right"])
        #     bbox_render[0, bbox_top:bbox_bottom, bbox_left:bbox_right] = 1
        # if "last_frame_has_bbox" in record and record["last_frame_has_bbox"]:
        #     has_bbox[-1] = True
        #     bbox_top = int(record["last_frame_bbox_top"])
        #     bbox_bottom = int(record["last_frame_bbox_bottom"])
        #     bbox_left = int(record["last_frame_bbox_left"])
        #     bbox_right = int(record["last_frame_bbox_right"])
        #     bbox_render[-1, bbox_top:bbox_bottom, bbox_left:bbox_right] = 1
        bbox_render = self.no_augment_transforms(bbox_render)
        return has_bbox, bbox_render

    def _load_records(self) -> Tuple[List[str], List[str]]:
        """
        Given the metadata file, loads the records as a list.
        Each record is a dictionary containing a datapoint's video path / caption etc.
        Require these entries: "video_path", "caption", "height", "width", "n_frames", "fps"
        Optional entry: "split" - if present, will be used instead of test_percentage
        """
        records = pd.read_csv(self.data_root / self.metadata_path, na_filter=False)
        records = records.to_dict("records")
        len_pre_filter = len(records)
        if not self.filtering.disable:
            records = [record for record in records if self._filter_record(record)]
        len_post_filter = len(records)

        print(
            f"{self.data_root / self.metadata_path}: filtered {len_pre_filter - len_post_filter} records from {len_pre_filter} to {len_post_filter}, rataining rate: {len_post_filter / len_pre_filter}"
        )

        if self.cfg.check_video_path and not self.debug:
            print("Checking records such that all video_path are valid...")
            print(
                "This could take a while. To skip, append `dataset.check_video_path=False` to your command."
            )
            for r in tqdm(records, desc="Checking video paths"):
                self._check_record(r)
            print("Done checking records")

        # Handle split selection
        if self.split != "all":
            if "split" in records[0]:
                # Use split field from records
                records = [r for r in records if r["split"] == self.split]
            else:
                # Random split logic
                all_indices = list(range(len(records)))
                # Fix seed for consistency between train/val dataset instantiation
                random.Random(0).shuffle(all_indices)

                num_test = int(len(records) * self.test_percentage)
                if self.split == "training":
                    split_indices = all_indices[:-num_test] if num_test > 0 else all_indices
                else:  # validation/test
                    split_indices = all_indices[-num_test:] if num_test > 0 else []

                records = [records[i] for i in split_indices]

                # Record split indices
                try:
                    from hydra.core.hydra_config import HydraConfig
                    if HydraConfig.initialized():
                        output_dir = Path(HydraConfig.get().runtime.output_dir)
                    else:
                        output_dir = Path.cwd()
                    
                    save_path = output_dir / f"{self.split}_split_indices.txt"
                    print(f"Saving {len(split_indices)} {self.split} indices to {save_path}")
                    with open(save_path, "w") as f:
                        for idx in sorted(split_indices):
                            f.write(f"{idx}\n")
                except Exception as e:
                    print(f"Failed to record split indices: {e}")

        if not records:
            raise ValueError(
                f"No record found for split '{self.split}', check dataset.test_percentage or dataset.filtering"
            )

        random.Random(0).shuffle(records)

        records = [self.preprocess_record(record) for record in records]

        return records

    def _filter_record(self, x: Dict[str, Any]) -> bool:
        """
        x is a record dictionary containing a datapoint's video path / caption etc.
        Returns True if the record should be kept, False otherwise.
        """
        h, w, fps = x["height"], x["width"], x["fps"]

        # if record specified a crop, use that
        if "crop_left" in x and "crop_right" in x:
            w = x["crop_right"] - x["crop_left"]
        if "crop_top" in x and "crop_bottom" in x:
            h = x["crop_bottom"] - x["crop_top"]
        if "trim_start" in x and "trim_end" in x:
            n_frames = x["trim_end"] - x["trim_start"]
        elif "n_frames" in x:
            n_frames = x["n_frames"]
        else:
            raise ValueError(
                "Record missing required key 'n_frames', if trim not specified"
            )

        h_range = self.filtering.height
        if h_range is not None and h < h_range[0] or h > h_range[1]:
            return False
        w_range = self.filtering.width
        if w_range is not None and w < w_range[0] or w > w_range[1]:
            return False
        f_range = self.filtering.n_frames
        if f_range is not None and n_frames < f_range[0] or n_frames > f_range[1]:
            return False
        fps_range = self.filtering.fps
        if fps_range is not None and fps < fps_range[0] or fps > fps_range[1]:
            return False
        if n_frames < self._n_frames_in_src(fps) and self.pad_mode == "discard":
            return False

        # then filter using stable_background, stable_brightness,
        # note that some datasets may not have these keys
        if "stable_background" in x and not x["stable_background"]:
            return False
        if "stable_brightness" in x and not x["stable_brightness"]:
            return False

        return True

    def _check_record(self, x: Dict[str, Any]) -> bool:
        """
        x is a record dictionary containing a datapoint's video path / caption etc.
        raise an error if the record is not valid. e.g.
        """
        video_path = self.data_root / x["video_path"]
        if not video_path.is_file():
            msg = f"Expected `{video_path=}` to be a valid file but found it to be invalid."
            if self.debug:
                print(msg)
            else:
                raise ValueError(msg)

    def _load_video_latent(
        self, record: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if "video_latent_path" not in record:
            raise ValueError("Record missing required key 'video_latent_path'")
        video_latent_path = self.data_root / record["video_latent_path"]

        image_latent = None
        if self.image_to_video:
            if "image_latent_path" not in record:
                raise ValueError("Record missing required key 'image_latent_path'")
            image_latent_path = self.data_root / record["image_latent_path"]
            image_latent = torch.load(
                image_latent_path, map_location="cpu", weights_only=True
            )
        video_latent = torch.load(
            video_latent_path, map_location="cpu", weights_only=True
        )

        return image_latent, video_latent

    def _load_prompt_embed(self, record: Dict[str, Any]) -> torch.Tensor:
        # if self.debug:
        #     return torch.zeros(self.max_text_tokens, 4096), self.max_text_tokens

        if "prompt_embed_path" not in record:
            raise ValueError("Record missing required key 'prompt_embed_path'")
        prompt_embed_path = self.data_root / record["prompt_embed_path"]
        prompt_embed = torch.load(
            prompt_embed_path, map_location="cpu", weights_only=True
        )

        prompt_embed_len = prompt_embed.size(0)
        if prompt_embed_len < self.max_text_tokens:
            # Pad with zeros to max_text_tokens
            padding = torch.zeros(
                self.max_text_tokens - prompt_embed.size(0),
                prompt_embed.size(1),
                dtype=prompt_embed.dtype,
                device=prompt_embed.device,
            )
            prompt_embed = torch.cat([prompt_embed, padding], dim=0)

        return prompt_embed, prompt_embed_len

    def download(self):
        """
        Automatically download the dataset to self.data_root. Optional.
        """
        raise NotImplementedError(
            "Automatic download not implemented for this dataset."
        )


class SingleFrameVideoDataset(VideoDataset):
    """
    Datasets for our own images for testing purposes. The video_path is placed with image_path.
    The image is treated as a first frame, and a dummy video is generated by repeating the image.
    """

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if "image_path" in record and "video_path" not in record:
            record["video_path"] = record["image_path"]
        return record

    def _load_video(self, record: Dict[str, Any]) -> torch.Tensor:
        image_path = self.data_root / record["video_path"]
        image = Image.open(image_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).to(torch.uint8)
        frames = image.unsqueeze(0).repeat(self.n_frames, 1, 1, 1)

        # crop if specified in the record
        if "crop_top" in record and "crop_bottom" in record:
            frames = frames[:, record["crop_top"] : record["crop_bottom"]]
        if "crop_left" in record and "crop_right" in record:
            frames = frames[:, :, record["crop_left"] : record["crop_right"]]

        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0

        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
        else:
            frames = self.augment_transforms(frames)
        frames = self.img_normalize(frames)

        return frames

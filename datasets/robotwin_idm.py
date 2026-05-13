from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from idm import DinoPreprocessor, flip_action


def robotwin_idm_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["images"] for sample in batch], dim=0)
    actions = torch.stack([sample["actions"] for sample in batch], dim=0)

    return {
        "images": images,
        "actions": actions,
        "metadata": [sample["metadata"] for sample in batch],
    }


class RobotwinIDMDataset(Dataset):
    def __init__(self, cfg, split: str = "training") -> None:
        self.cfg = cfg
        self.split = split
        self.data_root = Path(cfg.data_root)
        self.task_config = cfg.get("task_config", "demo_clean_vidar")
        self.action_key = cfg.get("action_key", "joint_action/vector")
        self.rgb_key = cfg.get("rgb_key", "observation/head_camera/rgb")
        self.num_frames = cfg.get("num_frames", 3)
        self.frame_stride = cfg.get("frame_stride", 2)
        self.do_flip = split == "training" and cfg.get("do_flip", False)
        self.resize_to_target_size = cfg.get("resize_to_target_size", True)
        self.image_width = int(cfg.get("image_width", 640))
        self.image_height = int(cfg.get("image_height", 480))
        self.preprocessor = DinoPreprocessor(
            {
                "use_transform": split == "training" and cfg.get("use_transform", True),
                "dino_size": cfg.get("dino_size", 512),
            }
        )
        self.preprocessor.set_augmentation_progress(0.0)

        self.h5_paths: list[Path] = []
        self.lengths: list[int] = []
        self.actions: list[torch.Tensor] = []
        self.task_names: list[str] = []

        self._scan_metadata()

    def _scan_metadata(self) -> None:
        h5_files: list[Path] = []
        for task_dir in sorted(self.data_root.iterdir()):
            if not task_dir.is_dir():
                continue
            cfg_dir = task_dir / self.task_config
            if not cfg_dir.exists():
                continue
            for path in sorted(cfg_dir.rglob("*.hdf5")):
                h5_files.append(path)

        if not h5_files:
            raise RuntimeError(
                f"No HDF5 files found under {self.data_root} with task_config={self.task_config}"
            )

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as handle:
                if self.action_key not in handle or self.rgb_key not in handle:
                    continue
                rgb_ds = handle[self.rgb_key]
                action_ds = handle[self.action_key]
                length = min(len(rgb_ds), len(action_ds))
                if length <= 0:
                    continue
                self.h5_paths.append(h5_path)
                self.lengths.append(length)
                self.actions.append(torch.from_numpy(action_ds[:length]).float())
                self.task_names.append(h5_path.relative_to(self.data_root).parts[0])

        if not self.h5_paths:
            raise RuntimeError(
                f"Found {len(h5_files)} HDF5 files under {self.data_root}, but none matched "
                f"the required keys {self.rgb_key} and {self.action_key}."
            )

        self.data_begin = np.cumsum([0] + self.lengths[:-1])
        self.data_end = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.data_end[-1])

    def _resize_image(self, image: np.ndarray, interpolation: int) -> np.ndarray:
        if not self.resize_to_target_size:
            return image
        height, width = image.shape[:2]
        if width == self.image_width and height == self.image_height:
            return image
        return cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=interpolation,
        )

    def _decode_image(self, data):
        if isinstance(data, np.bytes_):
            data = bytes(data)

        if isinstance(data, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode encoded RGB image.")
            image = self._resize_image(image, cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

        if isinstance(data, np.ndarray):
            image = self._resize_image(data, cv2.INTER_CUBIC)
            return Image.fromarray(image)

        raise TypeError(f"Unsupported RGB payload type: {type(data)}")

    def _frame_indices(self, local_idx: int, video_len: int) -> list[int]:
        if self.num_frames == 1:
            indices = [local_idx]
        elif self.num_frames == 2:
            indices = [local_idx - self.frame_stride, local_idx]
        elif self.num_frames == 3:
            indices = [
                local_idx - self.frame_stride,
                local_idx,
                local_idx + self.frame_stride,
            ]
        else:
            raise ValueError(f"Unsupported num_frames={self.num_frames}")
        return [max(0, min(idx, video_len - 1)) for idx in indices]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video_idx = int(np.searchsorted(self.data_end, idx, side="right"))
        local_idx = int(idx - self.data_begin[video_idx])
        h5_path = self.h5_paths[video_idx]

        with h5py.File(h5_path, "r") as handle:
            rgb_ds = handle[self.rgb_key]
            video_len = len(rgb_ds)
            image_indices = self._frame_indices(local_idx, video_len)
            images = [
                self.preprocessor.process_image(self._decode_image(rgb_ds[frame_idx]))
                for frame_idx in image_indices
            ]

        images = torch.stack(images, dim=0)
        actions = self.actions[video_idx][local_idx].clone()

        is_flipped = self.do_flip and np.random.rand() < 0.5
        if is_flipped:
            images = torch.flip(images, dims=[-1])
            actions = flip_action(actions)

        metadata = {
            "task": self.task_names[video_idx],
            "file_path": str(h5_path),
            "frame_idx": local_idx,
            "video_idx": video_idx,
            "split": self.split,
        }
        return {
            "images": images,
            "actions": actions,
            "metadata": metadata,
        }

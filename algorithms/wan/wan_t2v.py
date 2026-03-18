import logging
import gc
import torch
import numpy as np
import torch.distributed as dist
from einops import rearrange, repeat
from tqdm import tqdm
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from transformers import get_scheduler
import zmq
import msgpack
import io
from PIL import Image
import torchvision.transforms as transforms
from utils.video_utils import numpy_to_mp4_bytes
import peft


from .modules.model import WanModel, WanAttentionBlock
from .modules.t5 import umt5_xxl, T5CrossAttention, T5SelfAttention
from .modules.tokenizers import HuggingfaceTokenizer
from .modules.vae import video_vae_factory
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.distributed_utils import is_rank_zero


def print_module_hierarchy(model, indent=0):
    for name, module in model.named_children():
        print(" " * indent + f"{name}: {type(module)}")
        print_module_hierarchy(module, indent + 2)


class WanTextToVideo(BasePytorchAlgo):
    """
    Main class for WanTextToVideo
    """

    def __init__(self, cfg):
        self.num_train_timesteps = cfg.num_train_timesteps
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.gradient_checkpointing_rate = cfg.gradient_checkpointing_rate
        self.sample_solver = cfg.sample_solver
        self.sample_steps = cfg.sample_steps
        self.sample_shift = cfg.sample_shift
        self.lang_guidance = cfg.lang_guidance
        self.neg_prompt = cfg.neg_prompt
        self.hist_guidance = cfg.hist_guidance
        self.sliding_hist = cfg.sliding_hist
        self.diffusion_forcing = cfg.diffusion_forcing
        self.vae_stride = cfg.vae.stride
        self.patch_size = cfg.model.patch_size
        self.diffusion_type = cfg.diffusion_type  # "discrete"  # or "continuous"

        self.lat_h = self.height // self.vae_stride[1]
        self.lat_w = self.width // self.vae_stride[2]
        self.lat_t = 1 + (self.n_frames - 1) // self.vae_stride[0]
        self.lat_c = cfg.vae.z_dim
        self.max_area = self.height * self.width
        self.max_tokens = (
            self.lat_t
            * self.lat_h
            * self.lat_w
            // (self.patch_size[1] * self.patch_size[2])
        )

        self.load_prompt_embed = cfg.load_prompt_embed
        self.load_video_latent = cfg.load_video_latent
        self.socket = None
        if (self.sliding_hist - 1) % self.vae_stride[0] != 0:
            raise ValueError(
                "sliding_hist - 1 must be a multiple of vae_stride[0] due to temporal "
                f"vae. Got {self.sliding_hist} and vae stride {self.vae_stride[0]}"
            )
        if self.load_video_latent:
            raise NotImplementedError("Loading video latent is not implemented yet")
        super().__init__(cfg)

    @staticmethod
    def classes_to_shard():
        classes = {WanAttentionBlock, T5CrossAttention, T5SelfAttention}  # ,
        return classes

    @property
    def is_inference(self) -> bool:
        # Check for force_training flag in config (added for manual training loops without PL Trainer)
        if self.cfg.get("force_training", False):
            return False
            
        return self._trainer is None or not self.trainer.training


    def configure_model(self):
        logging.info("Building model...")
        target_dtype = torch.bfloat16
        # Initialize text encoder
        if not self.cfg.load_prompt_embed or (self.is_inference and self.lang_guidance > 0):
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=target_dtype,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            if self.cfg.text_encoder.ckpt_path is not None:
                text_encoder.load_state_dict(
                    torch.load(
                        self.cfg.text_encoder.ckpt_path,
                        map_location="cpu",
                        weights_only=True,
                        # mmap=True,
                    )
                )
            if self.cfg.text_encoder.compile:
                text_encoder = torch.compile(text_encoder)
        else:
            text_encoder = None
        self.text_encoder = text_encoder

        # Initialize tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=self.cfg.text_encoder.name,
            seq_len=self.cfg.text_encoder.text_len,
            clean="whitespace",
        )

        # Initialize VAE
        self.vae = (
            video_vae_factory(
                pretrained_path=self.cfg.vae.ckpt_path,
                z_dim=self.cfg.vae.z_dim,
            )
            .eval()
            .requires_grad_(False)
        ).to(target_dtype)
        self.register_buffer(
            "vae_mean", torch.tensor(self.cfg.vae.mean, dtype=target_dtype)
        )
        self.register_buffer(
            "vae_inv_std", 1.0 / torch.tensor(self.cfg.vae.std, dtype=target_dtype)
        )
        self.vae_scale = [self.vae_mean, self.vae_inv_std]
        if self.cfg.vae.compile:
            self.vae = torch.compile(self.vae)

        # Initialize main diffusion model
        use_lora = self.cfg.model.get("use_lora", False)
        if self.cfg.model.tuned_ckpt_path is None:
            self.model = WanModel.from_pretrained(self.cfg.model.ckpt_path)
        elif use_lora:
            # 1. Load Base Model Structure (using original config/weights as init)
            logging.info(f"Loading Base Model structure from {self.cfg.model.ckpt_path}")
            self.model = WanModel.from_pretrained(
                self.cfg.model.ckpt_path, 
                torch_dtype=torch.bfloat16
            )

            # 2. Check and Load Finetuned Weights (.ckpt) if provided
            tuned_ckpt = self.cfg.model.get("tuned_ckpt_path", None)
            if tuned_ckpt and tuned_ckpt.endswith(".ckpt"):
                logging.info(f"Loading Finetuned Weights from {tuned_ckpt}...")
                # Manually loading state dict from .ckpt
                checkpoint = torch.load(tuned_ckpt, map_location="cpu", weights_only=True)
                
                # Filter state dict for 'model.' prefix (standard PL format)
                prefix = "model."
                state_dict = {}
                if "state_dict" in checkpoint:
                    for k, v in checkpoint["state_dict"].items():
                        if k.startswith(prefix):
                            state_dict[k[len(prefix):]] = v
                else:
                    # Fallback if no "state_dict" key (raw save)
                    state_dict = checkpoint

                # Load into model
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                
                del checkpoint, state_dict
                gc.collect()

            # 3. Load LoRA Adapter
            lora_path = self.cfg.model.get("tuned_lora_path", None)
            if lora_path:
                logging.info(f"Loading LoRA Adapter from {lora_path}")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    lora_path, 
                    torch_dtype=torch.bfloat16
                )
            
                if self.is_inference:
                    logging.info("Merging LoRA weights for inference")
                    self.model.merge_adapter()
                    self.model = self.model.unload()
                    self.model.to(torch.bfloat16)

        else:
            self.model = WanModel.from_config(
                WanModel._dict_from_json_file(self.cfg.model.ckpt_path + "/config.json")
            )
            if self.is_inference:
                self.model.to(torch.bfloat16)
            self.model.load_state_dict(
                self._load_tuned_state_dict(), assign=not self.is_inference
            )

        # Disable gradients by default, then re-enable the trainable subset as needed.
        for p in self.model.parameters():
            p.requires_grad_(False)
            
        if not self.is_inference:
            # FSDP requires uniform dtype for all params in a wrapped unit.
            # VAE and TextEncoder are hardcoded to bfloat16 to save memory.
            # Thus, we must initialize the main model in bfloat16 as well.
            self.model.to(torch.bfloat16).train()

            if self.cfg.model.get("use_lora", False):
                from peft import LoraConfig, get_peft_model
                
                lora_rank = self.cfg.model.get("lora_rank", 32)
                lora_alpha = self.cfg.model.get("lora_alpha", 32)
                lora_dropout = self.cfg.model.get("lora_dropout", 0.05)
                target_modules = list(self.cfg.model.get("lora_target_modules", ["q", "k", "v", "o"]))
                
                logging.info(f"Applying LoRA: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}")
                
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
            else:
                # Ensure parameters require gradients (assign=True in load_state_dict may freeze them)
                for p in self.model.parameters():
                    p.requires_grad_(True)

        else:
            self.model.to(torch.bfloat16).eval()

        if self.gradient_checkpointing_rate > 0:
            self.model.gradient_checkpointing_enable(p=self.gradient_checkpointing_rate)

            # Ensure input to the checkpointed part requires gradients.
            if hasattr(self.model, "enable_input_require_grads"):
                # Usually works if get_input_embeddings is implemented
                try:
                    self.model.enable_input_require_grads()
                except (AttributeError, NotImplementedError):
                    # Fallback for WanModel which might not implement get_input_embeddings
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    if hasattr(self.model, "patch_embedding"):
                        self.model.patch_embedding.register_forward_hook(make_inputs_require_grad)
                    else:
                        logging.warning("patch_embedding not found while enabling gradient checkpointing inputs.")
            else:
                # Fallback manual hook.
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                if hasattr(self.model, "patch_embedding"):
                    self.model.patch_embedding.register_forward_hook(make_inputs_require_grad)
                else:
                    self.model.register_forward_hook(make_inputs_require_grad)

        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

        self.training_scheduler, self.training_timesteps = self.build_scheduler(True)

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def _load_tuned_state_dict(self, prefix="model."):
        ckpt = torch.load(
            self.cfg.model.tuned_ckpt_path,
            mmap=True,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = {
            k[len(prefix) :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(prefix)
        }
        del ckpt
        gc.collect()
        return state_dict

    def build_scheduler(self, is_training=True):
        # Solver
        if self.sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                scheduler.set_timesteps(
                    self.sample_steps, device=self.device, shift=self.sample_shift
                )
            timesteps = scheduler.timesteps
        elif self.sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=self.sample_shift,
                use_dynamic_shifting=False,
            )
            if not is_training:
                sampling_sigmas = get_sampling_sigmas(
                    self.sample_steps, self.sample_shift
                )
                timesteps, _ = retrieve_timesteps(
                    scheduler, device=self.device, sigmas=sampling_sigmas
                )
        else:
            raise NotImplementedError("Unsupported solver.")
        return scheduler, timesteps

    def encode_text(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        # Ensure inputs match text_encoder device
        enc_device = next(self.text_encoder.parameters()).device
        ids = ids.to(enc_device)
        mask = mask.to(enc_device)
        
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        
        # Move output back to model device if needed, but context is usually used immediately.
        # However, to be safe for subsequent model calls:
        if context.device != self.device:
             context = context.to(self.device)
             
        return [u[:v] for u, v in zip(context, seq_lens)]

    @staticmethod
    def pad_text_context_to_tensor(context_list, text_len: int):
        """
        context_list: List[Tensor], each [L_i, C]
        return: Tensor [B, text_len, C]
        """
        assert isinstance(context_list, (list, tuple))
        assert len(context_list) > 0
        device = context_list[0].device
        dtype = context_list[0].dtype
        C = context_list[0].shape[-1]

        out = []
        for u in context_list:
            if u.dim() != 2:
                raise ValueError(f"each context must be [L, C], got {u.shape}")
            L, C_u = u.shape
            if C_u != C:
                raise ValueError(f"context dim mismatch: {C_u} vs {C}")

            if L < text_len:
                pad = u.new_zeros((text_len - L, C))
                u2 = torch.cat([u, pad], dim=0)
            else:
                u2 = u[:text_len]
            out.append(u2)

        return torch.stack(out, dim=0).to(device=device, dtype=dtype)  # [B, text_len, C]


    def encode_video(self, videos):
        """videos: [B, C, T, H, W]"""
        # Ensure videos match VAE dtype (likely bfloat16)
        if hasattr(self, 'vae') and hasattr(self.vae, 'parameters'):
             target_dtype = next(self.vae.parameters()).dtype
             videos = videos.to(dtype=target_dtype)
        return self.vae.encode(videos, self.vae_scale)

    def decode_video(self, zs):
        # Ensure scale is in the same dtype as zs to avoid implicit casting to float32 if scale is float32
        scale = [s.to(zs.dtype) for s in self.vae_scale]
        return self.vae.decode(zs, scale).clamp_(-1, 1)

    def clone_batch(self, batch):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.clone()
            else:
                new_batch[k] = v
        return new_batch

    @torch.no_grad()
    def prepare_embeds(self, batch):
        videos = batch["videos"]
        prompts = batch["prompts"]

        batch_size, t, _, h, w = videos.shape

        if t != self.n_frames:
            raise ValueError(f"Number of frames in videos must be {self.n_frames}")
        if h != self.height or w != self.width:
            raise ValueError(
                f"Height and width of videos must be {self.height} and {self.width}"
            )

        if not self.cfg.load_prompt_embed:
            prompt_embeds = self.encode_text(prompts)
        else:
            prompt_embeds = batch["prompt_embeds"].to(self.dtype)
            prompt_embed_len = batch["prompt_embed_len"]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, prompt_embed_len)]

        video_lat = self.encode_video(rearrange(videos, "b t c h w -> b c t h w"))
        # video_lat ~ (b, lat_c, lat_t, lat_h, lat_w

        batch["prompt_embeds"] = prompt_embeds
        batch["video_lat"] = video_lat
        batch["image_embeds"] = None
        batch["clip_embeds"] = None

        return batch

    def add_training_noise(self, video_lat):
        b, _, f = video_lat.shape[:3]
        device = video_lat.device
        if self.diffusion_type == "discrete":
            video_lat = rearrange(video_lat, "b c f h w -> (b f) c h w")
            noise = torch.randn_like(video_lat)
            timesteps = self.num_train_timesteps
            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = np.random.randint(timesteps, size=(b, f))
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = timesteps - 1
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 2, 3, 4, 5, 6]
                        hist_length_probs = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
                        t = np.zeros((b, f), dtype=np.int64)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.randint(timesteps)
                            future_t = np.random.randint(timesteps)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = timesteps - 1
                t = self.training_timesteps[t.flatten()].reshape(b, f)
                t_expanded = t.flatten()
            else:
                t = np.random.randint(timesteps, size=(b,))
                t_expanded = repeat(t, "b -> (b f)", f=f)
                t = self.training_timesteps[t]
                t_expanded = self.training_timesteps[t_expanded]

            noisy_lat = self.training_scheduler.add_noise(video_lat, noise, t_expanded)
            noisy_lat = rearrange(noisy_lat, "(b f) c h w -> b c f h w", b=b, f=f)
            noise = rearrange(noise, "(b f) c h w -> b c f h w", b=b, f=f)
        elif self.diffusion_type == "continuous":
            # continious time steps.
            # 1. first sample t ~ U[0, 1]
            # 2. shift t with equation: t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
            # 3. expand t to [b, 1/f, 1, 1, 1]
            # 4. compute noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
            # 5. scale t to [0, num_train_timesteps]
            # returns:
            #  t is in [0, num_train_timesteps] of shape [b, f] or [b,], of dtype torch.float32
            # video_lat is shape [b, c, f, h, w]
            # noise is shape [b, c, f, h, w]
            dist = torch.distributions.uniform.Uniform(0, 1)
            noise = torch.randn_like(video_lat)  # [b, c, f, h, w]

            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = dist.sample((b, f)).to(device)
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = 0.0
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = [1, 2]
                        hist_length_probs = [0.9, 0.1]
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            hist_len_idx = np.random.choice(
                                len(possible_hist_lengths), p=hist_length_probs
                            )
                            hist_len = possible_hist_lengths[hist_len_idx]
                            history_t = np.random.uniform(0, 1)
                            future_t = np.random.uniform(0, 1)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = 0

                        # cast dtype of t
                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                # t is [b, f] in range [0, 1] or dtype torch.float32  0 indicates clean.
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = (
                    t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # [b, f] -> [b, 1, f, 1, 1]

                # [b, c, f, h, w] * [b, 1, f, 1, 1] + [b, c, f, h, w] * [b, 1, f, 1, 1]
                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b, f] -> [b, f]
                # now t is in [0, num_train_timesteps] of shape [b, f]
            else:
                t = dist.sample((b,)).to(device)
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = t.view(-1, 1, 1, 1, 1)

                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b,]
                # now t is in [0, num_train_timesteps] of shape [b,]
        else:
            raise NotImplementedError("Unsupported time step type.")

        return noisy_lat, noise, t

    def remove_noise(self, flow_pred, t, video_pred_lat):
        b, _, f = video_pred_lat.shape[:3]
        video_pred_lat = rearrange(video_pred_lat, "b c f h w -> (b f) c h w")
        flow_pred = rearrange(flow_pred, "b c f h w -> (b f) c h w")
        if t.ndim == 1:
            t = repeat(t, "b -> (b f)", f=f)
        elif t.ndim == 2:
            t = t.flatten()
        video_pred_lat = self.inference_scheduler.step(
            flow_pred,
            t,
            video_pred_lat,
            return_dict=False,
        )[0]
        video_pred_lat = rearrange(video_pred_lat, "(b f) c h w -> b c f h w", b=b)
        return video_pred_lat

    def training_step(self, batch, batch_idx=None):
        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]

        noisy_lat, noise, t = self.add_training_noise(video_lat)
        flow = noise - video_lat

        # Ensure inputs are in the correct dtype (bfloat16) to match model weights
        # This is necessary when upstream encoders (Text/CLIP) might be in float32 
        # but the model is in bfloat16 (e.g. via FSDP mixed precision or LoRA)
        target_dtype = torch.bfloat16
        
        if noisy_lat.dtype != target_dtype:
            noisy_lat = noisy_lat.to(target_dtype)

        if clip_embeds is not None and clip_embeds.dtype != target_dtype:
            clip_embeds = clip_embeds.to(target_dtype)
        if image_embeds is not None and image_embeds.dtype != target_dtype:
            image_embeds = image_embeds.to(target_dtype)
        if prompt_embeds is not None:
             prompt_embeds = [u.to(target_dtype) if u.dtype != target_dtype else u for u in prompt_embeds]

        flow_pred = self.model(
            noisy_lat,
            t=t,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=self.max_tokens,
            y=image_embeds,
        )
        
        # Cast target to the same dtype as the prediction to ensure correct gradient types for DeepSpeed
        loss = torch.nn.functional.mse_loss(flow_pred, flow.to(flow_pred.dtype))

        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        """
        Filter checkpoint to only save LoRA parameters if LoRA is enabled.
        This significantly reduces checkpoint size.
        """
        if self.cfg.model.get("use_lora", False):
            # Filter the existing state_dict to keep only trainable params (LoRA)
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if "lora_" in k or "modules_to_save" in k or ".lora" in k:
                    new_state_dict[k] = v
            checkpoint["state_dict"] = new_state_dict

    @torch.no_grad()
    def sample_seq(self, batch, hist_len=1, pbar=None):
        """
        Main sampling loop. Only first hist_len frames are used for conditioning
        batch: dict
            batch["videos"]: [B, T, C, H, W]
            batch["prompts"]: [B]
        """
        if (hist_len - 1) % self.vae_stride[0] != 0:
            raise ValueError(
                "hist_len - 1 must be a multiple of vae_stride[0] due to temporal vae. "
                f"Got {hist_len} and vae stride {self.vae_stride[0]}"
            )
        hist_len = (hist_len - 1) // self.vae_stride[0] + 1  #  length in latent

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)
        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0

        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]

        # Ensure all tensors are compatible with the model's dtype
        target_dtype = self.model.patch_embedding.weight.dtype
        if video_lat.dtype != target_dtype:
             video_lat = video_lat.to(target_dtype)
        if clip_embeds is not None and clip_embeds.dtype != target_dtype:
             clip_embeds = clip_embeds.to(target_dtype)
        if image_embeds is not None and image_embeds.dtype != target_dtype:
             image_embeds = image_embeds.to(target_dtype)
        if prompt_embeds is not None:
             prompt_embeds = [u.to(target_dtype) if u.dtype != target_dtype else u for u in prompt_embeds]

        batch_size = video_lat.shape[0]

        video_pred_lat = torch.randn_like(video_lat)
        if self.lang_guidance:
            neg_prompt_embeds = self.encode_text(
                [self.neg_prompt] * len(batch["prompts"])
            )
        if pbar is None:
            pbar = tqdm(range(len(self.inference_timesteps)), desc="Sampling")
        for t in self.inference_timesteps:
            if self.diffusion_forcing.enabled:
                video_pred_lat[:, :, :hist_len] = video_lat[:, :, :hist_len]
                t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
                t_expanded[:, :hist_len] = self.inference_timesteps[-1]
            else:
                t_expanded = torch.full((batch_size,), t, device=self.device)

            # normal conditional sampling
            flow_pred = self.model(
                video_pred_lat,
                t=t_expanded,
                context=prompt_embeds,
                seq_len=self.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
            )

            # language unconditional sampling
            if lang_guidance:
                no_lang_flow_pred = self.model(
                    video_pred_lat,
                    t=t_expanded,
                    context=neg_prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                )
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            # history guidance sampling:
            if hist_guidance and self.diffusion_forcing.enabled:
                no_hist_video_pred_lat = video_pred_lat.clone()
                no_hist_video_pred_lat[:, :, :hist_len] = torch.randn_like(
                    no_hist_video_pred_lat[:, :, :hist_len]
                )
                t_expanded[:, :hist_len] = self.inference_timesteps[0]
                no_hist_flow_pred = self.model(
                    no_hist_video_pred_lat,
                    t=t_expanded,
                    context=prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                )
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = flow_pred * (1 + lang_guidance + hist_guidance)
            flow_pred = (
                flow_pred
                - lang_guidance * no_lang_flow_pred
                - hist_guidance * no_hist_flow_pred
            )

            video_pred_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            pbar.update(1)

        video_pred_lat[:, :, :hist_len] = video_lat[:, :, :hist_len]

        # Cast latents to VAE's dtype before decoding
        # Because diffusion sampling might produce float32 (or bfloat16), but VAE weights are strictly typed.
        # We check the actual dtype of the VAE parameters to be safe (vae_mean buffer might lag behind in float32)
        vae_dtype = next(self.vae.parameters()).dtype
        if video_pred_lat.dtype != vae_dtype:
            video_pred_lat = video_pred_lat.to(vae_dtype)

        video_pred = self.decode_video(video_pred_lat)
        video_pred = rearrange(video_pred, "b c t h w -> b t c h w")

        return video_pred

    def validation_step(self, batch, batch_idx=None):
        video_pred = self.sample_seq(batch)
        self.visualize(video_pred, batch, batch_idx=batch_idx)

    def visualize(self, video_pred, batch, batch_idx=0):
        video_gt = batch["videos"]

        if self.cfg.logging.video_type == "single":
            video_vis = video_pred
        else:
            # Modified: GT on top, Pred on bottom
            video_vis = torch.cat([video_gt, video_pred], dim=-2)
        
        video_vis = video_vis * 0.5 + 0.5

        if dist.is_available() and dist.is_initialized():
            # Keep on device for NCCL all_gather to avoid timeouts/errors with CPU tensors
            video_vis = self.all_gather(video_vis)
            video_vis = rearrange(video_vis, "p b ... -> (p b) ...").cpu()

            all_prompts = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_prompts, batch["prompts"])
            all_prompts = [item for sublist in all_prompts for item in sublist]
        else:
            video_vis = video_vis.cpu()
            all_prompts = list(batch["prompts"])

        if is_rank_zero:
            if self.cfg.logging.video_type == "single":
                for i in range(min(len(video_vis), 16)):
                    self.log_video(
                        f"validation_vis/video_pred_{i}",
                        video_vis[i],
                        fps=self.cfg.logging.fps,
                        caption=all_prompts[i],
                    )
            else:
                self.log_video(
                    "validation_vis/video_pred",
                    video_vis[:16],
                    fps=self.cfg.logging.fps,
                    step=self.global_step,
                )

    def maybe_reset_socket(self):
        if not self.socket:
            ctx = zmq.Context()
            socket = ctx.socket(zmq.ROUTER)
            socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
            socket.bind(f"tcp://*:{self.cfg.serving.port}")
            self.socket = socket

            print(f"Server ready on port {self.cfg.serving.port}...")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """
        This function is used to test the model.
        It will receive an image and a prompt from remote gradio and generate a video.
        The remote client shall run scripts/inference_client.py to send requests to this server.
        """

        # Only rank zero sets up the socket
        if is_rank_zero:
            self.maybe_reset_socket()

        print(f"Waiting for request on local rank: {dist.get_rank()}")
        if is_rank_zero:
            ident, payload = self.socket.recv_multipart()
            request = msgpack.unpackb(payload, raw=False)
            print(f"Received request with prompt: {request['prompt']}")

            # Prepare data to broadcast
            image_bytes = request["image"]
            prompt = request["prompt"]
            data_to_broadcast = [image_bytes, prompt]
        else:
            data_to_broadcast = [None, None]

        # Broadcast the image and prompt to all ranks
        dist.broadcast_object_list(data_to_broadcast, src=0)
        image_bytes, prompt = data_to_broadcast
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=(1.0, 1.0),
                    ratio=(self.width / self.height, self.width / self.height),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            ]
        )
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(pil_image)
        batch["videos"][:, 0] = image[None]

        prompt_segments = prompt.split("<sep>")
        hist_len = 1
        videos = batch["videos"][:, :hist_len]
        for i, prompt in enumerate(prompt_segments):
            # extending the video until all prompt segments are used
            print(f"Generating task {i+1} out of {len(prompt_segments)} sub-tasks")
            batch["prompts"] = [prompt] * batch["videos"].shape[0]
            batch["videos"][:, :hist_len] = videos[:, -hist_len:]
            videos = torch.cat([videos, self.sample_seq(batch, hist_len)], dim=1)
            videos = torch.clamp(videos, -1, 1)
            hist_len = self.sliding_hist
        videos = rearrange(self.all_gather(videos), "p b t c h w -> (p b) t h w c")
        videos = videos.float().cpu().numpy()

        # Only rank zero sends the reply
        if is_rank_zero:
            videos = np.clip(videos * 0.5 + 0.5, 0, 1)
            videos = (videos * 255).astype(np.uint8)
            # Convert videos to mp4 bytes using the utility function
            video_bytes_list = [
                numpy_to_mp4_bytes(video, fps=self.cfg.logging.fps) for video in videos
            ]

            # Send the reply
            reply = {"videos": video_bytes_list}
            self.socket.send_multipart([ident, msgpack.packb(reply)])
            print(f"Sent reply to {ident}")

            self.log_video(
                "test_vis/video_pred",
                rearrange(videos, "b t h w c -> b t c h w"),
                fps=self.cfg.logging.fps,
                caption="<sep>\n".join(prompt_segments),
            )

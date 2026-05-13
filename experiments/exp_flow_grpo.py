import json
import os
import tempfile
import time
from collections import defaultdict
from concurrent import futures
import imageio
import numpy as np
import peft
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import FullyShardedDataParallelPlugin, set_seed
from diffusers import ModelMixin
from omegaconf import OmegaConf
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .exp_base import BasePytorchExperiment
from algorithms.wan import WanImageToVideo, WanTextToVideo
from algorithms.wan.modules.model import WanAttentionBlock
from datasets.robotwin import RobotwinDataset
from flow_grpo import EMAModuleWrapper, PerPromptStatTracker, idm_smoothness_score
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import sde_step_with_logprob


def _get_tied_weights_keys_peft(self):
    if hasattr(self, "base_model"):
        return getattr(self.base_model, "_tied_weights_keys", set())
    return set()


def _get_tied_weights_keys_lora(self):
    if hasattr(self, "model"):
        return getattr(self.model, "_tied_weights_keys", set())
    return set()


if not hasattr(peft.peft_model.PeftModel, "_tied_weights_keys"):
    peft.peft_model.PeftModel._tied_weights_keys = property(_get_tied_weights_keys_peft)

if hasattr(peft.tuners.lora, "LoraModel") and not hasattr(
    peft.tuners.lora.LoraModel, "_tied_weights_keys"
):
    peft.tuners.lora.LoraModel._tied_weights_keys = property(
        _get_tied_weights_keys_lora
    )

if not hasattr(ModelMixin, "_tied_weights_keys"):
    ModelMixin._tied_weights_keys = property(lambda self: set())


logger = get_logger(__name__)


def _to_device(batch, device):
    if isinstance(batch, dict):
        out = {}
        for key, value in batch.items():
            out[key] = value.to(device) if torch.is_tensor(value) else value
        return out
    return batch


def _maybe_detach_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _log_metrics(accelerator: Accelerator, metrics: dict, step: int) -> None:
    if getattr(accelerator, "trackers", None):
        accelerator.log(metrics, step=step)


def compute_grpo_loss_per_timestep(new_logp, old_logp, adv, clip_range: float):
    ratio = torch.exp(new_logp - old_logp)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.mean(torch.maximum(unclipped, clipped))
    return loss, ratio


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"Total batch ({self.total_samples}) must be divisible by k ({k})"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        internal_step = 0
        while True:
            generator = torch.Generator()
            current_seed = (self.seed + self.epoch + internal_step) % (2**32)
            generator.manual_seed(current_seed)

            if self.m <= len(self.dataset):
                indices = torch.randperm(len(self.dataset), generator=generator)[
                    : self.m
                ].tolist()
            else:
                indices = torch.randint(
                    0, len(self.dataset), (self.m,), generator=generator
                ).tolist()

            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=generator
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield [(current_seed, idx) for idx in per_card_samples[self.rank]]
            internal_step += 1

    def set_epoch(self, epoch):
        self.epoch = epoch


class FlowGRPOTrainer:
    def __init__(self, wan_algo, cfg, accelerator: Accelerator):
        self.wan_algo = wan_algo
        self.cfg = cfg
        self.accelerator = accelerator
        self.stat_tracker = None
        if cfg.per_prompt_stat_tracking:
            self.stat_tracker = PerPromptStatTracker(cfg.sample.global_std)

    @torch.no_grad()
    def rollout_batch(
        self,
        batch,
        scheduler,
        timesteps,
        neg_prompt_embeds=None,
        determistic: bool = False,
    ):
        device = self.accelerator.device
        batch_size = len(batch["prompt_embeds"])
        video_lat = batch["video_lat"]
        latents = torch.randn_like(video_lat)

        hist_len = 1
        traj = [latents.clone()]
        old_logps = []
        kls = []

        for timestep in tqdm(
            timesteps,
            desc="Rollout Steps",
            leave=False,
            position=1,
            disable=not self.accelerator.is_local_main_process,
        ):
            if self.wan_algo.diffusion_forcing.enabled:
                latents[:, :, :hist_len] = video_lat[:, :, :hist_len]

            t_vec = self.wan_algo.construct_diffusion_forcing_timesteps(
                batch_size=batch_size,
                t_curr=timestep,
                device=device,
                clean_t=timesteps[-1],
                hist_len=hist_len,
            )

            prev_latents, logp, _, _, _ = self.wan_algo.sample_step_with_logprob(
                scheduler=scheduler,
                latents=latents,
                t_vec=t_vec,
                prompt_embeds=batch["prompt_embeds"],
                negative_prompt_embeds=neg_prompt_embeds,
                guidance_scale=self.cfg.sample.guidance_scale,
                do_cfg=self.cfg.train.cfg,
                clip_fea=batch.get("clip_embeds", None),
                y=batch.get("image_embeds", None),
                determistic=determistic,
            )

            traj.append(prev_latents)
            old_logps.append(logp)
            kls.append(torch.zeros_like(logp))
            latents = prev_latents.detach().clone()

        latents_traj = torch.stack(traj, dim=1)
        old_log_probs = torch.stack(old_logps, dim=1)
        kl = torch.stack(kls, dim=1)
        final_latents = latents_traj[:, -1]
        videos = self.wan_algo.decode_video(final_latents.to(self.wan_algo._dtype))
        return videos, latents_traj, old_log_probs, kl

    def compute_advantages(self, prompts_gathered, rewards_gathered):
        if self.cfg.per_prompt_stat_tracking and prompts_gathered is not None:
            advantages = self.stat_tracker.update(prompts_gathered, rewards_gathered)
            group_size, trained_prompt_num = self.stat_tracker.get_stats()
            self.stat_tracker.clear()
            return advantages, group_size, trained_prompt_num

        mean = rewards_gathered.mean()
        std = rewards_gathered.std()
        adv = (rewards_gathered - mean) / (std + 1e-4)
        return adv, None, None

    def train_on_samples(self, samples, scheduler, timesteps, optimizer, global_step, ema=None):
        accelerator = self.accelerator
        cfg = self.cfg
        model = self.wan_algo.model

        optimizer.zero_grad()
        total_batch_size = samples["latents"].shape[0]
        num_train_timesteps = int(cfg.sample.num_steps * cfg.train.timestep_fraction)
        train_timesteps_val = timesteps[:num_train_timesteps]

        perm = torch.randperm(total_batch_size, device=accelerator.device)
        samples = {k: v[perm] if v is not None else None for k, v in samples.items()}

        micro_batch_size = cfg.sample.train_batch_size
        model.train()
        info_acc = defaultdict(list)

        micro_batch_iter = range(0, total_batch_size, micro_batch_size)
        if accelerator.is_local_main_process:
            micro_batch_iter = tqdm(micro_batch_iter, desc="Micro-Batch", leave=False)

        for i in micro_batch_iter:
            batch_slice = slice(i, min(i + micro_batch_size, total_batch_size))
            current_bs = samples["latents"][batch_slice].shape[0]

            prompt_embeds = samples["prompt_embeds"][batch_slice]
            negative_prompt_embeds = (
                samples["negative_prompt_embeds"][batch_slice]
                if samples.get("negative_prompt_embeds") is not None
                else None
            )
            clip_fea = (
                samples["clip_fea"][batch_slice]
                if samples.get("clip_fea") is not None
                else None
            )
            image_embeds = (
                samples["y"][batch_slice] if samples.get("y") is not None else None
            )
            gt_latents = samples["gt_latents"][batch_slice].to(accelerator.device)
            hist_len = 1

            timesteps_iter = enumerate(train_timesteps_val)
            if accelerator.is_local_main_process:
                timesteps_iter = tqdm(
                    timesteps_iter,
                    desc="Timestep",
                    total=len(train_timesteps_val),
                    leave=False,
                )

            for t_idx, timestep in timesteps_iter:
                latents = samples["latents"][batch_slice, t_idx]
                next_latents = samples["next_latents"][batch_slice, t_idx]
                old_logp = samples["old_log_probs"][batch_slice, t_idx]
                advantage = samples["advantages"][batch_slice, t_idx]

                if self.wan_algo.diffusion_forcing.enabled:
                    latents[:, :, :hist_len] = gt_latents[:, :, :hist_len]

                t_vec = self.wan_algo.construct_diffusion_forcing_timesteps(
                    batch_size=current_bs,
                    t_curr=timestep,
                    device=accelerator.device,
                    clean_t=timesteps[-1],
                    hist_len=hist_len,
                )

                latents.requires_grad_(True)
                with accelerator.accumulate(model):
                    _, new_logp, new_mean, new_std, new_dt = self.wan_algo.logprob_of_transition(
                        scheduler=scheduler,
                        latents=latents,
                        next_latents=next_latents,
                        t_vec=t_vec,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        guidance_scale=cfg.sample.guidance_scale,
                        do_cfg=cfg.train.cfg,
                        clip_fea=clip_fea,
                        y=image_embeds,
                    )

                    advantage = torch.clamp(
                        advantage,
                        -cfg.train.adv_clip_max,
                        cfg.train.adv_clip_max,
                    )
                    policy_loss, ratio = compute_grpo_loss_per_timestep(
                        new_logp, old_logp, advantage, clip_range=cfg.train.clip_range
                    )

                    loss = policy_loss
                    if cfg.train.beta > 0 and hasattr(model, "disable_adapter"):
                        with torch.no_grad():
                            with model.disable_adapter():
                                _, _, ref_mean, _, ref_dt = self.wan_algo.logprob_of_transition(
                                    scheduler=scheduler,
                                    latents=latents,
                                    next_latents=next_latents,
                                    t_vec=t_vec,
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    guidance_scale=cfg.sample.guidance_scale,
                                    do_cfg=cfg.train.cfg,
                                    clip_fea=clip_fea,
                                    y=image_embeds,
                                )
                        denom = (new_std * ref_dt).pow(2) * 2.0
                        kl_term = ((new_mean - ref_mean).pow(2)).mean() / (
                            denom.mean() + 1e-12
                        )
                        loss = policy_loss + cfg.train.beta * kl_term
                    else:
                        kl_term = torch.tensor(0.0, device=accelerator.device)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            cfg.train.max_grad_norm,
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                        info_acc["policy_loss"].append(policy_loss.detach())
                        info_acc["loss"].append(loss.detach())
                        info_acc["kl_loss"].append(kl_term.detach())
                        info_acc["clipfrac"].append(
                            (torch.abs(ratio.detach() - 1.0) > cfg.train.clip_range)
                            .float()
                            .mean()
                        )
                        info_acc["approx_kl"].append(
                            0.5 * ((new_logp.detach() - old_logp) ** 2).mean()
                        )

                        if cfg.train.ema and ema is not None and accelerator.is_main_process:
                            ema.step(
                                [p for p in model.parameters() if p.requires_grad],
                                global_step,
                            )

        return {k: torch.stack(v).mean() for k, v in info_acc.items()}


class FlowGRPOExperiment(BasePytorchExperiment):
    compatible_algorithms = dict(
        wan_i2v=WanImageToVideo,
        wan_t2v=WanTextToVideo,
    )

    compatible_datasets = dict(
        robotwin=RobotwinDataset,
    )

    def _videos_to_uint8(self, videos: torch.Tensor) -> np.ndarray:
        videos = videos.detach().float().cpu()
        if videos.min() < 0:
            videos = (videos + 1.0) / 2.0
        videos = videos.clamp(0, 1)
        videos = (videos * 255).round().to(torch.uint8)
        return videos.permute(0, 2, 3, 4, 1).numpy()

    @torch.no_grad()
    def _run_eval(
        self,
        trainer,
        wan_algo,
        eval_loader,
        reward_fn,
        cfg,
        accelerator: Accelerator,
        global_step: int,
        eval_num_batches: int,
        eval_num_videos: int,
        eval_fps: int,
        epoch: int = 0,
    ):
        if eval_loader is None:
            return

        wan_algo.model.eval()
        all_rewards = defaultdict(list)
        logged_videos = None
        logged_prompts = None
        logged_rewards = None

        eval_neg_prompt_embeds = None
        if cfg.train.cfg:
            eval_bs = cfg.sample.get("eval_batch_size", cfg.sample.sample_batch_size)
            eval_neg_prompt_embeds = wan_algo.get_negative_prompt_embeds_cached(
                batch_size=eval_bs,
                device=accelerator.device,
            )

        for step, batch in enumerate(
            tqdm(eval_loader, desc="Eval", disable=not accelerator.is_local_main_process)
        ):
            if eval_num_batches is not None and step >= eval_num_batches:
                break

            batch = _to_device(batch, accelerator.device)
            batch = wan_algo.prepare_embeds(batch)
            if isinstance(batch.get("prompt_embeds", None), (list, tuple)):
                text_len = accelerator.unwrap_model(wan_algo.model).text_len
                batch["prompt_embeds"] = WanTextToVideo.pad_text_context_to_tensor(
                    batch["prompt_embeds"], text_len
                )

            scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
            eval_num_steps = cfg.sample.get("eval_num_steps", cfg.sample.num_steps)
            timesteps = timesteps[:eval_num_steps]

            videos, _, _, _ = trainer.rollout_batch(
                batch,
                scheduler,
                timesteps,
                neg_prompt_embeds=eval_neg_prompt_embeds,
                determistic=True,
            )

            rewards_dict, _ = reward_fn(
                videos,
                batch.get("prompts", None),
                [
                    {
                        "phase": "eval",
                        "step": global_step,
                        "epoch": epoch,
                        "rank": accelerator.process_index,
                    }
                    for _ in range(len(videos))
                ],
            )
            for key, value in rewards_dict.items():
                value_tensor = torch.as_tensor(
                    value,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                all_rewards[key].append(accelerator.gather(value_tensor).cpu().numpy())

            vis_dir = os.path.join(
                self.output_dir,
                "visualizations",
                "eval",
                f"epoch_{epoch}_step_{global_step}",
            )
            os.makedirs(vis_dir, exist_ok=True)
            videos_uint8 = self._videos_to_uint8(videos)
            for idx, frames in enumerate(videos_uint8):
                save_path = os.path.join(
                    vis_dir,
                    f"rank_{accelerator.process_index}_batch_{step}_vid_{idx}.mp4",
                )
                imageio.mimsave(
                    save_path,
                    frames,
                    fps=eval_fps,
                    codec="libx264",
                    format="FFMPEG",
                )

            if logged_videos is None:
                logged_videos = videos.detach()
                logged_prompts = batch.get("prompts", None)
                logged_rewards = rewards_dict

        if accelerator.is_main_process and logged_videos is not None:
            videos_uint8 = self._videos_to_uint8(logged_videos)
            num_samples = min(eval_num_videos, len(videos_uint8))
            with tempfile.TemporaryDirectory() as tmpdir:
                captions = []
                for idx in range(num_samples):
                    imageio.mimsave(
                        os.path.join(tmpdir, f"{idx}.mp4"),
                        [frame for frame in videos_uint8[idx]],
                        fps=eval_fps,
                        codec="libx264",
                        format="FFMPEG",
                    )
                    prompt = (
                        logged_prompts[idx]
                        if isinstance(logged_prompts, (list, tuple))
                        else ""
                    )
                    reward_str = ""
                    if isinstance(logged_rewards, dict):
                        reward_str = " " + " | ".join(
                            f"{k}: {float(logged_rewards[k][idx]):.2f}"
                            for k in logged_rewards
                        )
                    captions.append((prompt + reward_str).strip())

                _log_metrics(
                    accelerator,
                    {
                        "eval/videos": [
                            wandb.Video(
                                os.path.join(tmpdir, f"{idx}.mp4"),
                                caption=captions[idx],
                                format="mp4",
                                fps=eval_fps,
                            )
                            for idx in range(num_samples)
                        ]
                    },
                    step=global_step,
                )

        if all_rewards:
            all_rewards = {k: np.concatenate(v) for k, v in all_rewards.items()}
            if accelerator.is_main_process:
                _log_metrics(
                    accelerator,
                    {f"eval/reward_{k}": float(np.mean(v)) for k, v in all_rewards.items()},
                    step=global_step,
                )

        wan_algo.model.train()

    def _save_checkpoint(
        self,
        accelerator: Accelerator,
        save_root: str,
        ema,
        global_step: int,
        epoch: int,
        wan_algo=None,
        reward_info=None,
    ):
        if accelerator.is_main_process:
            os.makedirs(save_root, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(save_root)

        if accelerator.is_main_process and ema is not None:
            torch.save(
                {
                    "ema": ema.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                },
                os.path.join(save_root, "ema.pt"),
            )

        if accelerator.is_main_process and reward_info is not None:
            reward_info = {
                k: (v.item() if hasattr(v, "item") else v) for k, v in reward_info.items()
            }
            torch.save(reward_info, os.path.join(save_root, "reward_info.pt"))
            with open(os.path.join(save_root, "reward_info.json"), "w") as handle:
                json.dump(reward_info, handle, indent=2)

        if wan_algo is not None:
            full_state_dict = accelerator.get_state_dict(wan_algo.model)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(wan_algo.model)
                save_func = getattr(unwrapped_model, "save_pretrained", None)
                if save_func is not None:
                    lora_dir = os.path.join(save_root, "lora_adapter")
                    save_func(lora_dir, state_dict=full_state_dict)
                    logger.info("LoRA adapter saved to %s", lora_dir)
                else:
                    logger.warning(
                        "Model does not provide save_pretrained; skipping separate LoRA save."
                    )

    def training(self):
        cfg = self.cfg.training

        if self.cfg.reward.get("idm_smoothness", False) and not cfg.get(
            "reward_model_path", None
        ):
            raise ValueError(
                "experiment.training.reward_model_path is required when "
                "reward.idm_smoothness=true."
            )

        fsdp_plugin = None
        if cfg.get("strategy") == "fsdp":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=ModuleWrapPolicy([WanAttentionBlock]),
                mixed_precision_policy=mp_policy,
                use_orig_params=True,
            )

        log_with = "wandb" if self.root_cfg.wandb.mode != "disabled" else None
        accumulation_steps = cfg.train.gradient_accumulation_steps * int(
            cfg.sample.num_steps * cfg.train.timestep_fraction
        )
        accelerator = Accelerator(
            log_with=log_with,
            mixed_precision=cfg.get("precision", "bf16"),
            gradient_accumulation_steps=accumulation_steps,
            fsdp_plugin=fsdp_plugin,
        )

        set_seed(cfg.seed, device_specific=True)

        if accelerator.is_main_process and log_with is not None:
            accelerator.init_trackers(
                self.root_cfg.wandb.project,
                config=OmegaConf.to_container(self.root_cfg, resolve=True),
            )

        OmegaConf.set_struct(self.root_cfg.algorithm, False)
        self.root_cfg.algorithm.force_training = True
        OmegaConf.set_struct(self.root_cfg.algorithm, True)

        self._build_algo()
        self.algo._device = accelerator.device
        self.algo._dtype = (
            torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16
        )
        # This GRPO path uses a manual Accelerate loop, so Lightning will not invoke
        # configure_model() for us. Build Wan modules explicitly before accessing
        # wan_algo.model / vae / text_encoder / clip.
        if not hasattr(self.algo, "model"):
            self.algo.configure_model()

        wan_algo = WanAlgoWrapper(self.algo)

        if hasattr(wan_algo, "vae"):
            wan_algo.vae.to(accelerator.device)
            wan_algo.vae.requires_grad_(False)
        if hasattr(wan_algo, "text_encoder") and wan_algo.text_encoder is not None:
            wan_algo.text_encoder.to(accelerator.device)
            if not wan_algo.cfg.load_prompt_embed:
                wan_algo.text_encoder.requires_grad_(False)
        if hasattr(wan_algo, "clip") and wan_algo.clip is not None:
            wan_algo.clip.to(accelerator.device)
            wan_algo.clip.requires_grad_(False)
        if hasattr(wan_algo.algo, "vae_scale"):
            wan_algo.algo.vae_scale = [
                scale.to(device=accelerator.device, dtype=wan_algo.algo._dtype)
                for scale in wan_algo.algo.vae_scale
            ]

        transformer_trainable_parameters = [
            p for p in wan_algo.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            transformer_trainable_parameters,
            lr=cfg.lr,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
            weight_decay=cfg.get("adam_weight_decay", 1e-4),
            eps=cfg.get("adam_epsilon", 1e-8),
        )

        train_dataset = self._build_dataset("training")
        eval_dataset = self._build_dataset("validation")
        sampler = DistributedKRepeatSampler(
            train_dataset,
            batch_size=cfg.sample.sample_batch_size,
            k=cfg.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=cfg.seed,
        )

        num_workers = min(os.cpu_count() or cfg.data.num_workers, cfg.data.num_workers)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        eval_batch_size = cfg.sample.get("eval_batch_size", self.cfg.validation.batch_size)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

        wan_algo.model, optimizer, eval_loader = accelerator.prepare(
            wan_algo.model,
            optimizer,
            eval_loader,
        )

        reward_save_dir = os.path.join(self.output_dir, "idm_plots")
        os.makedirs(reward_save_dir, exist_ok=True)
        reward_fn = idm_smoothness_score(
            device=accelerator.device,
            checkpoint_path=cfg.reward_model_path,
            num_frames=cfg.idm_num_frames,
            save_dir=reward_save_dir,
        )

        ema = None
        if cfg.train.ema and accelerator.is_main_process:
            ema = EMAModuleWrapper(
                [p for p in wan_algo.model.parameters() if p.requires_grad],
                decay=cfg.train.ema_decay,
                update_step_interval=cfg.train.ema_interval,
                device=accelerator.device,
            )

        trainer = FlowGRPOTrainer(wan_algo, cfg, accelerator)
        neg_prompt_embeds = None
        if cfg.train.cfg:
            neg_prompt_embeds = wan_algo.get_negative_prompt_embeds_cached(
                batch_size=cfg.sample.sample_batch_size,
                device=accelerator.device,
            )

        executor = futures.ThreadPoolExecutor(max_workers=cfg.reward.num_workers)
        global_step = 0
        train_iter = iter(train_loader)
        best_reward = -float("inf")
        save_root = os.path.join(str(self.output_dir), "checkpoints")

        try:
            for epoch in range(cfg.num_epochs):
                if cfg.eval_every_n_epochs and epoch % cfg.eval_every_n_epochs == 0 and epoch > 0:
                    self._run_eval(
                        trainer,
                        wan_algo,
                        eval_loader,
                        reward_fn,
                        cfg,
                        accelerator,
                        global_step,
                        eval_num_batches=cfg.eval_num_batches,
                        eval_num_videos=cfg.eval_num_videos,
                        eval_fps=cfg.eval_fps,
                        epoch=epoch,
                    )

                if cfg.save_every_n_epochs and epoch % cfg.save_every_n_epochs == 0 and epoch >= 10:
                    ckpt_dir = os.path.join(save_root, f"epoch-{epoch}-step-{global_step}")
                    self._save_checkpoint(
                        accelerator,
                        ckpt_dir,
                        ema,
                        global_step=global_step,
                        epoch=epoch,
                        wan_algo=wan_algo,
                    )

                samples = []
                for batch_idx in tqdm(
                    range(cfg.sample.num_batches_per_epoch),
                    disable=not accelerator.is_local_main_process,
                    desc=f"Epoch {epoch}: sampling",
                ):
                    current_epoch_seed = epoch * cfg.sample.num_batches_per_epoch + batch_idx
                    sampler.set_epoch(current_epoch_seed)
                    if hasattr(train_dataset, "set_epoch"):
                        train_dataset.set_epoch(current_epoch_seed)

                    batch = next(train_iter)
                    batch = _to_device(batch, accelerator.device)
                    with torch.no_grad():
                        batch = wan_algo.prepare_embeds(batch)
                        if isinstance(batch.get("prompt_embeds", None), (list, tuple)):
                            text_len = accelerator.unwrap_model(wan_algo.model).text_len
                            batch["prompt_embeds"] = WanTextToVideo.pad_text_context_to_tensor(
                                batch["prompt_embeds"], text_len
                            )
                        if "prompt_ids" not in batch and "prompts" in batch:
                            batch["prompt_ids"] = wan_algo.tokenizer(batch["prompts"]).to(
                                accelerator.device
                            )

                    scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
                    timesteps = timesteps[: cfg.sample.num_steps]

                    with torch.no_grad():
                        videos, lat_traj, old_logps, kl = trainer.rollout_batch(
                            batch,
                            scheduler,
                            timesteps,
                            neg_prompt_embeds=neg_prompt_embeds,
                        )

                    vis_train_dir = os.path.join(
                        self.output_dir,
                        "visualizations",
                        "train",
                        f"epoch_{epoch}_step_{global_step}",
                    )
                    os.makedirs(vis_train_dir, exist_ok=True)
                    videos_uint8 = self._videos_to_uint8(videos)
                    for vid_idx, frames in enumerate(videos_uint8):
                        save_path = os.path.join(
                            vis_train_dir,
                            f"rank_{accelerator.process_index}_sample_{batch_idx}_vid_{vid_idx}.mp4",
                        )
                        imageio.mimsave(
                            save_path,
                            frames,
                            fps=cfg.eval_fps,
                            codec="libx264",
                            format="FFMPEG",
                        )

                    batch_metadata = [
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "phase": "train",
                            "rank": accelerator.process_index,
                            "sample_idx": batch_idx,
                        }
                        for _ in range(videos.shape[0])
                    ]
                    reward_future = executor.submit(
                        reward_fn,
                        videos,
                        batch.get("prompts", None),
                        batch_metadata,
                    )
                    time.sleep(0)

                    samples.append(
                        {
                            "prompt_embeds": _maybe_detach_cpu(batch["prompt_embeds"]),
                            "negative_prompt_embeds": _maybe_detach_cpu(
                                neg_prompt_embeds[: batch["prompt_embeds"].shape[0]]
                                if neg_prompt_embeds is not None
                                else None
                            ),
                            "clip_fea": _maybe_detach_cpu(batch.get("clip_embeds", None)),
                            "y": _maybe_detach_cpu(batch.get("image_embeds", None)),
                            "prompts": batch.get("prompts", []),
                            "prompt_ids": _maybe_detach_cpu(batch.get("prompt_ids", None)),
                            "latents": lat_traj[:, :-1].detach().cpu(),
                            "gt_latents": batch["video_lat"].detach().cpu(),
                            "next_latents": lat_traj[:, 1:].detach().cpu(),
                            "old_log_probs": old_logps.detach().cpu(),
                            "kl": kl.detach().cpu(),
                            "reward_fut": reward_future,
                        }
                    )

                all_train_samples = []
                for sample in tqdm(samples, desc="Processing Rewards", disable=not accelerator.is_local_main_process):
                    rewards_dict, _ = sample["reward_fut"].result()
                    kl_gpu = sample["kl"].to(accelerator.device)

                    if "avg" in rewards_dict:
                        reward_raw = rewards_dict["avg"]
                    else:
                        keys = [
                            key
                            for key, value in rewards_dict.items()
                            if isinstance(value, (list, np.ndarray))
                        ]
                        if not keys:
                            raise ValueError(
                                f"No valid reward scores found in keys: {rewards_dict.keys()}"
                            )
                        reward_raw = np.array(rewards_dict[keys[0]])
                        for key in keys[1:]:
                            reward_raw += np.array(rewards_dict[key])

                    reward_tensor = torch.as_tensor(
                        reward_raw,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    kl_penalty = kl_gpu.float().mean(dim=1)
                    sample["rewards"] = (
                        reward_tensor - cfg.sample.kl_reward * kl_penalty
                    ).unsqueeze(-1)
                    del sample["reward_fut"]
                    all_train_samples.append(sample)

                cat = {}
                keys = [
                    "prompt_embeds",
                    "negative_prompt_embeds",
                    "clip_fea",
                    "y",
                    "latents",
                    "next_latents",
                    "old_log_probs",
                    "kl",
                    "rewards",
                    "gt_latents",
                ]
                for key in keys:
                    first = all_train_samples[0].get(key)
                    if first is not None:
                        tensors = [sample[key].to(accelerator.device) for sample in all_train_samples]
                        cat[key] = torch.cat(tensors, dim=0)
                    else:
                        cat[key] = None

                local_rewards = cat["rewards"]
                gathered_rewards = accelerator.gather(local_rewards).detach().cpu().numpy()
                current_mean_reward = gathered_rewards.mean()
                _log_metrics(
                    accelerator,
                    {"train/mean_reward": current_mean_reward},
                    step=global_step,
                )

                if current_mean_reward > best_reward:
                    best_reward = current_mean_reward
                    best_save_dir = os.path.join(save_root, "best_reward")
                    self._save_checkpoint(
                        accelerator,
                        best_save_dir,
                        ema,
                        global_step=global_step,
                        epoch=epoch,
                        wan_algo=wan_algo,
                        reward_info={
                            "mean_reward": best_reward,
                            "epoch": epoch,
                            "step": global_step,
                        },
                    )

                if all_train_samples[0].get("prompt_ids") is not None:
                    ids_list = [
                        sample["prompt_ids"].to(accelerator.device)
                        for sample in all_train_samples
                    ]
                    gathered_ids = accelerator.gather(torch.cat(ids_list, dim=0)).cpu().numpy()
                    prompts_gathered = wan_algo.tokenizer.tokenizer.batch_decode(
                        gathered_ids,
                        skip_special_tokens=True,
                    )
                else:
                    logger.warning(
                        "No prompt_ids found; falling back to local prompts for advantage normalization."
                    )
                    prompts_gathered = sum(
                        [sample["prompts"] for sample in all_train_samples], []
                    )

                advantages_np, group_size, _ = trainer.compute_advantages(
                    prompts_gathered if cfg.per_prompt_stat_tracking else None,
                    gathered_rewards,
                )

                if epoch < 2 and cfg.per_prompt_stat_tracking:
                    _log_metrics(
                        accelerator,
                        {
                            "train/stat_warmup_mean_reward": float(current_mean_reward),
                            "train/stat_warmup_group_size": float(group_size or 0),
                        },
                        step=global_step,
                    )
                    continue

                adv_tensor = torch.as_tensor(
                    advantages_np,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                local_batch_size = cat["rewards"].shape[0]
                start_idx = accelerator.process_index * local_batch_size
                end_idx = start_idx + local_batch_size
                local_adv = adv_tensor[start_idx:end_idx]
                num_steps = cat["old_log_probs"].shape[1]
                local_adv = local_adv.expand(-1, num_steps)

                train_payload = {
                    "prompt_embeds": cat["prompt_embeds"],
                    "negative_prompt_embeds": cat["negative_prompt_embeds"],
                    "clip_fea": cat["clip_fea"],
                    "y": cat["y"],
                    "latents": cat["latents"],
                    "next_latents": cat["next_latents"],
                    "old_log_probs": cat["old_log_probs"],
                    "advantages": local_adv,
                    "gt_latents": cat["gt_latents"],
                }

                for _ in range(cfg.train.num_inner_epochs):
                    scheduler, timesteps = wan_algo.build_scheduler(is_training=False)
                    scheduler.timesteps = scheduler.timesteps.to(accelerator.device)
                    timesteps = scheduler.timesteps[: cfg.sample.num_steps]
                    info = trainer.train_on_samples(
                        train_payload,
                        scheduler,
                        timesteps,
                        optimizer,
                        global_step=global_step,
                        ema=ema,
                    )
                    if info:
                        _log_metrics(
                            accelerator,
                            {f"train/{key}": value.item() for key, value in info.items()},
                            step=global_step,
                        )
                    global_step += 1
        finally:
            executor.shutdown(wait=True)
            if getattr(accelerator, "trackers", None):
                accelerator.end_training()


class WanAlgoWrapper:
    def __init__(self, algo):
        self.algo = algo

    def __getattr__(self, name):
        return getattr(self.algo, name)

    def format_timestep(self, timestep, batch_size, device):
        if torch.is_tensor(timestep):
            timestep = timestep.to(device)
            if timestep.ndim == 0:
                timestep = timestep.expand(batch_size)
            elif timestep.ndim == 1 and len(timestep) == 1:
                timestep = timestep.expand(batch_size)
            return timestep.long()
        return torch.full((batch_size,), timestep, device=device, dtype=torch.long)

    def construct_diffusion_forcing_timesteps(
        self,
        batch_size,
        t_curr,
        device,
        clean_t,
        hist_len=1,
    ):
        if self.algo.diffusion_forcing.enabled:
            num_frames = self.algo.lat_t
            t_vec = torch.full((batch_size, num_frames), t_curr, device=device)
            t_vec[:, :hist_len] = clean_t
            return t_vec
        return self.format_timestep(t_curr, batch_size, device)

    def _predict_noise(
        self,
        latents,
        t_vec,
        prompt_embeds,
        negative_prompt_embeds,
        do_cfg,
        guidance_scale,
        clip_fea=None,
        y=None,
    ):
        def run_forward(lat, timestep, context):
            return self.model(
                lat,
                t=timestep,
                context=context,
                seq_len=self.algo.max_tokens,
                clip_fea=clip_fea,
                y=y,
            )

        if do_cfg:
            noise_uncond = run_forward(latents, t_vec, negative_prompt_embeds)
            noise_text = run_forward(latents, t_vec, prompt_embeds)
            return noise_uncond + guidance_scale * (noise_text - noise_uncond)
        return run_forward(latents, t_vec, prompt_embeds)

    @torch.no_grad()
    def sample_step_with_logprob(
        self,
        scheduler,
        latents,
        t_vec,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale,
        do_cfg,
        clip_fea=None,
        y=None,
        determistic=False,
    ):
        noise = self._predict_noise(
            latents,
            t_vec,
            prompt_embeds,
            negative_prompt_embeds,
            do_cfg,
            guidance_scale,
            clip_fea,
            y,
        )

        scheduler_timestep = t_vec[:, -1] if t_vec.ndim > 1 else t_vec
        return sde_step_with_logprob(
            scheduler,
            noise.float(),
            scheduler_timestep,
            latents.float(),
            prev_sample=None,
            determistic=determistic,
            return_dt_and_std_dev_t=True,
        )

    def logprob_of_transition(
        self,
        scheduler,
        latents,
        next_latents,
        t_vec,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale,
        do_cfg,
        clip_fea=None,
        y=None,
    ):
        noise = self._predict_noise(
            latents,
            t_vec,
            prompt_embeds,
            negative_prompt_embeds,
            do_cfg,
            guidance_scale,
            clip_fea,
            y,
        )

        scheduler_timestep = t_vec[:, -1] if t_vec.ndim > 1 else t_vec
        return sde_step_with_logprob(
            scheduler,
            noise.float(),
            scheduler_timestep,
            latents.float(),
            prev_sample=next_latents.float(),
            return_dt_and_std_dev_t=True,
        )

from __future__ import annotations

import math
import os
from pathlib import Path

import logging

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from algorithms.idm import IDMResNetPlus
from datasets.robotwin_idm import RobotwinIDMDataset, robotwin_idm_collate
from idm import is_close, seed_torch
from .exp_base import BaseExperiment


class IDMTrainingExperiment(BaseExperiment):
    compatible_algorithms = dict(
        idm_resnet_plus=IDMResNetPlus,
    )

    compatible_datasets = dict(
        robotwin_idm=RobotwinIDMDataset,
    )
    compatible_collate_fns = dict(
        robotwin_idm=robotwin_idm_collate,
    )

    def _build_dataset(self, split: str):
        dataset_name = self.root_cfg.dataset._name
        if dataset_name not in self.compatible_datasets:
            raise ValueError(
                f"Dataset {dataset_name} is not registered for IDMTrainingExperiment."
            )
        return self.compatible_datasets[dataset_name](self.root_cfg.dataset, split=split)

    def _init_trackers(self, accelerator: Accelerator) -> None:
        if self.root_cfg.wandb.mode == "disabled":
            return
        accelerator.init_trackers(
            self.root_cfg.wandb.project,
            config=OmegaConf.to_container(self.root_cfg, resolve=True),
            init_kwargs={
                "wandb": {
                    "name": self.root_cfg.name,
                    "entity": self.root_cfg.wandb.entity,
                    "mode": self.root_cfg.wandb.mode,
                }
            },
        )

    def _build_loaders(self):
        train_dataset = self._build_dataset("training")
        val_dataset = self._build_dataset("validation")
        dataset_size = len(train_dataset)
        if dataset_size < 2:
            raise ValueError("IDM training requires at least 2 samples for train/validation split.")

        requested_val = min(
            int(self.root_cfg.dataset.ratio_eval * dataset_size),
            1000,
        )
        val_size = max(1, min(requested_val, dataset_size - 1))
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(self.cfg.training.seed)
        permutation = torch.randperm(dataset_size, generator=generator).tolist()
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        dataset_name = self.root_cfg.dataset._name
        worker_count = min(os.cpu_count() or self.cfg.training.num_workers, self.cfg.training.num_workers)
        loader_kwargs = dict(
            num_workers=worker_count,
            pin_memory=True,
            collate_fn=self.compatible_collate_fns[dataset_name],
            persistent_workers=worker_count > 0,
        )
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = self.cfg.training.prefetch_factor

        train_loader = DataLoader(
            train_subset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.cfg.training.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

        if is_rank_zero := (not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            split_dir = Path(self.output_dir)
            (split_dir / "train_split_indices.txt").write_text(
                "\n".join(str(idx) for idx in train_indices)
            )
            (split_dir / "validation_split_indices.txt").write_text(
                "\n".join(str(idx) for idx in val_indices)
            )

        logging.info(f"Dataset split: {train_size} train / {val_size} val (total {dataset_size})")
        return train_loader, val_loader

    def _build_model_components(self):
        if self.algo is None:
            self._build_algo()
        model = self.algo.build_model()
        optimizer = AdamW(model.parameters(), lr=self.cfg.training.lr)
        loss_fn = nn.SmoothL1Loss()
        return model, optimizer, loss_fn

    def _warmup_steps(self) -> int:
        total_steps = int(self.cfg.training.num_iterations)
        configured_steps = self.cfg.training.get("warmup_steps", None)
        if configured_steps is not None:
            warmup_steps = int(configured_steps)
        else:
            warmup_ratio = float(self.cfg.training.get("warmup_ratio", 0.0))
            warmup_steps = int(total_steps * warmup_ratio)
        return max(0, min(warmup_steps, max(total_steps - 1, 0)))

    def _lr_scale(self, step: int) -> float:
        scheduler = self.cfg.training.get("scheduler", "none")
        if scheduler in {None, "none"}:
            return 1.0
        if scheduler != "cosine":
            raise ValueError(f"Unsupported IDM lr scheduler: {scheduler}")

        total_steps = int(self.cfg.training.num_iterations)
        warmup_steps = self._warmup_steps()
        min_lr_ratio = float(self.cfg.training.get("min_lr_ratio", 0.0))

        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), 1e-8)

        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    def _set_learning_rate(self, optimizer, step: int) -> float:
        base_lr = float(self.cfg.training.lr)
        lr = base_lr * self._lr_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _load_checkpoint(self, model, optimizer) -> int:
        checkpoint_path = self.cfg.training.get("load_from", None)
        if checkpoint_path is None:
            return 0

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        if (
            "optimizer_state_dict" in checkpoint
            and optimizer is not None
            and not self.cfg.training.get("eval_only", False)
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("step", 0))

    def _save_checkpoint(self, accelerator: Accelerator, model, optimizer, step: int, best: bool = False):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return

        ckpt_dir = Path(self.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
            "step": step,
        }
        torch.save(state, ckpt_dir / f"step_{step}.pt")
        torch.save(state, ckpt_dir / "latest.pt")
        if best:
            torch.save(state, ckpt_dir / "best.pt")

    def _forward(self, model, batch):
        return model(batch["images"])

    def _loss(self, model, loss_fn, prediction, target):
        if self.cfg.training.use_normalization:
            return loss_fn(model.normalize(prediction), model.normalize(target))
        return loss_fn(prediction, target)

    @torch.no_grad()
    def _evaluate(self, accelerator: Accelerator, model, val_loader, loss_fn, step: int):
        model.eval()
        total = torch.zeros(4, device=accelerator.device, dtype=torch.float64)

        pbar = tqdm(val_loader, desc=f"Eval step {step}", disable=not accelerator.is_main_process)
        for batch in pbar:
            prediction = self._forward(model, batch)
            target = batch["actions"]
            loss = self._loss(model, loss_fn, prediction, target)
            l1_error = torch.abs(target - prediction).mean(dim=1).sum()
            correct = is_close(target, prediction).float().sum()
            stats = torch.tensor(
                [
                    loss.detach().float() * target.shape[0],
                    l1_error.detach().float(),
                    correct.detach().float(),
                    float(target.shape[0]),
                ],
                device=accelerator.device,
                dtype=torch.float64,
            )
            total += accelerator.reduce(stats, reduction="sum")

        metrics = {
            "val_loss": (total[0] / total[3]).item(),
            "val_l1_error": (total[1] / total[3]).item(),
            "val_correct_rate": (total[2] / total[3]).item(),
        }
        if accelerator.is_main_process and getattr(accelerator, "trackers", None):
            accelerator.log(metrics, step=step)
        model.train()
        return metrics

    def _run(self, eval_only: bool = False) -> None:
        seed_torch(self.cfg.training.seed)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        log_with = "wandb" if self.root_cfg.wandb.mode != "disabled" else None
        accelerator = Accelerator(
            mixed_precision=self.cfg.training.mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            log_with=log_with,
        )
        self._init_trackers(accelerator)

        train_loader, val_loader = self._build_loaders()
        model, optimizer, loss_fn = self._build_model_components()
        start_step = self._load_checkpoint(model, optimizer)
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
        )
        model.normalize = accelerator.unwrap_model(model).normalize

        if eval_only:
            metrics = self._evaluate(accelerator, model, val_loader, loss_fn, start_step)
            if accelerator.is_main_process:
                accelerator.print(metrics)
            accelerator.end_training()
            return

        best_val_loss = float("inf")
        train_iter = iter(train_loader)
        global_step = start_step
        total_iters = self.cfg.training.num_iterations

        pbar = tqdm(
            initial=start_step,
            total=total_iters,
            desc="Training",
            disable=not accelerator.is_main_process,
        )

        while global_step < total_iters:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            prediction = self._forward(model, batch)
            target = batch["actions"]
            loss = self._loss(model, loss_fn, prediction, target)
            cur_lr = self._set_learning_rate(optimizer, global_step)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            global_step += 1

            batch_accuracy = is_close(target, prediction).float().mean()
            cur_loss = loss.detach().item()
            cur_acc = batch_accuracy.detach().item()
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.3f}", lr=f"{cur_lr:.1e}")
            pbar.update(1)

            if global_step % self.cfg.training.log_interval == 0:
                metrics = {
                    "train_loss": cur_loss,
                    "train_batch_accuracy": cur_acc,
                    "learning_rate": cur_lr,
                }
                if accelerator.is_main_process and getattr(accelerator, "trackers", None):
                    accelerator.log(metrics, step=global_step)

            if (
                global_step % self.cfg.training.eval_interval == 0
                or global_step == total_iters
            ):
                metrics = self._evaluate(accelerator, model, val_loader, loss_fn, global_step)
                is_best = metrics["val_loss"] < best_val_loss
                if is_best:
                    best_val_loss = metrics["val_loss"]
                self._save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    step=global_step,
                    best=is_best,
                )
                if accelerator.is_main_process:
                    logging.info(
                        f"[Step {global_step}] val_loss={metrics['val_loss']:.4f} "
                        f"val_acc={metrics['val_correct_rate']:.3f} "
                        f"val_l1={metrics['val_l1_error']:.4f}"
                        f"{' (best)' if is_best else ''}"
                    )

        pbar.close()
        accelerator.end_training()

    def training(self) -> None:
        self._run(eval_only=self.cfg.training.get("eval_only", False))

    def validation(self) -> None:
        self._run(eval_only=True)

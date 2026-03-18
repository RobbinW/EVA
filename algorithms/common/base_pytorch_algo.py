from abc import ABC, abstractmethod
from pathlib import Path
import warnings
from typing import Any, Union, Sequence, Optional

from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch
import numpy as np
from PIL import Image
import wandb
import einops

from utils.video_utils import write_numpy_to_mp4


class BasePytorchAlgo(pl.LightningModule, ABC):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.debug = self.cfg.debug
        super().__init__()


    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: (only if multiple dataloaders used) The index of the dataloader that produced this batch.

        Return:
            Any of these options:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch. This is only supported for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        """
        Return an optimizer. If you need to use more than one optimizer, refer to pytorch lightning documentation:
        https://lightning.ai/docs/pytorch/stable/common/optimization.html
        """
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)

    def _save_video_locally(
        self,
        key: str,
        video: np.ndarray,
        fps: int,
        caption: str = None,
        step: int = None,
    ) -> None:
        if video.ndim == 4:
            video = video[None]

        root_dir = Path(self.trainer.default_root_dir if self.trainer is not None else Path.cwd())
        video_dir = root_dir / "videos"
        key_path = Path(*[part for part in key.split("/") if part])
        target_dir = video_dir / key_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        step_value = self.global_step if step is None else step
        stem = key_path.name or "video"

        for idx, clip in enumerate(video):
            if clip.ndim != 4:
                raise ValueError(
                    f"Expected each video clip to be 4D, got shape {clip.shape}"
                )

            # Internal logging uses [T, C, H, W], while local mp4 writing expects [T, H, W, C].
            if clip.shape[1] == 3:
                clip = np.transpose(clip, (0, 2, 3, 1))
            elif clip.shape[-1] != 3:
                raise ValueError(
                    f"Expected RGB video in [T, C, H, W] or [T, H, W, C], got shape {clip.shape}"
                )

            suffix = f"_{idx}" if len(video) > 1 else ""
            video_path = target_dir / f"{stem}{suffix}_step{step_value:06d}.mp4"
            write_numpy_to_mp4(clip, video_path, fps=fps)

            if caption:
                caption_path = video_path.with_suffix(".txt")
                caption_path.write_text(caption + "\n")

    def log_video(
        self,
        key: str,
        video: Union[np.ndarray, torch.Tensor],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        fps: int = 12,
        format: str = "mp4",
        caption: str = None,
        step: int = None,
    ):
        """
        Log video to wandb. WandbLogger in pytorch lightning does not support video logging yet, so we call wandb directly.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8
                or [0, 1] otherwise.
            mean: optional, the mean to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize video tensor, assuming unnormalized data is in [0, 1].
            key: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().float().numpy()

        expand_shape = [1] * (len(video.shape) - 2) + [3, 1, 1]
        if std is not None:
            if isinstance(std, (float, int)):
                std = [std] * 3
            if isinstance(std, torch.Tensor):
                std = std.detach().cpu().numpy()
            std = np.array(std).reshape(*expand_shape)
            video = video * std
        if mean is not None:
            if isinstance(mean, (float, int)):
                mean = [mean] * 3
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()
            mean = np.array(mean).reshape(*expand_shape)
            video = video + mean

        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        experiment = getattr(self.logger, "experiment", None) if self.logger is not None else None
        if experiment is not None and hasattr(experiment, "log"):
            experiment.log(
                {
                    key: wandb.Video(video, fps=fps, format=format, caption=caption),
                },
                step=self.global_step if step is None else step,
            )
            return

        self._save_video_locally(key, video, fps=fps, caption=caption, step=step)

    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image, Sequence[Image.Image]],
        mean: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        std: Union[np.ndarray, torch.Tensor, Sequence, float] = None,
        **kwargs: Any,
    ):
        """
        Log image(s) using WandbLogger.
        Args:
            key: the name of the video.
            image: a single image or a batch of images. If a batch of images, the shape should be (batch, channel, height, width).
            mean: optional, the mean to unnormalize image tensor, assuming unnormalized data is in [0, 1].
            std: optional, the std to unnormalize tensor, assuming unnormalized data is in [0, 1].
            kwargs: optional, WandbLogger log_image kwargs, such as captions=xxx.
        """
        if isinstance(image, Image.Image):
            image = [image]
        elif len(image) and not isinstance(image[0], Image.Image):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()

            if len(image.shape) == 3:
                image = image[None]

            if image.shape[1] == 3:
                if image.shape[-1] == 3:
                    warnings.warn(
                        f"Two channels in shape {image.shape} have size 3, assuming channel first."
                    )
                image = einops.rearrange(image, "b c h w -> b h w c")

            if std is not None:
                if isinstance(std, (float, int)):
                    std = [std] * 3
                if isinstance(std, torch.Tensor):
                    std = std.detach().cpu().numpy()
                std = np.array(std)[None, None, None]
                image = image * std
            if mean is not None:
                if isinstance(mean, (float, int)):
                    mean = [mean] * 3
                if isinstance(mean, torch.Tensor):
                    mean = mean.detach().cpu().numpy()
                mean = np.array(mean)[None, None, None]
                image = image + mean

            if image.dtype != np.uint8:
                image = np.clip(image, a_min=0.0, a_max=1.0) * 255
                image = image.astype(np.uint8)
                image = [img for img in image]

        self.logger.log_image(key=key, images=image, **kwargs)

    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def register_data_mean_std(
        self,
        mean: Union[str, float, Sequence],
        std: Union[str, float, Sequence],
        namespace: str = "data",
    ):
        """
        Register mean and std of data as tensor buffer.

        Args:
            mean: the mean of data.
            std: the std of data.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("mean", mean), ("std", std)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))

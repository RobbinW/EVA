from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image


class DinoPreprocessor:
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = SimpleNamespace(**cfg)

        self.use_transform = getattr(cfg, "use_transform", False)
        self.dino_size = getattr(cfg, "dino_size", 512)
        self.current_progress = 0.0
        self._rebuild_transform()
        self.resize = torchvision.transforms.Resize((self.dino_size, self.dino_size))
        self.normalize_dino = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _lerp(self, start, end, progress):
        return start + (end - start) * progress

    def _rebuild_transform(self):
        p = self.current_progress
        brightness = (self._lerp(0.8, 0.4, p), self._lerp(1.2, 1.6, p))
        contrast = (self._lerp(0.7, 0.7, p), self._lerp(1.3, 1.7, p))
        saturation = (self._lerp(0.5, 0.5, p), self._lerp(1.5, 2.0, p))
        hue = self._lerp(0.05, 0.10, p)
        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def set_augmentation_progress(self, progress: float) -> None:
        self.current_progress = max(0.0, min(1.0, progress))
        self._rebuild_transform()

    def process_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image = self.resize(image)
            if self.use_transform:
                image = self.transform(image)
            image = torchvision.transforms.functional.to_tensor(image).float()
        return self.normalize_dino(image)

    def process_mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        if isinstance(mask, Image.Image):
            mask = torchvision.transforms.functional.resize(
                mask,
                (self.dino_size, self.dino_size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )
            mask = torchvision.transforms.functional.to_tensor(mask).float()
            if mask.shape[0] == 3:
                mask = mask[:1]
            mask = torch.where(mask > 0.5, 1.0, 0.0)
        return mask

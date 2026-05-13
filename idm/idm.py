import torch
import torch.nn as nn

from .resnet import ResNet, Resnet_plus
from .unet import UNet


class IDM(nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__()
        if model_name == "mask":
            self.model = Mask(*args, **kwargs)
        elif model_name == "unet":
            self.model = Unet(*args, **kwargs)
        elif model_name == "resnet":
            self.model = ResNet(*args, **kwargs)
        elif model_name == "resnet_plus":
            self.model = Resnet_plus(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # RoboTwin demo_clean_vidar joint_action/vector population statistics.
        train_mean = torch.tensor(
            [
                -0.21647900,
                1.08763040,
                0.79517410,
                -0.33707824,
                0.05907208,
                -0.02981021,
                0.67330010,
                0.23602079,
                1.11225000,
                0.82878625,
                -0.34568170,
                -0.01535936,
                0.01111238,
                0.67293334,
            ]
        )
        train_std = torch.tensor(
            [
                0.30222420,
                0.99274236,
                0.78431356,
                0.66731670,
                0.24801265,
                0.56997130,
                0.45218885,
                0.31425828,
                1.02002080,
                0.81537780,
                0.71815896,
                0.26411867,
                0.62363460,
                0.45184147,
            ]
        )
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.train_mean) / (self.train_std + 1e-8)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.train_std + self.train_mean

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return self.denormalize(output[0]), *output[1:]
        return self.denormalize(output)


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):
        super().__init__()
        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

    def forward(self, images, *args, **kwargs):
        return self.resnet_model(torch.tanh(self.mask_model(images)))


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):
        super().__init__()
        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, *args, **kwargs)

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        return outputs

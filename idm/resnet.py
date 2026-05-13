import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, output_dim=14, num_frames=3, input_channels=3, *args, **kwargs):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels * num_frames,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2_x = self._make_layer(64, 3, 1)
        self.conv3_x = self._make_layer(128, 4, 2)
        self.conv4_x = self._make_layer(256, 6, 2)
        self.conv5_x = self._make_layer(512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                BottleNeckResidualBlock(self.in_channels, out_channels, current_stride)
            )
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 5:
            batch, time, channels, height, width = x.shape
            x = x.view(batch, time * channels, height, width)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        return self.fc(output)


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=None):
        super().__init__()
        self.temperature = (
            nn.Parameter(torch.ones(1))
            if temperature is None
            else torch.tensor([temperature])
        )

    def forward(self, feature_map):
        batch, channels, height, width = feature_map.shape
        device = feature_map.device
        flat = feature_map.view(batch, channels, -1)
        heatmap = F.softmax(flat / self.temperature.to(device), dim=2).view(
            batch, channels, height, width
        )
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        expected_x = torch.sum(heatmap * pos_x, dim=[2, 3])
        expected_y = torch.sum(heatmap * pos_y, dim=[2, 3])
        return torch.cat([expected_x, expected_y], dim=1)


class Resnet_plus(nn.Module):
    def __init__(self, num_frames=3, output_dim=14, backbone_type="resnet50", *args, **kwargs):
        super().__init__()
        if backbone_type == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
            feature_dim = 512
        elif backbone_type == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError("Only support resnet18 or resnet50")

        input_channels = 3 * num_frames
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            self.backbone.conv1.weight[:] = (
                original_conv1.weight.repeat(1, num_frames, 1, 1) / num_frames
            )

        self.spatial_softmax = SpatialSoftmax()
        self.head_input_dim = feature_dim * 2
        self.mlp_head = nn.Sequential(
            nn.Linear(self.head_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, *args, **kwargs):
        if len(x.shape) != 5:
            raise ValueError("Input tensor must have shape [B, T, C, H, W]")
        batch, time, channels, height, width = x.shape
        x = x.view(batch, time * channels, height, width)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        coords = self.spatial_softmax(x)
        return self.mlp_head(coords)

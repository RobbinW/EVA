import torch
from torch import nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = None
        self.conv = double_conv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        if self.up is None:
            self.up = nn.Upsample(size=x2.size()[2:], mode="bilinear", align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


def double_conv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layers = [
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def down(in_channels, out_channels):
    return nn.Sequential(
        nn.MaxPool2d(2, stride=2),
        double_conv(in_channels, out_channels),
    )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel=64):
        super().__init__()
        self.in_conv = double_conv(in_channels, base_channel)
        self.num_layers = 5
        downs = []
        ups = []
        for i in range(self.num_layers):
            down_in_channel = base_channel * 2**i
            down_out_channel = down_in_channel * 2 if i < self.num_layers - 1 else down_in_channel
            up_in_channel = base_channel * 2 ** (self.num_layers - i)
            up_out_channel = up_in_channel // 4 if i < self.num_layers - 1 else base_channel
            downs.append(down(down_in_channel, down_out_channel))
            ups.append(Up(up_in_channel, up_out_channel))
        self.downs = nn.Sequential(*downs)
        self.ups = nn.Sequential(*ups)
        self.out = nn.Conv2d(base_channel, out_channels, kernel_size=1)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)

        x = self.in_conv(x)
        xs = [x]
        for down_layer in self.downs:
            x = down_layer(xs[-1])
            xs.append(x)

        x_out = None
        for x_skip, up_layer in zip(xs[::-1][1:], self.ups):
            if x_out is None:
                x_out = up_layer(xs[-1], xs[-2])
            else:
                x_out = up_layer(x_out, x_skip)
        return self.out(x_out)

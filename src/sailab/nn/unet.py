"""U-Net with a torchvision ResNet encoder, shared by Model 1 and Model 2.

With ResNet-34 this is the doc's Model 1 baseline (~24 million parameters, 1 to 2 GB of GPU memory
at 256 x 256 with mixed precision). Model 2 reuses the encoder and decoder around its time
transformer. Inputs must have height and width divisible by 32; `pad_to_multiple` handles that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

ENCODER_CHANNELS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
}


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int) -> None:
        super().__init__()
        self.conv = ConvBlock(cin + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder: str = "resnet34", pretrained: bool = False,
                 decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)) -> None:
        super().__init__()
        if encoder not in ENCODER_CHANNELS:
            raise ValueError(f"unknown encoder {encoder}; choose from {sorted(ENCODER_CHANNELS)}")
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, encoder)(weights=weights)
        if in_channels != 3:
            old = backbone.conv1
            new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    new.weight.copy_(mean_w.repeat(1, in_channels, 1, 1) * 3.0 / in_channels)
            backbone.conv1 = new
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.encoder_channels = ENCODER_CHANNELS[encoder]

        skips = self.encoder_channels[:-1][::-1] + [0]  # /16, /8, /4, /2, none
        blocks = []
        cin = self.encoder_channels[-1]
        for cskip, cout in zip(skips, decoder_channels, strict=True):
            blocks.append(UpBlock(cin, cskip, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(x)                     # /2
        f1 = self.layer1(self.pool(f0))       # /4
        f2 = self.layer2(f1)                  # /8
        f3 = self.layer3(f2)                  # /16
        f4 = self.layer4(f3)                  # /32
        return [f0, f1, f2, f3, f4]

    def decode(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        skips = feats[:-1][::-1] + [None]
        for block, skip in zip(self.decoder, skips, strict=True):
            x = block(x, skip)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pad_to_multiple(x: torch.Tensor, multiple: int = 32, value: float = 0.0) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad (..., H, W) on the bottom and right; returns the padded tensor and the original size."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=value)
    return x, (h, w)

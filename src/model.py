from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if shapes mismatch due to odd dims
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [0, diffX, 0, diffY, 0, diffZ])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 16):
        super().__init__()
        self.down1 = Down(in_channels, base)
        self.down2 = Down(base, base * 2)
        self.bottom = conv3x3(base * 2, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up1 = Up(base * 2, base)
        self.out = nn.Conv3d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, s1 = self.down1(x)
        x, s2 = self.down2(x)
        x = self.bottom(x)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        logits = self.out(x)
        return logits
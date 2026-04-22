from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(feat))

from __future__ import annotations

import torch
import torch.nn as nn


class AdvancedParadigmProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)

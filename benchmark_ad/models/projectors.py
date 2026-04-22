from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        h = int(hidden_dim or max(in_dim, out_dim))
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), h),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, int(out_dim)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        out = self.net(tokens.reshape(-1, d))
        return out.reshape(b, n, -1)


class Conv1x1Projector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


class DepthwiseSeparableProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(feat))


class TokenToMapProjector(nn.Module):
    """
    Convert token sequence [B,N,D] to 2D feature map and align to target feature map size.
    """

    def __init__(self, in_dim: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(int(in_dim), int(out_ch), kernel_size=3, padding=1, bias=False)

    def forward(self, tokens: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = tokens.shape
        side = int(n**0.5)
        if side * side != n:
            raise ValueError(f"Token count {n} cannot be reshaped to square map.")
        fmap = tokens.transpose(1, 2).reshape(b, d, side, side)
        fmap = F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
        return self.conv(fmap)

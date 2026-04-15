from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGDDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MGDLoss(nn.Module):
    def __init__(self, mask_ratio_min: float = 0.3, mask_ratio_max: float = 0.5) -> None:
        super().__init__()
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)

    def random_mask(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        keep = 1.0 - ratio
        mask = (torch.rand((b, 1, h, w), device=feat.device, dtype=feat.dtype) < keep).float()
        return feat * mask

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor, decoder: nn.Module) -> torch.Tensor:
        if student_map.shape != teacher_map.shape:
            raise ValueError(f"MGD shape mismatch: {student_map.shape} vs {teacher_map.shape}")
        masked = self.random_mask(student_map)
        pred = decoder(masked)
        return F.mse_loss(pred, teacher_map.detach())


class SPKDLoss(nn.Module):
    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        x = feat.reshape(b, c, h * w).transpose(1, 2)
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        return torch.bmm(x, x.transpose(1, 2))

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor) -> torch.Tensor:
        if student_map.shape != teacher_map.shape:
            raise ValueError(f"SPKD shape mismatch: {student_map.shape} vs {teacher_map.shape}")
        gram_s = self._gram(student_map)
        gram_t = self._gram(teacher_map.detach())
        return torch.norm(gram_s - gram_t, p="fro") / max(1, student_map.shape[0])

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(f"Token shape mismatch: {student_tokens.shape} vs {teacher_tokens.shape}")
        sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)
        b, n, _ = student_tokens.shape
        side = int(n**0.5)
        if side * side != n:
            mask_token = F.interpolate(mask, size=(1, n), mode="nearest").reshape(b, n)
        else:
            mask_token = F.interpolate(mask, size=(side, side), mode="nearest").reshape(b, n)
        normal_term = (1.0 - mask_token) * (1.0 - sim)
        anomaly_term = mask_token * torch.relu(self.margin - sim)
        return (normal_term + anomaly_term).mean()


class AttentionKLLoss(nn.Module):
    def forward(self, student_attn: torch.Tensor, teacher_attn: torch.Tensor) -> torch.Tensor:
        return F.kl_div(torch.log(student_attn + 1e-8), teacher_attn.detach(), reduction="batchmean")


class CLSSimilarityLoss(nn.Module):
    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student_cls, dim=-1, eps=1e-8)
        t = F.normalize(teacher_cls, dim=-1, eps=1e-8)
        return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()


class SpatialContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if student_map.shape != teacher_map.shape:
            raise ValueError(f"Feature shape mismatch: {student_map.shape} vs {teacher_map.shape}")
        s = F.normalize(student_map, dim=1, eps=1e-8)
        t = F.normalize(teacher_map, dim=1, eps=1e-8)
        sim = (s * t).sum(dim=1, keepdim=True)
        m = F.interpolate(mask, size=sim.shape[-2:], mode="nearest")
        normal_term = (1.0 - m) * (1.0 - sim)
        anomaly_term = m * torch.relu(self.margin - sim)
        return (normal_term + anomaly_term).mean()


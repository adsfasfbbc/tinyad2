from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens_to_attention(tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    tokens: [B, N, D]
    return attention proxy: [B, N, N]
    """
    t = F.normalize(tokens, dim=-1, eps=1e-8)
    logits = torch.matmul(t, t.transpose(-1, -2)) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=-1)


class TokenContrastiveLoss(nn.Module):
    """
    Normal-region attraction + anomaly-region repulsion.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(f"token shape mismatch: {tuple(student_tokens.shape)} vs {tuple(teacher_tokens.shape)}")
        sim = F.cosine_similarity(student_tokens, teacher_tokens, dim=-1)  # [B, N]
        if mask is None:
            return (1.0 - sim).mean()

        b, n, _ = student_tokens.shape
        side = int(n**0.5)
        if side * side != n:
            raise ValueError(f"token count {n} is not a square number for mask projection")
        token_mask = F.interpolate(mask.float(), size=(side, side), mode="nearest").reshape(b, -1)
        normal_term = (1.0 - token_mask) * (1.0 - sim)
        anomaly_term = token_mask * torch.relu(self.margin - sim)
        return (normal_term + anomaly_term).mean()


class AttentionMimicryLoss(nn.Module):
    """KL divergence between teacher and student attention proxy matrices."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        s_attn = _tokens_to_attention(student_tokens, temperature=self.temperature)
        t_attn = _tokens_to_attention(teacher_tokens, temperature=self.temperature).detach()
        return F.kl_div(torch.log(s_attn + 1e-8), t_attn, reduction="batchmean")


class CLSTokenAlignmentLoss(nn.Module):
    """Global semantic alignment on CLS-like pooled token."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student_cls, dim=-1, eps=1e-8)
        t = F.normalize(teacher_cls, dim=-1, eps=1e-8)
        return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()

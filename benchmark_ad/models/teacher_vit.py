from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

import VisualAD_lib


def _tokens_to_attn(tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    t = F.normalize(tokens, dim=-1, eps=1e-8)
    logits = torch.matmul(t, t.transpose(-1, -2)) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=-1)


@dataclass
class TeacherOutput:
    cls_token: torch.Tensor
    patch_tokens: List[torch.Tensor]
    attention_maps: List[torch.Tensor]


class VisualADTeacherViTL14(torch.nn.Module):
    """Frozen teacher wrapper; backbone fixed to VisualAD ViT-L/14."""

    def __init__(self, features_list: Sequence[int], device: torch.device, backbone: str = "ViT-L/14@336px") -> None:
        super().__init__()
        if str(backbone) != "ViT-L/14@336px":
            raise ValueError("Teacher backbone must be fixed to ViT-L/14@336px for this benchmark.")
        self.features_list = [int(v) for v in features_list]
        self.model, _ = VisualAD_lib.load("ViT-L/14@336px", device=device)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim = int(self.model.visual.embed_dim)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> TeacherOutput:
        out = self.model.encode_image(image, self.features_list)
        patch_start_idx = int(out["patch_start_idx"])
        patch_tokens = [pt[:, patch_start_idx:, :] for pt in out["patch_tokens"]]
        attn = [_tokens_to_attn(pt) for pt in patch_tokens]
        cls_token = out["class_features"]
        return TeacherOutput(
            cls_token=F.normalize(cls_token, dim=-1, eps=1e-8),
            patch_tokens=[F.normalize(pt, dim=-1, eps=1e-8) for pt in patch_tokens],
            attention_maps=attn,
        )


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F

import VisualAD_lib


@dataclass
class TeacherTokenOutput:
    stage_tokens: List[torch.Tensor]  # each [B,N,D]
    cls_token: torch.Tensor  # [B,D]
    patch_start_idx: int


class VisualADViTTeacherAdapter(torch.nn.Module):
    """Frozen VisualAD ViT teacher for token-level distillation."""

    def __init__(self, backbone: str, features_list: Sequence[int], device: torch.device) -> None:
        super().__init__()
        self.features_list = [int(v) for v in features_list]
        self.model, _ = VisualAD_lib.load(backbone, device=device)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim = int(self.model.visual.embed_dim)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherTokenOutput:
        out = self.model.encode_image(images, self.features_list)
        patch_start_idx = int(out["patch_start_idx"])
        raw_tokens = out["patch_tokens"]
        stage_tokens = [F.normalize(t[:, patch_start_idx:, :], dim=-1, eps=1e-8) for t in raw_tokens]
        cls_token = F.normalize(out["class_features"], dim=-1, eps=1e-8)
        return TeacherTokenOutput(stage_tokens=stage_tokens, cls_token=cls_token, patch_start_idx=patch_start_idx)

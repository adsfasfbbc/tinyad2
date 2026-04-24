from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

import VisualAD_lib

from .contracts import TeacherOutput


class VisualADTeacher(nn.Module):
    """
    Frozen VisualAD ViT-L/14@336px teacher extracting multi-layer patch/cls tokens.
    VisualAD inserts two learnable tokens (anomaly/normal) before class token:
      token indices = [0: anomaly, 1: normal, 2: class, 3...: patches]
    For ViT-L/14@336px specifically, expected outputs are:
      - patch_tokens[layer]: [B, 576, 1024]
      - cls_tokens[layer]: [B, 1024]
    """

    def __init__(
        self,
        backbone: str = "ViT-L/14@336px",
        layers: Iterable[int] = (8, 16, 24),
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.layers: List[int] = sorted({int(l) for l in layers})
        self.backbone = backbone
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = VisualAD_lib.load(self.backbone, device=self.device)
        self.to(self.device)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        images = images.to(self.device)
        patch_tokens: Dict[int, torch.Tensor] = {}
        cls_tokens: Dict[int, torch.Tensor] = {}

        image_features = self.model.encode_image(images, feature_list=self.layers)
        token_list = image_features["patch_tokens"]  # each: [B, 579, 1024] for 336px model
        patch_start_idx = int(image_features.get("patch_start_idx", 3))

        if len(token_list) != len(self.layers):
            raise RuntimeError(
                f"Requested {len(self.layers)} teacher layers {self.layers}, "
                f"but VisualAD returned {len(token_list)} token outputs."
            )

        class_idx = patch_start_idx - 1  # default class index = 2 when patch_start_idx = 3
        for layer, token in zip(self.layers, token_list):
            cls_tokens[layer] = token[:, class_idx, :]
            patch_tokens[layer] = token[:, patch_start_idx:, :]

        return TeacherOutput(patch_tokens=patch_tokens, cls_tokens=cls_tokens)


# Backward compatible alias for existing imports.
OpenCLIPTeacher = VisualADTeacher

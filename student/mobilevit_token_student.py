from __future__ import annotations

from typing import Dict, List, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from distill.projector import MLPProjectionHead


class MobileViTTokenStudent(nn.Module):
    """
    MobileViT student with token extraction + MLP token projectors.
    """

    def __init__(
        self,
        backbone_name: str = "mobilevit_s",
        feature_out_indices: Sequence[int] = (1, 2, 3, 4),
        teacher_dim: int = 1024,
        projector_hidden_dim: int = 1024,
        projector_dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(int(i) for i in feature_out_indices),
        )
        self.feature_out_indices = [int(i) for i in feature_out_indices]
        self.stage_channels = list(self.backbone.feature_info.channels())
        self.teacher_dim = int(teacher_dim)

        self.token_projectors = nn.ModuleList(
            [
                MLPProjectionHead(
                    input_dim=int(ch),
                    hidden_dim=int(projector_hidden_dim),
                    output_dim=int(teacher_dim),
                    dropout=float(projector_dropout),
                )
                for ch in self.stage_channels
            ]
        )
        self.cls_projector = MLPProjectionHead(
            input_dim=int(teacher_dim),
            hidden_dim=int(projector_hidden_dim),
            output_dim=int(teacher_dim),
            dropout=float(projector_dropout),
        )

    @staticmethod
    def _map_to_tokens(feat: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H*W, C]
        return feat.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        feats = self.backbone(x)
        raw_tokens = [self._map_to_tokens(f) for f in feats]
        proj_tokens = [p(t) for p, t in zip(self.token_projectors, raw_tokens)]
        cls_tokens = []
        for t in proj_tokens:
            pooled = t.mean(dim=1, keepdim=True)
            cls_tokens.append(self.cls_projector(pooled).squeeze(1))
        cls_token = torch.stack(cls_tokens, dim=0).mean(dim=0)

        return {
            "raw_features": feats,
            "stage_tokens": proj_tokens,
            "cls_token": F.normalize(cls_token, dim=-1, eps=1e-8),
        }

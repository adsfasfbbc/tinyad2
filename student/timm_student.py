from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import timm
import torch
import torch.nn as nn


class TimmStudent(nn.Module):
    """Generic timm-based student with dynamic channel alignment."""

    def __init__(
        self,
        backbone_name: str,
        feature_out_indices: Sequence[int],
        teacher_channels: Optional[Sequence[int]] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.feature_out_indices = [int(i) for i in feature_out_indices]

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(self.feature_out_indices),
        )

        self._student_channels: List[int] = list(self.backbone.feature_info.channels())
        if teacher_channels is None:
            self.teacher_channels = list(self._student_channels)
        else:
            self.teacher_channels = [int(c) for c in teacher_channels]

        if len(self.teacher_channels) != len(self._student_channels):
            raise ValueError(
                "teacher_channels length must match number of extracted student feature maps: "
                f"{len(self.teacher_channels)} vs {len(self._student_channels)}"
            )

        self.align_layers = nn.ModuleList(
            [
                # Bias is disabled since this layer is used for channel projection/alignment only.
                nn.Conv2d(in_channels=s_c, out_channels=t_c, kernel_size=1, bias=False)
                for s_c, t_c in zip(self._student_channels, self.teacher_channels)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        student_features = self.backbone(x)
        aligned_features = [
            align(feature)
            for align, feature in zip(self.align_layers, student_features)
        ]
        return {
            "student_features": student_features,
            "aligned_features": aligned_features,
        }

    @property
    def output_channels(self) -> List[int]:
        return list(self._student_channels)

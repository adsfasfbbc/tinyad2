from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import AdapterOutput, StudentOutput


class _StageProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationAdapter(nn.Module):
    """
    Align VMamba stages to CLIP ViT-L/14 dense token space.
      1) Spatial align -> 24x24
      2) Channel projection -> 1024
      3) Virtual cls generation -> adaptive avg pooling
    """

    def __init__(self, stage_channels: Dict[int, int], output_dim: int = 1024, target_hw: int = 24) -> None:
        super().__init__()
        self.target_hw = target_hw
        self.output_dim = output_dim
        self.projectors = nn.ModuleDict(
            {
                f"stage_{stage}": _StageProjector(in_ch, output_dim)
                for stage, in_ch in sorted(stage_channels.items())
            }
        )
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, student_output: StudentOutput) -> AdapterOutput:
        dense: Dict[int, torch.Tensor] = {}
        cls: Dict[int, torch.Tensor] = {}
        stage_shapes: Dict[int, torch.Size] = {}

        for stage, feat in sorted(student_output.stages.items()):
            stage_shapes[stage] = feat.shape
            aligned = F.interpolate(feat, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
            projected = self.projectors[f"stage_{stage}"](aligned)
            dense[stage] = projected
            cls[stage] = self.cls_pool(projected).flatten(1)

        return AdapterOutput(dense=dense, cls=cls, stage_shapes=stage_shapes)

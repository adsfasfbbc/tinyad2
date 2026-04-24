from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TeacherOutput:
    """Feature contract for frozen CLIP teacher."""

    patch_tokens: Dict[int, torch.Tensor]  # layer -> [B, 576, 1024]
    cls_tokens: Dict[int, torch.Tensor]  # layer -> [B, 1024]


@dataclass
class StudentOutput:
    """Feature contract for VMamba student stages."""

    stages: Dict[int, torch.Tensor]  # stage -> [B, C, H, W]


@dataclass
class AdapterOutput:
    """Feature contract for aligned student outputs."""

    dense: Dict[int, torch.Tensor]  # stage -> [B, 1024, 24, 24]
    cls: Dict[int, torch.Tensor]  # stage -> [B, 1024]
    stage_shapes: Dict[int, torch.Size]  # stage -> original feature map shape

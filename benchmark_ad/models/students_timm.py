from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens_to_attn(tokens: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    t = F.normalize(tokens, dim=-1, eps=1e-8)
    logits = torch.matmul(t, t.transpose(-1, -2)) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=-1)


@dataclass(frozen=True)
class StudentSpec:
    route: str
    candidates: List[str]
    default_out_indices: List[int]
    token_depth: int


STUDENT_SPECS: Dict[str, StudentSpec] = {
    "mobilevit_s": StudentSpec(route="B", candidates=["mobilevit_s"], default_out_indices=[1, 2, 3, 4], token_depth=2),
    "tinyvit_11m": StudentSpec(
        route="A",
        candidates=["tiny_vit_11m_224", "tinyvit_11m_224"],
        default_out_indices=[0, 1, 2, 3],
        token_depth=4,
    ),
    "fastvit_t8": StudentSpec(route="C", candidates=["fastvit_t8", "fasternet_t0"], default_out_indices=[0, 1, 2, 3], token_depth=0),
    "mobilenetv4_hybrid": StudentSpec(
        route="B",
        candidates=["mobilenetv4_hybrid_medium", "mobilenetv3_small_100"],
        default_out_indices=[1, 2, 3, 4],
        token_depth=2,
    ),
    "unireplknet_s": StudentSpec(
        route="C",
        candidates=["unireplknet_s", "repvgg_a1"],
        default_out_indices=[0, 1, 2, 3],
        token_depth=0,
    ),
}


class UnifiedStudent(nn.Module):
    def __init__(self, model_name: str, out_indices: Sequence[int], pretrained: bool = True) -> None:
        super().__init__()
        if model_name not in STUDENT_SPECS:
            raise KeyError(f"Unsupported student_name={model_name}, supported={list(STUDENT_SPECS.keys())}")
        self.spec = STUDENT_SPECS[model_name]
        self.model_name = model_name
        self.out_indices = [int(v) for v in out_indices]
        self.backbone_name = None
        self.backbone = None
        errors = []
        for candidate in self.spec.candidates:
            try:
                self.backbone = timm.create_model(
                    candidate,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=tuple(self.out_indices),
                )
                self.backbone_name = candidate
                break
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{candidate}: {exc}")
        if self.backbone is None:
            raise RuntimeError(f"Cannot create student {model_name}, tried candidates: {errors}")
        self.stage_channels = list(self.backbone.feature_info.channels())

    @staticmethod
    def _map_to_tokens(feat: torch.Tensor) -> torch.Tensor:
        return feat.flatten(2).transpose(1, 2)

    def forward(self, image: torch.Tensor):
        feats = self.backbone(image)
        tokens = [self._map_to_tokens(f) for f in feats]
        token_count = min(len(tokens), max(0, int(self.spec.token_depth)))
        deep_tokens = tokens[-token_count:] if token_count > 0 else []
        attn_maps = [_tokens_to_attn(t) for t in deep_tokens]
        cls_token = F.normalize(tokens[-1].mean(dim=1), dim=-1, eps=1e-8)
        return {
            "route": self.spec.route,
            "feature_maps": feats,
            "tokens_all": tokens,
            "tokens_deep": [F.normalize(t, dim=-1, eps=1e-8) for t in deep_tokens],
            "attention_maps": attn_maps,
            "cls_token": cls_token,
            "stage_channels": self.stage_channels,
            "backbone_name": self.backbone_name,
        }


def build_student(model_name: str, out_indices: Sequence[int] | None = None, pretrained: bool = True) -> UnifiedStudent:
    spec = STUDENT_SPECS[model_name]
    resolved = list(spec.default_out_indices) if out_indices is None else [int(v) for v in out_indices]
    return UnifiedStudent(model_name=model_name, out_indices=resolved, pretrained=pretrained)

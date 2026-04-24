from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .contracts import StudentOutput


class _FallbackStudent(nn.Module):
    """Fallback hierarchical CNN used only when VMamba implementation is unavailable."""

    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 96, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {1: s1, 2: s2, 3: s3, 4: s4}


class VMambaStudent(nn.Module):
    """
    VMamba-Small wrapper exposing four stage feature maps as [B, C, H, W].
    If VMamba cannot be imported, a fallback student can be optionally used.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        use_fallback_if_unavailable: bool = True,
    ) -> None:
        super().__init__()
        self._is_fallback = False
        self.stage_channels = {1: 96, 2: 192, 3: 384, 4: 768}

        model = None
        try:
            # Common import paths in VMamba repos
            from vmamba import build_vmamba_small  # type: ignore

            model = build_vmamba_small(pretrained=pretrained)
        except Exception:
            try:
                from vmamba.models.vmamba import VMamba  # type: ignore

                model = VMamba(variant="small", pretrained=pretrained)
            except Exception:
                if not use_fallback_if_unavailable:
                    raise RuntimeError(
                        "VMamba-Small implementation not found. Install VMamba package or set fallback enabled."
                    )

        if model is None:
            self._is_fallback = True
            self.backbone = _FallbackStudent()
        else:
            self.backbone = model

    def _extract_vmamba_stages(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
            if isinstance(feat, dict):
                mapping = {}
                for i, key in enumerate(sorted(feat.keys()), start=1):
                    if i <= 4:
                        mapping[i] = feat[key]
                if len(mapping) == 4:
                    return mapping
            if isinstance(feat, (list, tuple)) and len(feat) >= 4:
                return {i + 1: feat[i] for i in range(4)}

        out = self.backbone(x)
        if isinstance(out, dict):
            if all(k in out for k in (1, 2, 3, 4)):
                return {1: out[1], 2: out[2], 3: out[3], 4: out[4]}
            keys = sorted(out.keys())[:4]
            return {i + 1: out[k] for i, k in enumerate(keys)}
        if isinstance(out, (list, tuple)) and len(out) >= 4:
            return {i + 1: out[i] for i in range(4)}

        raise RuntimeError("Unable to parse VMamba stage outputs.")

    def forward(self, images: torch.Tensor) -> StudentOutput:
        stages = self._extract_vmamba_stages(images)
        return StudentOutput(stages=stages)

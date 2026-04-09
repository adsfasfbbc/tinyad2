from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_noise_2d(height: int, width: int, res_h: int, res_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    res_h = max(1, int(res_h))
    res_w = max(1, int(res_w))
    yy = torch.linspace(0.0, float(res_h) - 1e-6, steps=height, device=device, dtype=dtype)
    xx = torch.linspace(0.0, float(res_w) - 1e-6, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    y0 = grid_y.floor().long()
    x0 = grid_x.floor().long()
    y1 = y0 + 1
    x1 = x0 + 1

    yf = grid_y - y0.to(dtype)
    xf = grid_x - x0.to(dtype)

    angles = 2 * torch.pi * torch.rand((res_h + 1, res_w + 1), device=device, dtype=dtype)
    gradients = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]

    d00 = torch.stack([xf, yf], dim=-1)
    d10 = torch.stack([xf - 1.0, yf], dim=-1)
    d01 = torch.stack([xf, yf - 1.0], dim=-1)
    d11 = torch.stack([xf - 1.0, yf - 1.0], dim=-1)

    n00 = (g00 * d00).sum(dim=-1)
    n10 = (g10 * d10).sum(dim=-1)
    n01 = (g01 * d01).sum(dim=-1)
    n11 = (g11 * d11).sum(dim=-1)

    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


class SyntheticAnomalyGenerator:
    """
    Online synthetic anomaly generator for unified training.
    Input/Output image is tensor [C,H,W], mask [1,H,W], label in {0,1}.
    """

    def __init__(
        self,
        mode: str = "perlin",
        perlin_scale_min: int = 2,
        perlin_scale_max: int = 6,
        threshold_min: float = 0.60,
        threshold_max: float = 0.75,
        blend_alpha: float = 0.5,
        min_patch_ratio: float = 0.08,
        max_patch_ratio: float = 0.30,
    ) -> None:
        self.mode = str(mode).strip().lower()
        self.perlin_scale_min = int(perlin_scale_min)
        self.perlin_scale_max = int(perlin_scale_max)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.blend_alpha = float(blend_alpha)
        self.min_patch_ratio = float(min_patch_ratio)
        self.max_patch_ratio = float(max_patch_ratio)

    def _cutpaste(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, h, w = image.shape
        out = image.clone()
        mask = torch.zeros((1, h, w), device=image.device, dtype=image.dtype)
        min_h = max(4, int(h * self.min_patch_ratio))
        max_h = max(5, int(h * self.max_patch_ratio))
        min_w = max(4, int(w * self.min_patch_ratio))
        max_w = max(5, int(w * self.max_patch_ratio))
        patch_h = random.randint(min_h, max_h)
        patch_w = random.randint(min_w, max_w)
        patch_h = min(patch_h, h - 1)
        patch_w = min(patch_w, w - 1)
        src_y = random.randint(0, h - patch_h - 1)
        src_x = random.randint(0, w - patch_w - 1)
        dst_y = random.randint(0, h - patch_h - 1)
        dst_x = random.randint(0, w - patch_w - 1)

        patch = image[:, src_y : src_y + patch_h, src_x : src_x + patch_w]
        if random.random() < 0.5:
            patch = torch.flip(patch, dims=[-1])
        if random.random() < 0.5:
            patch = torch.rot90(patch, k=1, dims=[-2, -1])
            patch = F.interpolate(
                patch.unsqueeze(0), size=(patch_h, patch_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        patch = torch.clamp(patch + torch.empty_like(patch).uniform_(-0.1, 0.1), min=image.min(), max=image.max())

        out[:, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch
        mask[:, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = 1.0
        return out, mask

    def _perlin(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c, h, w = image.shape
        scale = random.randint(self.perlin_scale_min, self.perlin_scale_max)
        noise = _perlin_noise_2d(h, w, scale, scale, image.device, image.dtype)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        threshold = random.uniform(self.threshold_min, self.threshold_max)
        mask = (noise > threshold).to(image.dtype).unsqueeze(0)
        if mask.mean().item() < 0.01:
            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)
            ry, rx = max(2, h // 12), max(2, w // 12)
            y0, y1 = max(0, cy - ry), min(h, cy + ry)
            x0, x1 = max(0, cx - rx), min(w, cx + rx)
            mask[:, y0:y1, x0:x1] = 1.0

        noise_rgb = noise.unsqueeze(0).repeat(c, 1, 1)
        synth = torch.clamp(image + 0.25 * (2.0 * noise_rgb - 1.0), min=image.min(), max=image.max())
        out = image * (1.0 - mask) + ((1.0 - self.blend_alpha) * image + self.blend_alpha * synth) * mask
        return out, mask

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.mode in {"cutpaste", "cut_paste"}:
            out, mask = self._cutpaste(image)
        else:
            out, mask = self._perlin(image)
        label = int(mask.max().item() > 0)
        return out, mask, label


def build_anomaly_generator(config: Optional[Dict[str, float]] = None) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
    cfg = config or {}
    return SyntheticAnomalyGenerator(
        mode=str(cfg.get("mode", "perlin")),
        perlin_scale_min=int(cfg.get("perlin_scale_min", 2)),
        perlin_scale_max=int(cfg.get("perlin_scale_max", 6)),
        threshold_min=float(cfg.get("threshold_min", 0.60)),
        threshold_max=float(cfg.get("threshold_max", 0.75)),
        blend_alpha=float(cfg.get("blend_alpha", 0.5)),
        min_patch_ratio=float(cfg.get("min_patch_ratio", 0.08)),
        max_patch_ratio=float(cfg.get("max_patch_ratio", 0.30)),
    )

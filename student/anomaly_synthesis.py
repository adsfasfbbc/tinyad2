from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class AnomalySynthesisResult:
    images: torch.Tensor
    masks: torch.Tensor


class BaseAnomalySynthesizer:
    def __init__(self, anomaly_prob: float = 1.0) -> None:
        self.anomaly_prob = float(max(0.0, min(1.0, anomaly_prob)))

    def __call__(self, images: torch.Tensor) -> AnomalySynthesisResult:
        raise NotImplementedError


class IdentitySynthesizer(BaseAnomalySynthesizer):
    def __call__(self, images: torch.Tensor) -> AnomalySynthesisResult:
        b, _, h, w = images.shape
        return AnomalySynthesisResult(images=images, masks=torch.zeros((b, 1, h, w), device=images.device, dtype=images.dtype))


class CutPasteSynthesizer(BaseAnomalySynthesizer):
    def __init__(
        self,
        anomaly_prob: float = 1.0,
        min_patch_ratio: float = 0.08,
        max_patch_ratio: float = 0.30,
    ) -> None:
        super().__init__(anomaly_prob=anomaly_prob)
        self.min_patch_ratio = float(min_patch_ratio)
        self.max_patch_ratio = float(max_patch_ratio)

    def __call__(self, images: torch.Tensor) -> AnomalySynthesisResult:
        b, c, h, w = images.shape
        out = images.clone()
        masks = torch.zeros((b, 1, h, w), device=images.device, dtype=images.dtype)

        for i in range(b):
            if torch.rand(1, device=images.device).item() > self.anomaly_prob:
                continue

            min_h = max(4, int(h * self.min_patch_ratio))
            max_h = max(5, int(h * self.max_patch_ratio))
            min_w = max(4, int(w * self.min_patch_ratio))
            max_w = max(5, int(w * self.max_patch_ratio))
            patch_h = int(torch.randint(min_h, max_h + 1, (1,), device=images.device).item())
            patch_w = int(torch.randint(min_w, max_w + 1, (1,), device=images.device).item())
            patch_h = max(1, min(patch_h, h - 1))
            patch_w = max(1, min(patch_w, w - 1))

            max_src_y = max(0, h - patch_h)
            max_src_x = max(0, w - patch_w)
            max_dst_y = max(0, h - patch_h)
            max_dst_x = max(0, w - patch_w)
            src_y = int(torch.randint(0, max_src_y + 1, (1,), device=images.device).item()) if max_src_y > 0 else 0
            src_x = int(torch.randint(0, max_src_x + 1, (1,), device=images.device).item()) if max_src_x > 0 else 0
            dst_y = int(torch.randint(0, max_dst_y + 1, (1,), device=images.device).item()) if max_dst_y > 0 else 0
            dst_x = int(torch.randint(0, max_dst_x + 1, (1,), device=images.device).item()) if max_dst_x > 0 else 0

            patch = images[i : i + 1, :, src_y : src_y + patch_h, src_x : src_x + patch_w]
            out[i : i + 1, :, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = patch
            masks[i, :, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = 1.0

        return AnomalySynthesisResult(images=out, masks=masks)


def _fade(t: torch.Tensor) -> torch.Tensor:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin_noise_2d(height: int, width: int, res_h: int, res_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    res_h = max(1, int(res_h))
    res_w = max(1, int(res_w))

    # Keep max value strictly below res_{h,w}; otherwise floor()==res and y1=y0+1 / x1=x0+1 overflow gradient grid.
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


class PerlinNoiseSynthesizer(BaseAnomalySynthesizer):
    def __init__(
        self,
        anomaly_prob: float = 1.0,
        perlin_scale_min: int = 2,
        perlin_scale_max: int = 6,
        threshold_min: float = 0.60,
        threshold_max: float = 0.75,
        blend_alpha: float = 0.5,
    ) -> None:
        super().__init__(anomaly_prob=anomaly_prob)
        self.perlin_scale_min = int(perlin_scale_min)
        self.perlin_scale_max = int(perlin_scale_max)
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.blend_alpha = float(blend_alpha)

    def __call__(self, images: torch.Tensor) -> AnomalySynthesisResult:
        b, c, h, w = images.shape
        out = images.clone()
        masks = torch.zeros((b, 1, h, w), device=images.device, dtype=images.dtype)
        perm = torch.randperm(b, device=images.device)

        for i in range(b):
            if torch.rand(1, device=images.device).item() > self.anomaly_prob:
                continue

            scale = int(torch.randint(self.perlin_scale_min, self.perlin_scale_max + 1, (1,), device=images.device).item())
            noise = _perlin_noise_2d(h, w, scale, scale, images.device, images.dtype)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            th = torch.empty((1,), device=images.device, dtype=images.dtype).uniform_(self.threshold_min, self.threshold_max).item()
            mask = (noise > th).to(images.dtype).unsqueeze(0).unsqueeze(0)

            if mask.mean().item() < 0.01:
                cy = int(torch.randint(0, h, (1,), device=images.device).item())
                cx = int(torch.randint(0, w, (1,), device=images.device).item())
                ry, rx = max(2, h // 12), max(2, w // 12)
                y0, y1 = max(0, cy - ry), min(h, cy + ry)
                x0, x1 = max(0, cx - rx), min(w, cx + rx)
                mask[:, :, y0:y1, x0:x1] = 1.0

            ref = images[perm[i] : perm[i] + 1]
            noise_rgb = noise.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
            synth_patch = (1.0 - self.blend_alpha) * ref + self.blend_alpha * (images[i : i + 1] + 0.25 * (2.0 * noise_rgb - 1.0))
            out[i : i + 1] = images[i : i + 1] * (1.0 - mask) + synth_patch * mask
            masks[i : i + 1] = mask

        return AnomalySynthesisResult(images=out, masks=masks)


def build_anomaly_synthesizer(name: str, config: Dict[str, float] | None = None) -> BaseAnomalySynthesizer:
    cfg = config or {}
    normalized = (name or "none").strip().lower()
    if normalized in {"none", "identity"}:
        return IdentitySynthesizer(anomaly_prob=float(cfg.get("anomaly_prob", 0.0)))
    if normalized in {"cutpaste", "cut_paste"}:
        return CutPasteSynthesizer(
            anomaly_prob=float(cfg.get("anomaly_prob", 1.0)),
            min_patch_ratio=float(cfg.get("min_patch_ratio", 0.08)),
            max_patch_ratio=float(cfg.get("max_patch_ratio", 0.30)),
        )
    if normalized in {"perlin", "perlin_noise"}:
        return PerlinNoiseSynthesizer(
            anomaly_prob=float(cfg.get("anomaly_prob", 1.0)),
            perlin_scale_min=int(cfg.get("perlin_scale_min", 2)),
            perlin_scale_max=int(cfg.get("perlin_scale_max", 6)),
            threshold_min=float(cfg.get("threshold_min", 0.60)),
            threshold_max=float(cfg.get("threshold_max", 0.75)),
            blend_alpha=float(cfg.get("blend_alpha", 0.5)),
        )
    raise ValueError(f"Unsupported anomaly synthesizer: {name}")

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class LossAdjustmentResult:
    total_loss: torch.Tensor
    weights: Dict[str, float]


class BaseLossAdjuster:
    def combine(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> LossAdjustmentResult:
        raise NotImplementedError


class FixedLossAdjuster(BaseLossAdjuster):
    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = {k: float(v) for k, v in (weights or {}).items()}

    def combine(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> LossAdjustmentResult:
        total = None
        applied_weights: Dict[str, float] = {}
        for name, value in losses.items():
            w = float(self.weights.get(name, 1.0))
            applied_weights[name] = w
            item = value * w
            total = item if total is None else (total + item)
        if total is None:
            raise ValueError("No losses provided to FixedLossAdjuster.combine")
        return LossAdjustmentResult(total_loss=total, weights=applied_weights)


class WarmupAnomalyLossAdjuster(BaseLossAdjuster):
    def __init__(
        self,
        clean_weight: float = 1.0,
        anomaly_start_weight: float = 0.1,
        anomaly_target_weight: float = 1.0,
        warmup_epochs: int = 5,
    ) -> None:
        self.clean_weight = float(clean_weight)
        self.anomaly_start_weight = float(anomaly_start_weight)
        self.anomaly_target_weight = float(anomaly_target_weight)
        # warmup_epochs=0 means no warmup and anomaly weight uses target value from epoch 0.
        self.warmup_epochs = max(0, int(warmup_epochs))

    def _anomaly_weight(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.anomaly_target_weight
        ratio = min(1.0, max(0.0, float(epoch) / float(self.warmup_epochs)))
        return self.anomaly_start_weight + ratio * (self.anomaly_target_weight - self.anomaly_start_weight)

    def combine(self, losses: Dict[str, torch.Tensor], epoch: int, step: int) -> LossAdjustmentResult:
        clean = losses.get("clean_distill")
        anomaly = losses.get("anomaly_distill")
        if clean is None and anomaly is None:
            raise ValueError("No losses provided to WarmupAnomalyLossAdjuster.combine")

        w_clean = self.clean_weight
        w_anomaly = self._anomaly_weight(epoch)
        ref = clean if clean is not None else anomaly
        total = torch.tensor(0.0, device=ref.device)
        applied = {"clean_distill": w_clean, "anomaly_distill": w_anomaly}
        if clean is not None:
            total = total + clean * w_clean
        if anomaly is not None:
            total = total + anomaly * w_anomaly
        return LossAdjustmentResult(total_loss=total, weights=applied)


def build_loss_adjuster(name: str, config: Dict[str, float] | None = None) -> BaseLossAdjuster:
    cfg = config or {}
    normalized = (name or "fixed").strip().lower()
    if normalized == "fixed":
        weights = cfg.get("weights", {})
        if weights is None:
            weights = {}
        if not isinstance(weights, dict):
            raise ValueError(f"loss_adjuster_config.weights must be a dict, got {type(weights).__name__}")
        return FixedLossAdjuster(weights=weights)
    if normalized in {"warmup", "curriculum"}:
        return WarmupAnomalyLossAdjuster(
            clean_weight=float(cfg.get("clean_weight", 1.0)),
            anomaly_start_weight=float(cfg.get("anomaly_start_weight", 0.1)),
            anomaly_target_weight=float(cfg.get("anomaly_target_weight", 1.0)),
            warmup_epochs=int(cfg.get("warmup_epochs", 5)),
        )
    raise ValueError(f"Unsupported loss adjuster: {name}")

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareSpatialContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.3,
        boundary_focus: bool = False,
        boundary_weight: float = 2.0,
        boundary_dilation: int = 1,
        normalize_features: bool = True,
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.boundary_focus = bool(boundary_focus)
        self.boundary_weight = float(boundary_weight)
        self.boundary_dilation = max(1, int(boundary_dilation))
        self.normalize_features = bool(normalize_features)

    @staticmethod
    def _compute_boundary(mask: torch.Tensor, dilation: int) -> torch.Tensor:
        kernel = 2 * int(dilation) + 1
        dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
        eroded = -F.max_pool2d(-mask, kernel_size=kernel, stride=1, padding=dilation)
        boundary = (dilated - eroded).clamp(min=0.0, max=1.0)
        return (boundary > 0.0).float()

    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if student_map.shape != teacher_map.shape:
            raise ValueError(f"Feature shape mismatch: {student_map.shape} vs {teacher_map.shape}")
        if self.normalize_features:
            student_map = F.normalize(student_map, p=2, dim=1, eps=1e-8)
            teacher_map = F.normalize(teacher_map, p=2, dim=1, eps=1e-8)

        sim = (student_map * teacher_map).sum(dim=1, keepdim=True)
        m = F.interpolate(mask, size=sim.shape[-2:], mode="nearest")
        normal_term = (1.0 - m) * (1.0 - sim)
        anomaly_term = m * torch.relu(self.margin - sim)
        pointwise = normal_term + anomaly_term

        if self.boundary_focus:
            boundary = self._compute_boundary(m, self.boundary_dilation)
            weights = torch.ones_like(pointwise)
            weights = torch.where(boundary > 0.0, torch.full_like(weights, self.boundary_weight), weights)
            pointwise = pointwise * weights
        return pointwise.mean()


class GlobalCosineLoss(nn.Module):
    def forward(self, student_map: torch.Tensor, teacher_map: torch.Tensor, normalize_features: bool = True) -> torch.Tensor:
        if student_map.shape != teacher_map.shape:
            raise ValueError(f"Feature shape mismatch: {student_map.shape} vs {teacher_map.shape}")
        if normalize_features:
            student_map = F.normalize(student_map, p=2, dim=1, eps=1e-8)
            teacher_map = F.normalize(teacher_map, p=2, dim=1, eps=1e-8)
        s_vec = student_map.mean(dim=(2, 3))
        t_vec = teacher_map.mean(dim=(2, 3))
        return (1.0 - F.cosine_similarity(s_vec, t_vec, dim=1)).mean()

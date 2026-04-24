from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.contracts import AdapterOutput, TeacherOutput


@dataclass
class DistillLossConfig:
    lambda_dense: float = 1.0
    lambda_cls: float = 1.0
    pairings: Optional[List[Tuple[str, int]]] = None
    # pairing example: [("3+4", 16), ("4", 24)] meaning stage3/4 fused -> teacher layer16, stage4 -> layer24


class DistillationLoss(nn.Module):
    def __init__(self, cfg: DistillLossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _teacher_tokens_to_map(tokens: torch.Tensor, grid_size: int = 24) -> torch.Tensor:
        # tokens: [B, grid_size*grid_size, C] -> [B, C, grid_size, grid_size]
        b, n, c = tokens.shape
        if n != grid_size * grid_size:
            raise ValueError(f"Expected {grid_size * grid_size} tokens, got {n}.")
        return tokens.reshape(b, grid_size, grid_size, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _dense_term(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(student, teacher)
        s = student.flatten(2).transpose(1, 2)  # [B, 576, C]
        t = teacher.flatten(2).transpose(1, 2)
        cos = F.cosine_similarity(s, t, dim=-1).mean()
        return mse + (1.0 - cos)

    @staticmethod
    def _resolve_student_dense(adapter_output: AdapterOutput, stage_selector: str) -> torch.Tensor:
        if "+" in stage_selector:
            stages = [int(s.strip()) for s in stage_selector.split("+")]
            feats = [adapter_output.dense[s] for s in stages]
            return torch.stack(feats, dim=0).mean(dim=0)
        return adapter_output.dense[int(stage_selector)]

    @staticmethod
    def _resolve_student_cls(adapter_output: AdapterOutput, stage_selector: str) -> torch.Tensor:
        if "+" in stage_selector:
            stages = [int(s.strip()) for s in stage_selector.split("+")]
            feats = [adapter_output.cls[s] for s in stages]
            return torch.stack(feats, dim=0).mean(dim=0)
        return adapter_output.cls[int(stage_selector)]

    def forward(self, teacher: TeacherOutput, adapter_output: AdapterOutput) -> Dict[str, torch.Tensor]:
        if self.cfg.pairings is None:
            pairings = [("3+4", 16), ("4", 24)]
        else:
            pairings = self.cfg.pairings

        dense_losses: List[torch.Tensor] = []
        cls_losses: List[torch.Tensor] = []

        for student_selector, teacher_layer in pairings:
            if teacher_layer not in teacher.patch_tokens:
                continue

            s_dense = self._resolve_student_dense(adapter_output, student_selector)
            t_dense = self._teacher_tokens_to_map(teacher.patch_tokens[teacher_layer])
            dense_losses.append(self._dense_term(s_dense, t_dense))

            if teacher_layer in teacher.cls_tokens:
                s_cls = self._resolve_student_cls(adapter_output, student_selector)
                t_cls = teacher.cls_tokens[teacher_layer]
                cls_losses.append(1.0 - F.cosine_similarity(s_cls, t_cls, dim=1).mean())

        if not dense_losses:
            raise RuntimeError("No valid teacher/student pairings found for distillation loss.")

        dense_loss = torch.stack(dense_losses).mean()
        cls_loss = torch.stack(cls_losses).mean() if cls_losses else dense_loss.new_tensor(0.0)
        total = self.cfg.lambda_dense * dense_loss + self.cfg.lambda_cls * cls_loss

        return {
            "loss_total": total,
            "loss_dense": dense_loss,
            "loss_cls": cls_loss,
        }

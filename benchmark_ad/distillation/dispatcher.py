from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_ad.distillation.contrastive_loss import (
    AttentionKLLoss,
    CLSSimilarityLoss,
    SpatialContrastiveLoss,
    TokenContrastiveLoss,
)
from benchmark_ad.models.projectors import Conv1x1Projector, TokenMLPProjector, TokenToMapProjector


class HeterogeneousDistillationDispatcher(nn.Module):
    """
    Route strategies:
    - A: token + attention + cls (TinyViT)
    - B: deep token + shallow feature align + cls (MobileViT/MNv4-Hybrid)
    - C: teacher token->2D map + spatial contrastive (FastViT/UniRepLKNet)
    """

    def __init__(self, route: str, student_stage_channels: List[int], teacher_dim: int, cfg: Dict) -> None:
        super().__init__()
        self.route = str(route).strip().upper()
        self.student_stage_channels = [int(c) for c in student_stage_channels]
        self.teacher_dim = int(teacher_dim)
        self.deep_token_blocks = int(cfg.get("deep_token_blocks", 2))

        self.loss_token = TokenContrastiveLoss(margin=float(cfg.get("token_margin", 0.5)))
        self.loss_attn = AttentionKLLoss()
        self.loss_cls = CLSSimilarityLoss()
        self.loss_spatial = SpatialContrastiveLoss(margin=float(cfg.get("spatial_margin", 0.3)))

        self.w_token = float(cfg.get("weight_token", 1.0))
        self.w_attn = float(cfg.get("weight_attention", 1.0))
        self.w_cls = float(cfg.get("weight_cls", 0.5))
        self.w_spatial = float(cfg.get("weight_spatial", 1.0))
        self.w_shallow = float(cfg.get("weight_shallow", 0.2))
        self.shallow_align_max_channels = int(cfg.get("shallow_align_max_channels", 512))

        self.token_projectors = nn.ModuleList(
            [TokenMLPProjector(in_dim=c, out_dim=self.teacher_dim, hidden_dim=max(c, self.teacher_dim)) for c in self.student_stage_channels]
        )
        self.cls_projector = nn.Linear(self.student_stage_channels[-1], self.teacher_dim, bias=False)
        self.shallow_align_dims = [
            min(int(c), self.teacher_dim, self.shallow_align_max_channels) for c in self.student_stage_channels
        ]
        self.shallow_student_projectors = nn.ModuleList(
            [Conv1x1Projector(in_ch=c, out_ch=d) for c, d in zip(self.student_stage_channels, self.shallow_align_dims)]
        )
        self.shallow_teacher_projectors = nn.ModuleList(
            [TokenToMapProjector(in_dim=self.teacher_dim, out_ch=d) for d in self.shallow_align_dims]
        )
        self.teacher_to_map_projectors = nn.ModuleList(
            [TokenToMapProjector(in_dim=self.teacher_dim, out_ch=c) for c in self.student_stage_channels]
        )

    @staticmethod
    def _align_token_count(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs, n_s, d_s = student_tokens.shape
        bt, n_t, d_t = teacher_tokens.shape
        if bs != bt or d_s != d_t:
            raise ValueError(f"token shape mismatch {student_tokens.shape} vs {teacher_tokens.shape}")
        if n_s == n_t:
            return student_tokens, teacher_tokens
        side_s = int(n_s**0.5)
        side_t = int(n_t**0.5)
        if side_s * side_s == n_s and side_t * side_t == n_t:
            s = student_tokens.transpose(1, 2).reshape(bs, d_s, side_s, side_s)
            s = F.interpolate(s, size=(side_t, side_t), mode="bilinear", align_corners=False)
            s = s.flatten(2).transpose(1, 2)
            return s, teacher_tokens
        n = min(n_s, n_t)
        return student_tokens[:, :n], teacher_tokens[:, :n]

    @staticmethod
    def _mse_strict(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for mse: {tuple(a.shape)} vs {tuple(b.shape)}")
        return F.mse_loss(a, b)

    def forward(self, teacher_out: Dict, student_out: Dict, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        t_tokens: List[torch.Tensor] = teacher_out["patch_tokens"]
        t_attn: List[torch.Tensor] = teacher_out["attention_maps"]
        t_cls: torch.Tensor = teacher_out["cls_token"]

        s_maps: List[torch.Tensor] = student_out["feature_maps"]
        s_tokens_all: List[torch.Tensor] = student_out["tokens_all"]
        s_tokens_deep: List[torch.Tensor] = student_out["tokens_deep"]
        s_attn: List[torch.Tensor] = student_out["attention_maps"]
        s_cls_raw: torch.Tensor = student_out["cls_token"]

        if self.route == "A":
            n = min(len(s_tokens_deep), len(t_tokens))
            token_loss = torch.tensor(0.0, device=t_cls.device)
            attn_loss = torch.tensor(0.0, device=t_cls.device)
            for idx in range(n):
                s_proj = self.token_projectors[-n + idx](s_tokens_deep[idx])
                s_proj, t_aligned = self._align_token_count(s_proj, t_tokens[idx])
                token_loss = token_loss + self.loss_token(s_proj, t_aligned, mask)
                s_att = torch.softmax(torch.matmul(F.normalize(s_proj, dim=-1), F.normalize(s_proj, dim=-1).transpose(-1, -2)), dim=-1)
                t_att_map = t_att[idx]
                if s_att.shape != t_att_map.shape:
                    m = min(s_att.shape[-1], t_att_map.shape[-1])
                    s_att = s_att[:, :m, :m]
                    t_att_map = t_att_map[:, :m, :m]
                attn_loss = attn_loss + self.loss_attn(s_att, t_att_map)
            token_loss = token_loss / max(1, n)
            attn_loss = attn_loss / max(1, n)
            cls_loss = self.loss_cls(self.cls_projector(s_cls_raw), t_cls)
            total = self.w_token * token_loss + self.w_attn * attn_loss + self.w_cls * cls_loss
            return {"total": total, "token": token_loss, "attn": attn_loss, "cls": cls_loss}

        if self.route == "B":
            n = min(len(s_tokens_deep), len(t_tokens))
            token_loss = torch.tensor(0.0, device=t_cls.device)
            for idx in range(n):
                s_proj = self.token_projectors[-n + idx](s_tokens_deep[idx])
                s_proj, t_aligned = self._align_token_count(s_proj, t_tokens[idx])
                token_loss = token_loss + self.loss_token(s_proj, t_aligned, mask)
            token_loss = token_loss / max(1, n)

            shallow_k = max(0, len(s_maps) - self.deep_token_blocks)
            shallow_loss = torch.tensor(0.0, device=t_cls.device)
            for idx in range(shallow_k):
                s_map = self.shallow_student_projectors[idx](s_maps[idx])
                t_map = self.shallow_teacher_projectors[idx](t_tokens[idx], target_hw=s_map.shape[-2:])
                shallow_loss = shallow_loss + self._mse_strict(s_map, t_map.detach())
            shallow_loss = shallow_loss / max(1, shallow_k)

            cls_loss = self.loss_cls(self.cls_projector(s_cls_raw), t_cls)
            total = self.w_token * token_loss + self.w_shallow * shallow_loss + self.w_cls * cls_loss
            return {"total": total, "token": token_loss, "shallow": shallow_loss, "cls": cls_loss}

        if self.route == "C":
            n = min(len(s_maps), len(t_tokens))
            spatial = torch.tensor(0.0, device=t_cls.device)
            for idx in range(n):
                s_map = s_maps[idx]
                t_map = self.teacher_to_map_projectors[idx](t_tokens[idx], target_hw=s_map.shape[-2:])
                spatial = spatial + self.loss_spatial(s_map, t_map.detach(), mask)
            spatial = spatial / max(1, n)
            return {"total": self.w_spatial * spatial, "spatial": spatial}

        raise ValueError(f"Unsupported route: {self.route}")

    @torch.no_grad()
    def anomaly_map(self, teacher_out: Dict, student_out: Dict, image_size: int) -> torch.Tensor:
        t_tokens: List[torch.Tensor] = teacher_out["patch_tokens"]
        s_maps: List[torch.Tensor] = student_out["feature_maps"]
        maps = []
        n = min(len(s_maps), len(t_tokens))
        for idx in range(n):
            t_map = self.teacher_to_map_projectors[idx](t_tokens[idx], target_hw=s_maps[idx].shape[-2:])
            s_map = s_maps[idx]
            diff = (F.normalize(s_map, dim=1, eps=1e-8) - F.normalize(t_map, dim=1, eps=1e-8)).pow(2).mean(dim=1, keepdim=True)
            diff = F.interpolate(diff, size=(image_size, image_size), mode="bilinear", align_corners=False)
            maps.append(diff[:, 0])
        return torch.stack(maps).sum(dim=0) if maps else torch.zeros((s_maps[0].shape[0], image_size, image_size), device=s_maps[0].device)

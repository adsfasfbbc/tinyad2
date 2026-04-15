from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_ad.distillation.advanced_paradigm.losses import MGDDecoder, MGDLoss, SPKDLoss
from benchmark_ad.distillation.contrastive_loss import (
    AttentionKLLoss,
    CLSSimilarityLoss,
    SpatialContrastiveLoss,
    TokenContrastiveLoss,
)
from benchmark_ad.distillation.route_c_plus.losses import BoundaryAwareSpatialContrastiveLoss, GlobalCosineLoss
from benchmark_ad.models.advanced_paradigm.projectors import AdvancedParadigmProjector
from benchmark_ad.models.projectors import Conv1x1Projector, TokenMLPProjector, TokenToMapProjector
from benchmark_ad.models.route_c_plus.projectors import DepthwiseSeparableProjector


class HeterogeneousDistillationDispatcher(nn.Module):
    """
    Route strategies:
    - A: token + cls (attention distillation optional)
    - B: deep token + shallow feature align + cls
    - C: token->2D routing with branch-specific distillation (route-c-plus / advanced-paradigm)
    """

    def __init__(self, route: str, student_stage_channels: List[int], teacher_dim: int, cfg: Dict) -> None:
        super().__init__()
        self.route = str(route).strip().upper()
        self.student_stage_channels = [int(c) for c in student_stage_channels]
        self.teacher_dim = int(teacher_dim)
        self.deep_token_blocks = int(cfg.get("deep_token_blocks", 2))
        self.distill_branch = str(cfg.get("distill_branch", cfg.get("branch", "route_c_plus"))).strip().lower()

        self.loss_token = TokenContrastiveLoss(margin=float(cfg.get("token_margin", 0.5)))
        self.loss_attn = AttentionKLLoss()
        self.loss_cls = CLSSimilarityLoss()
        self.loss_spatial = SpatialContrastiveLoss(margin=float(cfg.get("spatial_margin", 0.3)))

        self.w_token = float(cfg.get("weight_token", 1.0))
        self.enable_attention_kl = bool(cfg.get("enable_attention_kl", False))
        self.w_attn = float(cfg.get("weight_attention", 0.0)) if self.enable_attention_kl else 0.0
        self.w_cls = float(cfg.get("weight_cls", 0.5))
        self.w_spatial = float(cfg.get("weight_spatial", 1.0))
        self.w_shallow = float(cfg.get("weight_shallow", 0.2))
        self.shallow_align_max_channels = int(cfg.get("shallow_align_max_channels", 512))

        route_plus_cfg = cfg.get("route_c_plus", {})
        self.route_plus_enabled = bool(route_plus_cfg.get("enabled", self.distill_branch == "route_c_plus"))
        self.route_plus_use_ds_projector = bool(route_plus_cfg.get("use_ds_projector", False))
        self.route_plus_normalize = bool(route_plus_cfg.get("normalize_features", True))
        self.route_plus_global = bool(route_plus_cfg.get("enable_global_cosine", False))
        self.route_plus_w_global = float(route_plus_cfg.get("weight_global", 0.0))
        self.route_plus_w_local = float(route_plus_cfg.get("weight_local", 1.0))
        self.route_plus_loss = BoundaryAwareSpatialContrastiveLoss(
            margin=float(route_plus_cfg.get("spatial_margin", cfg.get("spatial_margin", 0.3))),
            boundary_focus=bool(route_plus_cfg.get("boundary_focus", {}).get("enabled", False)),
            boundary_weight=float(route_plus_cfg.get("boundary_focus", {}).get("weight", 2.0)),
            boundary_dilation=int(route_plus_cfg.get("boundary_focus", {}).get("dilation", 1)),
            normalize_features=self.route_plus_normalize,
        )
        self.route_plus_global_loss = GlobalCosineLoss()

        advanced_cfg = cfg.get("advanced_paradigm", {})
        self.advanced_enabled = bool(advanced_cfg.get("enabled", self.distill_branch == "advanced_paradigm"))
        self.advanced_method = str(advanced_cfg.get("method", "mgd")).strip().lower()
        self.advanced_disable_direct = bool(advanced_cfg.get("disable_direct_alignment", True))
        self.advanced_w_mgd = float(advanced_cfg.get("weight_mgd", 1.0))
        self.advanced_w_spkd = float(advanced_cfg.get("weight_spkd", 0.0))
        self.mgd_loss = MGDLoss(
            mask_ratio_min=float(advanced_cfg.get("mgd_mask_ratio_min", 0.3)),
            mask_ratio_max=float(advanced_cfg.get("mgd_mask_ratio_max", 0.5)),
        )
        self.spkd_loss = SPKDLoss()

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
        self.spatial_align_dims = [min(int(c), self.teacher_dim, self.shallow_align_max_channels) for c in self.student_stage_channels]
        if self.route_plus_enabled and self.route_plus_use_ds_projector:
            self.student_spatial_projectors = nn.ModuleList(
                [DepthwiseSeparableProjector(in_ch=c, out_ch=d) for c, d in zip(self.student_stage_channels, self.spatial_align_dims)]
            )
        elif self.advanced_enabled:
            self.student_spatial_projectors = nn.ModuleList(
                [AdvancedParadigmProjector(in_ch=c, out_ch=d) for c, d in zip(self.student_stage_channels, self.spatial_align_dims)]
            )
        else:
            self.student_spatial_projectors = nn.ModuleList(
                [Conv1x1Projector(in_ch=c, out_ch=d) for c, d in zip(self.student_stage_channels, self.spatial_align_dims)]
            )
        self.teacher_spatial_projectors = nn.ModuleList(
            [TokenToMapProjector(in_dim=self.teacher_dim, out_ch=d) for d in self.spatial_align_dims]
        )
        self.mgd_decoders = nn.ModuleList([MGDDecoder(channels=d) for d in self.spatial_align_dims])

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
            raise ValueError(f"student-teacher shape mismatch for mse: student={tuple(a.shape)} teacher={tuple(b.shape)}")
        a = F.normalize(a, dim=1, eps=1e-8)
        b = F.normalize(b, dim=1, eps=1e-8)
        return F.mse_loss(a, b)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = tokens.shape
        side = int(n**0.5)
        if side * side != n and n > 1:
            maybe_side = int((n - 1) ** 0.5)
            if maybe_side * maybe_side == (n - 1):
                tokens = tokens[:, 1:, :]
                n = n - 1
                side = maybe_side
        if side * side != n:
            raise ValueError(f"Token count {n} cannot be reshaped to square map.")
        fmap = tokens.transpose(1, 2).reshape(b, d, side, side)
        if fmap.shape[-2:] != target_hw:
            fmap = F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
        return fmap

    def _route_c_forward(self, t_tokens: List[torch.Tensor], s_tokens_all: List[torch.Tensor], s_maps: List[torch.Tensor], mask: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        n = min(len(s_tokens_all), len(t_tokens), len(s_maps))

        if self.advanced_enabled and self.distill_branch == "advanced_paradigm":
            mgd = torch.tensor(0.0, device=device)
            spkd = torch.tensor(0.0, device=device)
            for idx in range(n):
                target_hw = s_maps[idx].shape[-2:]
                s_map = self._tokens_to_map(s_tokens_all[idx], target_hw=target_hw)
                s_map = self.student_spatial_projectors[idx](s_map)
                t_map = self.teacher_spatial_projectors[idx](t_tokens[idx], target_hw=s_map.shape[-2:])
                if self.advanced_method in {"mgd", "hybrid"}:
                    mgd = mgd + self.mgd_loss(s_map, t_map, self.mgd_decoders[idx])
                if self.advanced_method in {"spkd", "hybrid"} or self.advanced_w_spkd > 0.0:
                    spkd = spkd + self.spkd_loss(s_map, t_map)
            mgd = mgd / max(1, n)
            spkd = spkd / max(1, n)
            total = self.advanced_w_mgd * mgd + self.advanced_w_spkd * spkd
            return {"total": total, "mgd": mgd, "spkd": spkd}

        local = torch.tensor(0.0, device=device)
        global_loss = torch.tensor(0.0, device=device)
        for idx in range(n):
            target_hw = s_maps[idx].shape[-2:]
            s_map = self._tokens_to_map(s_tokens_all[idx], target_hw=target_hw)
            s_map = self.student_spatial_projectors[idx](s_map)
            t_map = self.teacher_spatial_projectors[idx](t_tokens[idx], target_hw=s_map.shape[-2:])

            if self.route_plus_enabled and self.distill_branch == "route_c_plus":
                local = local + self.route_plus_loss(s_map, t_map.detach(), mask)
                if self.route_plus_global:
                    global_loss = global_loss + self.route_plus_global_loss(
                        s_map,
                        t_map.detach(),
                        normalize_features=self.route_plus_normalize,
                    )
            else:
                s_map = F.normalize(s_map, dim=1, eps=1e-8)
                t_map = F.normalize(t_map, dim=1, eps=1e-8)
                local = local + self.loss_spatial(s_map, t_map.detach(), mask)

        local = local / max(1, n)
        global_loss = global_loss / max(1, n)
        if self.route_plus_enabled and self.distill_branch == "route_c_plus":
            total = self.route_plus_w_local * local + self.route_plus_w_global * global_loss
            return {"total": total, "spatial": local, "global": global_loss}
        return {"total": self.w_spatial * local, "spatial": local}

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
            for idx in range(n):
                s_proj = self.token_projectors[-n + idx](s_tokens_deep[idx])
                s_proj, t_aligned = self._align_token_count(s_proj, t_tokens[idx])
                token_loss = token_loss + self.loss_token(s_proj, t_aligned, mask)
            token_loss = token_loss / max(1, n)
            attn_loss = torch.zeros_like(token_loss)
            if self.enable_attention_kl and self.w_attn > 0.0:
                for idx in range(min(len(s_attn), len(t_attn))):
                    s_att = s_attn[idx]
                    t_att_map = t_attn[idx]
                    if s_att.shape != t_att_map.shape:
                        m = min(s_att.shape[-1], t_att_map.shape[-1])
                        s_att = s_att[:, :m, :m]
                        t_att_map = t_att_map[:, :m, :m]
                    attn_loss = attn_loss + self.loss_attn(s_att, t_att_map)
                attn_loss = attn_loss / max(1, min(len(s_attn), len(t_attn)))
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
            return self._route_c_forward(t_tokens=t_tokens, s_tokens_all=s_tokens_all, s_maps=s_maps, mask=mask, device=t_cls.device)

        raise ValueError(f"Unsupported route: {self.route}")

    @torch.no_grad()
    def anomaly_map(self, teacher_out: Dict, student_out: Dict, image_size: int) -> torch.Tensor:
        t_tokens: List[torch.Tensor] = teacher_out["patch_tokens"]
        s_maps: List[torch.Tensor] = student_out["feature_maps"]
        s_tokens_all: List[torch.Tensor] = student_out["tokens_all"]
        maps = []
        n = min(len(s_maps), len(t_tokens), len(s_tokens_all))
        for idx in range(n):
            target_hw = s_maps[idx].shape[-2:]
            s_map = self._tokens_to_map(s_tokens_all[idx], target_hw=target_hw)
            s_map = self.student_spatial_projectors[idx](s_map)
            t_map = self.teacher_spatial_projectors[idx](t_tokens[idx], target_hw=s_map.shape[-2:])
            if self.route_plus_enabled and self.distill_branch == "route_c_plus":
                if self.route_plus_normalize:
                    s_map = F.normalize(s_map, p=2, dim=1, eps=1e-8)
                    t_map = F.normalize(t_map, p=2, dim=1, eps=1e-8)
            diff = (F.normalize(s_map, dim=1, eps=1e-8) - F.normalize(t_map, dim=1, eps=1e-8)).pow(2).mean(dim=1, keepdim=True)
            diff = F.interpolate(diff, size=(image_size, image_size), mode="bilinear", align_corners=False)
            maps.append(diff[:, 0])
        return (
            torch.stack(maps).sum(dim=0)
            if maps
            else torch.zeros((s_maps[0].shape[0], image_size, image_size), device=s_maps[0].device)
        )

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import VisualAD_lib
from dataset import Dataset
from student import TimmStudent
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.feature_transform import create_feature_transform
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
from utils.transforms import get_transform


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        p, r, _ = precision_recall_curve(y_true, y_score)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        return float(np.max(f1))
    except ValueError:
        return float("nan")


def _avg_ignore_nan(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if valid.size else float("nan")


def _count_params(model: torch.nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def _estimate_flops_conv_linear(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    total_flops = 0.0
    hooks = []

    def conv_hook(mod: torch.nn.Conv2d, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        nonlocal total_flops
        out_h, out_w = out.shape[-2], out.shape[-1]
        kernel_ops = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels / mod.groups)
        total_flops += float(out.shape[0] * out_h * out_w * mod.out_channels * kernel_ops * 2.0)

    def _infer_linear_batch(inp: Tuple[torch.Tensor]) -> int:
        if not inp:
            return 1
        x = inp[0]
        if x.dim() > 1:
            return int(np.prod(x.shape[:-1]))
        return int(x.shape[0])

    def linear_hook(mod: torch.nn.Linear, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        nonlocal total_flops
        in_features = mod.in_features
        out_features = mod.out_features
        batch = _infer_linear_batch(inp)
        total_flops += float(batch * in_features * out_features * 2.0)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()
    return total_flops / 1e9


def _onnx_param_count(onnx_path: Path) -> Optional[float]:
    if not onnx_path.exists():
        return None
    try:
        import onnx  # type: ignore
    except Exception:
        return None

    model = onnx.load(str(onnx_path))
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += n
    return float(total / 1e6)


@dataclass
class EvalResult:
    image_auroc: float
    pixel_auroc: float
    image_map: float
    image_f1: float
    pixel_map: float
    pixel_f1: float


class BaseAdapter:
    def predict_map(self, image: torch.Tensor, image_size: int) -> torch.Tensor:
        raise NotImplementedError

    def stat_module(self) -> torch.nn.Module:
        raise NotImplementedError

    def stat_forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VisualADTeacherAdapter(BaseAdapter):
    def __init__(self, checkpoint_path: Path, backbone: str, device: torch.device) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        self.features_list = checkpoint.get("features_list", [6, 12, 18, 24])
        self.image_size = int(checkpoint.get("image_size", 518))

        self.model, _ = VisualAD_lib.load(backbone, device=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

        self.model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
        self.model.visual.normal_token.data = checkpoint["normal_token"].to(device)

        self.layer_transforms = torch.nn.ModuleDict()
        if "layer_transforms" in checkpoint:
            feature_dim = self.model.visual.embed_dim
            for layer_name, state_dict in checkpoint["layer_transforms"].items():
                hidden_dim = state_dict["mlp.0.weight"].shape[0]
                transform = create_feature_transform(
                    transform_type="mlp",
                    input_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=feature_dim,
                    dropout=0.0,
                ).to(device)
                transform.load_state_dict(state_dict)
                transform.eval()
                self.layer_transforms[layer_name] = transform

    def predict_map(self, image: torch.Tensor, image_size: int) -> torch.Tensor:
        with torch.no_grad():
            vision_output = self.model.encode_image(image, self.features_list)
            anomaly_features = vision_output["anomaly_features"]
            normal_features = vision_output["normal_features"]
            patch_tokens = vision_output["patch_tokens"]
            patch_start_idx = vision_output["patch_start_idx"]

            maps = []
            for idx, patch_feature in enumerate(patch_tokens):
                a_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
                n_norm = F.normalize(normal_features, dim=1, eps=1e-8)
                layer_key = f"layer_{self.features_list[idx]}"
                if layer_key in self.layer_transforms:
                    b, n, d = patch_feature.shape
                    patch_feature = self.layer_transforms[layer_key](patch_feature.reshape(-1, d)).reshape(b, n, d)
                amap = generate_anomaly_map_from_tokens(
                    a_norm, n_norm, patch_feature[:, patch_start_idx:, :], image_size
                )
                maps.append(amap)
            return torch.stack(maps).sum(dim=0)

    def stat_module(self) -> torch.nn.Module:
        return self.model.visual

    def stat_forward(self, image: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_image(image, self.features_list)
        return out["patch_tokens"][-1]


class TimmDistillAdapter(BaseAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        student_backbone: str,
        teacher_backbone: str,
        feature_out_indices: List[int],
        teacher_channels: Optional[List[int]],
        device: torch.device,
        pretrained: bool = True,
    ) -> None:
        import timm

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        self.device = device
        self.feature_out_indices = [int(v) for v in feature_out_indices]

        self.teacher = timm.create_model(
            teacher_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(self.feature_out_indices),
        ).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        resolved_teacher_channels = (
            list(teacher_channels)
            if teacher_channels is not None
            else list(self.teacher.feature_info.channels())
        )

        self.student = TimmStudent(
            backbone_name=student_backbone,
            feature_out_indices=self.feature_out_indices,
            teacher_channels=resolved_teacher_channels,
            pretrained=pretrained,
        ).to(device)
        self.student.eval()

        if "student_state_dict" in checkpoint:
            self.student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        else:
            self.student.load_state_dict(checkpoint, strict=False)

    def predict_map(self, image: torch.Tensor, image_size: int) -> torch.Tensor:
        with torch.no_grad():
            student_out = self.student(image)["aligned_features"]
            teacher_out = self.teacher(image)

        maps = []
        for s_feat, t_feat in zip(student_out, teacher_out):
            diff = (s_feat - t_feat).pow(2).mean(dim=1, keepdim=True)
            diff = F.interpolate(diff, size=(image_size, image_size), mode="bilinear", align_corners=False)
            maps.append(diff[:, 0])
        return torch.stack(maps).sum(dim=0)

    def stat_module(self) -> torch.nn.Module:
        return self.student

    def stat_forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.student(image)["aligned_features"][-1]


def _evaluate_single_dataset(
    adapter: BaseAdapter,
    dataset_name: str,
    dataset_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    sigma: int,
    device: torch.device,
) -> EvalResult:
    args_ns = SimpleNamespace(image_size=image_size)
    preprocess, target_transform = get_transform(args_ns)
    data = Dataset(
        root=dataset_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name,
        mode="test",
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true_img: List[float] = []
    y_score_img: List[float] = []
    y_true_px: List[np.ndarray] = []
    y_score_px: List[np.ndarray] = []

    for batch in loader:
        image = batch["img"].to(device, non_blocking=True)
        gt = batch["img_mask"].to(device, non_blocking=True)
        gt = (gt > 0.5).float()
        amap = adapter.predict_map(image, image_size=image_size)
        amap_np = amap.detach().cpu().numpy()
        amap_filtered = np.stack([gaussian_filter(sample, sigma=sigma) for sample in amap_np], axis=0)

        for i in range(amap_filtered.shape[0]):
            score = reduce_anomaly_map(torch.from_numpy(amap_filtered[i]), mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
            y_true_img.append(float(batch["anomaly"][i].item()))
            y_score_img.append(float(score.item()))
            y_true_px.append(gt[i, 0].detach().cpu().numpy().reshape(-1))
            y_score_px.append(amap_filtered[i].reshape(-1))

    gt_img = np.array(y_true_img, dtype=np.float64)
    pr_img = np.array(y_score_img, dtype=np.float64)
    gt_px = np.concatenate(y_true_px, axis=0) if y_true_px else np.array([], dtype=np.float64)
    pr_px = np.concatenate(y_score_px, axis=0) if y_score_px else np.array([], dtype=np.float64)

    return EvalResult(
        image_auroc=_safe_auroc(gt_img, pr_img),
        pixel_auroc=_safe_auroc(gt_px, pr_px),
        image_map=_safe_ap(gt_img, pr_img),
        image_f1=_safe_f1(gt_img, pr_img),
        pixel_map=_safe_ap(gt_px, pr_px),
        pixel_f1=_safe_f1(gt_px, pr_px),
    )


def _measure_fps(adapter: BaseAdapter, image_size: int, device: torch.device, warmup: int, iters: int) -> float:
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    for _ in range(warmup):
        _ = adapter.predict_map(dummy, image_size=image_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = adapter.predict_map(dummy, image_size=image_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return float(iters / max(elapsed, 1e-8))


def _format_pct(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v * 100.0:.2f}"


def _build_adapter(model_cfg: Dict[str, Any], device: torch.device) -> BaseAdapter:
    model_type = str(model_cfg["type"]).strip().lower()
    if model_type == "visualad_teacher":
        checkpoint = str(model_cfg["checkpoint"])
        if "REPLACE_WITH_" in checkpoint:
            raise ValueError(f"Model '{model_cfg.get('name', 'unknown')}' checkpoint is still placeholder: {checkpoint}")
        return VisualADTeacherAdapter(
            checkpoint_path=Path(checkpoint),
            backbone=str(model_cfg.get("backbone", "ViT-L/14@336px")),
            device=device,
        )
    if model_type == "timm_student":
        checkpoint = str(model_cfg["checkpoint"])
        if "REPLACE_WITH_" in checkpoint:
            raise ValueError(f"Model '{model_cfg.get('name', 'unknown')}' checkpoint is still placeholder: {checkpoint}")
        return TimmDistillAdapter(
            checkpoint_path=Path(checkpoint),
            student_backbone=str(model_cfg["student_backbone"]),
            teacher_backbone=str(model_cfg.get("teacher_backbone", "wide_resnet50_2")),
            feature_out_indices=[int(v) for v in model_cfg.get("feature_out_indices", [1, 2, 3, 4])],
            teacher_channels=model_cfg.get("teacher_channels"),
            device=device,
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _markdown_table(rows: List[List[str]]) -> str:
    headers = ["Model", "Params (M)", "FLOPs (G)", "FPS", "Image-AUROC", "Pixel-AUROC", "MAP", "F1-SCORE"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser("Unified benchmark for MVTec/VisA distillation models")
    parser.add_argument("--config", type=str, default="configs/eval_benchmark.yaml")
    parser.add_argument("--output_dir", type=str, default="experiments/eval_benchmark")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    runtime = cfg.get("runtime", {})
    image_size = int(runtime.get("image_size", 256))
    batch_size = int(runtime.get("batch_size", 1))
    num_workers = int(runtime.get("num_workers", 2))
    sigma = int(runtime.get("sigma", 4))
    warmup = int(runtime.get("fps_warmup", 10))
    fps_iters = int(runtime.get("fps_iters", 50))
    device = torch.device(runtime.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    datasets = cfg.get("datasets", [])
    models = cfg.get("models", [])
    if not datasets or not models:
        raise ValueError("config must contain non-empty datasets and models")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_lines = []
    final_rows = []

    for model_cfg in models:
        model_name = str(model_cfg["name"])
        adapter = _build_adapter(model_cfg, device)

        params_m = _count_params(adapter.stat_module())
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        flops_g = _estimate_flops_conv_linear(adapter.stat_module(), dummy)
        fps = _measure_fps(adapter, image_size=image_size, device=device, warmup=warmup, iters=fps_iters)

        onnx_path = model_cfg.get("onnx_path")
        onnx_params_m = None
        if onnx_path:
            onnx_params_m = _onnx_param_count(Path(onnx_path))

        dataset_metrics = []
        for ds in datasets:
            ds_name = str(ds["name"])
            ds_path = str(ds["path"])
            if "REPLACE_WITH_" in ds_path:
                raise ValueError(f"Dataset path for '{ds_name}' is still placeholder: {ds_path}")
            res = _evaluate_single_dataset(
                adapter=adapter,
                dataset_name=ds_name,
                dataset_path=ds_path,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                sigma=sigma,
                device=device,
            )
            dataset_metrics.append((ds_name, res))
            detail_lines.append(
                ",".join(
                    [
                        model_name,
                        ds_name,
                        f"{res.image_auroc:.6f}",
                        f"{res.pixel_auroc:.6f}",
                        f"{res.image_map:.6f}",
                        f"{res.image_f1:.6f}",
                        f"{res.pixel_map:.6f}",
                        f"{res.pixel_f1:.6f}",
                    ]
                )
            )

        image_auroc_avg = _avg_ignore_nan([m.image_auroc for _, m in dataset_metrics])
        pixel_auroc_avg = _avg_ignore_nan([m.pixel_auroc for _, m in dataset_metrics])
        map_avg = _avg_ignore_nan([m.image_map for _, m in dataset_metrics])
        f1_avg = _avg_ignore_nan([m.image_f1 for _, m in dataset_metrics])

        params_text = f"{params_m:.3f}"
        if onnx_params_m is not None:
            params_text = f"{params_m:.3f} (pth) / {onnx_params_m:.3f} (onnx)"

        final_rows.append(
            [
                model_name,
                params_text,
                f"{flops_g:.3f}",
                f"{fps:.2f}",
                _format_pct(image_auroc_avg),
                _format_pct(pixel_auroc_avg),
                _format_pct(map_avg),
                _format_pct(f1_avg),
            ]
        )

    table_md = _markdown_table(final_rows)
    (output_dir / "core_table.md").write_text(table_md + "\n", encoding="utf-8")

    detail_header = "model,dataset,image_auroc,pixel_auroc,image_map,image_f1,pixel_map,pixel_f1"
    (output_dir / "per_dataset_metrics.csv").write_text(
        detail_header + "\n" + "\n".join(detail_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "core_table.csv").write_text(
        "Model,Params (M),FLOPs (G),FPS,Image-AUROC,Pixel-AUROC,MAP,F1-SCORE\n"
        + "\n".join([",".join(r) for r in final_rows])
        + "\n",
        encoding="utf-8",
    )

    print(table_md)
    print(f"\nSaved: {output_dir / 'core_table.md'}")
    print(f"Saved: {output_dir / 'core_table.csv'}")
    print(f"Saved: {output_dir / 'per_dataset_metrics.csv'}")


if __name__ == "__main__":
    main()

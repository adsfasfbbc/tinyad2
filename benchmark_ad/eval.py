from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from benchmark_ad.datasets import UnifiedMVTecDataset
from benchmark_ad.distillation import HeterogeneousDistillationDispatcher
from benchmark_ad.models import VisualADTeacherViTL14, build_student
from utils.transforms import get_transform


def load_yaml(path: str) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _teacher_to_dict(out) -> Dict:
    return {"cls_token": out.cls_token, "patch_tokens": out.patch_tokens, "attention_maps": out.attention_maps}


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


def _safe_acc_at_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        p, r, thresholds = precision_recall_curve(y_true, y_score)
        if thresholds.size == 0:
            return float("nan")
        f1 = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-8)
        best_idx = int(np.nanargmax(f1))
        best_threshold = thresholds[best_idx]
        pred = (y_score >= best_threshold).astype(np.float64)
        return float((pred == y_true).mean())
    except ValueError:
        return float("nan")


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

    def linear_hook(mod: torch.nn.Linear, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        nonlocal total_flops
        x = inp[0]
        batch = int(np.prod(x.shape[:-1])) if x.dim() > 1 else int(x.shape[0])
        total_flops += float(batch * mod.in_features * mod.out_features * 2.0)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        _ = model(input_tensor)
    for h in hooks:
        h.remove()
    return total_flops / 1e9


@torch.no_grad()
def _benchmark_fps(model: torch.nn.Module, input_tensor: torch.Tensor, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(max(1, warmup)):
        _ = model(input_tensor)
    if input_tensor.device.type == "cuda":
        torch.cuda.synchronize(input_tensor.device)
    t0 = time.time()
    for _ in range(max(1, iters)):
        _ = model(input_tensor)
    if input_tensor.device.type == "cuda":
        torch.cuda.synchronize(input_tensor.device)
    elapsed = time.time() - t0
    return float(max(1, iters) / max(elapsed, 1e-8))


def _save_heatmap(anomaly_map: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(anomaly_map, cmap="jet")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def _reduce_anomaly_map_topk(anomaly_map: torch.Tensor, k: int = 100) -> torch.Tensor:
    if anomaly_map.dim() < 2:
        raise ValueError(f"anomaly_map must have at least 2 dims (B, ...), got {tuple(anomaly_map.shape)}")
    flat_map = anomaly_map.reshape(anomaly_map.shape[0], -1)
    topk = max(1, min(int(k), int(flat_map.shape[1])))
    return torch.topk(flat_map, k=topk, dim=1).values.mean(dim=1)


def _gaussian_smooth_anomaly_map(anomaly_map: torch.Tensor, kernel_size: int = 33, sigma: float = 4.0) -> torch.Tensor:
    """Apply torchvision Gaussian blur to anomaly maps before image-level scoring.

    Args:
        anomaly_map: Tensor with shape [B, H, W] or [B, C, H, W].
        kernel_size: Gaussian kernel size; values <= 1 disable smoothing.
            Even values are auto-adjusted to the next odd value.
        sigma: Gaussian sigma; values <= 0 disable smoothing.

    Returns:
        Smoothed tensor with the same shape as input.
    """
    if float(sigma) <= 0.0 or int(kernel_size) <= 1:
        return anomaly_map
    k = int(kernel_size)
    s = float(sigma)
    if k % 2 == 0:
        k += 1
    if anomaly_map.dim() == 3:
        return tvf.gaussian_blur(anomaly_map.unsqueeze(1), kernel_size=[k, k], sigma=[s, s]).squeeze(1)
    if anomaly_map.dim() == 4:
        return tvf.gaussian_blur(anomaly_map, kernel_size=[k, k], sigma=[s, s])
    raise ValueError(f"anomaly_map must be 3D or 4D, got {tuple(anomaly_map.shape)}")


def _normalize_anomaly_map_zscore(anomaly_map: torch.Tensor, mean: float, std: float, eps: float = 1e-6) -> torch.Tensor:
    return (anomaly_map - float(mean)) / max(float(std), float(eps))


@torch.no_grad()
def _compute_train_normal_anomaly_stats(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    dispatcher: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    image_size: int,
) -> Dict[str, Dict[str, float]]:
    stats = defaultdict(lambda: {"sum": 0.0, "sq_sum": 0.0, "count": 0})

    for images, _, labels, cls_names, _ in train_loader:
        images = images.to(next(student.parameters()).device, non_blocking=True)
        labels = labels.float()
        t_out = teacher(images)
        s_out = student(images)
        amap = dispatcher.anomaly_map(_teacher_to_dict(t_out), s_out, image_size=image_size)
        amap = torch.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()

        for i in range(amap.shape[0]):
            if float(labels[i].item()) != 0.0:
                continue
            cls_name = str(cls_names[i])
            vals = amap[i].reshape(-1)
            v_sum = float(vals.sum().item())
            v_sq_sum = float((vals * vals).sum().item())
            v_count = int(vals.numel())
            stats[cls_name]["sum"] += v_sum
            stats[cls_name]["sq_sum"] += v_sq_sum
            stats[cls_name]["count"] += v_count
            stats["__global__"]["sum"] += v_sum
            stats["__global__"]["sq_sum"] += v_sq_sum
            stats["__global__"]["count"] += v_count

    out: Dict[str, Dict[str, float]] = {}
    for k, v in stats.items():
        cnt = int(v["count"])
        if cnt <= 0:
            continue
        mean = float(v["sum"] / cnt)
        var = max(float(v["sq_sum"] / cnt) - mean * mean, 0.0)
        out[k] = {"mean": mean, "std": max(float(math.sqrt(var)), 1e-6), "count": float(cnt)}
    if "__global__" not in out:
        raise ValueError("No normal samples found in train set for Z-score statistics.")
    return out


@torch.no_grad()
def evaluate(
    config: Dict,
    checkpoint_path: Path,
    save_dir: Path,
    save_heatmaps: bool = True,
    max_heatmaps: int = 50,
    image_topk: int = 100,
    image_blur_kernel: int = 33,
    image_blur_sigma: float = 4.0,
) -> Dict:
    runtime = config.get("runtime", {})
    data_cfg = config.get("data", {})
    teacher_cfg = config.get("teacher", {})
    student_cfg = config.get("student", {})
    distill_cfg = config.get("distill", {})

    device = torch.device(runtime.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    image_size = int(runtime.get("image_size", 256))
    num_workers = int(runtime.get("num_workers", 4))
    dataset_root = str(data_cfg.get("root", ""))
    category = str(data_cfg.get("category", "all"))
    if not dataset_root or "REPLACE_WITH_YOUR" in dataset_root:
        raise ValueError("Please set data.root to your real dataset path.")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    teacher_layers = checkpoint.get("teacher_layers", teacher_cfg.get("features_list", [6, 12, 18, 24]))
    teacher = VisualADTeacherViTL14(features_list=teacher_layers, device=device)
    student_name = str(checkpoint.get("student_name", student_cfg.get("name", "mobilevit_s")))
    student_out_indices = checkpoint.get("student_out_indices", student_cfg.get("out_indices"))
    student = build_student(student_name, out_indices=student_out_indices, pretrained=False).to(device)
    student.load_state_dict(checkpoint["student_state_dict"], strict=False)
    student.eval()

    dispatcher = HeterogeneousDistillationDispatcher(
        route=str(checkpoint.get("route", student.spec.route)),
        student_stage_channels=student.stage_channels,
        teacher_dim=teacher.embed_dim,
        cfg=distill_cfg,
    ).to(device)
    dispatcher.load_state_dict(checkpoint["dispatcher_state_dict"], strict=False)
    dispatcher.eval()

    t_args = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(t_args)
    train_set = UnifiedMVTecDataset(
        root=dataset_root,
        transform=preprocess,
        target_transform=target_transform,
        mode="train",
        category=category,
        synth_prob=0.0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=max(1, int(runtime.get("batch_size", 8))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    normal_stats = _compute_train_normal_anomaly_stats(teacher, student, dispatcher, train_loader, image_size=image_size)
    stats_path = save_dir / "normal_map_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(normal_stats, f, ensure_ascii=False, indent=2)

    test_set = UnifiedMVTecDataset(
        root=dataset_root,
        transform=preprocess,
        target_transform=target_transform,
        mode="test",
        category=category,
        synth_prob=0.0,
    )
    loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    image_labels: List[float] = []
    image_scores: List[float] = []
    pixel_labels: List[float] = []
    pixel_scores: List[float] = []
    heatmap_count = 0

    for images, masks, labels, cls_names, img_paths in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        t_out = teacher(images)
        s_out = student(images)
        amap = dispatcher.anomaly_map(_teacher_to_dict(t_out), s_out, image_size=image_size)
        amap = torch.nan_to_num(amap, nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(amap.shape[0]):
            cls_key = str(cls_names[i])
            cls_stat = normal_stats.get(cls_key, normal_stats["__global__"])
            amap[i] = _normalize_anomaly_map_zscore(amap[i], mean=cls_stat["mean"], std=cls_stat["std"])

        amap_for_image = _gaussian_smooth_anomaly_map(amap, kernel_size=image_blur_kernel, sigma=image_blur_sigma)
        img_score = _reduce_anomaly_map_topk(amap_for_image, k=image_topk)
        image_labels.extend(labels.detach().cpu().numpy().tolist())
        image_scores.extend(img_score.detach().cpu().numpy().tolist())

        pixel_labels.extend(masks.detach().cpu().reshape(-1).numpy().tolist())
        pixel_scores.extend(amap.detach().cpu().reshape(-1).numpy().tolist())

        if save_heatmaps and heatmap_count < max_heatmaps:
            cls_name = str(cls_names[0])
            stem = Path(str(img_paths[0])).stem
            hm_path = save_dir / "heatmaps" / cls_name / f"{stem}_heatmap.png"
            _save_heatmap(amap[0].detach().cpu().numpy(), hm_path)
            heatmap_count += 1

    image_true = np.array(image_labels, dtype=np.float64)
    image_pred = np.array(image_scores, dtype=np.float64)
    pixel_true = np.array(pixel_labels, dtype=np.float64)
    pixel_pred = np.array(pixel_scores, dtype=np.float64)

    metrics = {
        "image_auroc": _safe_auroc(image_true, image_pred),
        "pixel_auroc": _safe_auroc(pixel_true, pixel_pred),
        "image_map": _safe_ap(image_true, image_pred),
        "image_f1": _safe_f1(image_true, image_pred),
        "image_accuracy": _safe_acc_at_best_f1_threshold(image_true, image_pred),
        "pixel_map": _safe_ap(pixel_true, pixel_pred),
        "pixel_f1": _safe_f1(pixel_true, pixel_pred),
        "pixel_accuracy": _safe_acc_at_best_f1_threshold(pixel_true, pixel_pred),
    }
    # Backward compatibility for existing consumers that still read *_ap keys.
    # TODO: remove *_ap aliases after downstream scripts fully migrate to *_map naming.
    metrics["image_ap"] = metrics["image_map"]
    metrics["pixel_ap"] = metrics["pixel_map"]
    return metrics


def evaluate_latency(
    checkpoint_path: Path,
    student_name: Optional[str],
    image_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    resolved_student_name = student_name or str(ckpt.get("student_name", ""))
    if not resolved_student_name:
        raise ValueError("Cannot resolve student_name; pass --student_name or ensure checkpoint has student_name.")
    student = build_student(
        model_name=resolved_student_name,
        out_indices=ckpt.get("student_out_indices"),
        pretrained=False,
    ).to(device)
    student.load_state_dict(ckpt["student_state_dict"], strict=False)
    student.eval()

    x = torch.randn(1, 3, int(image_size), int(image_size), device=device)
    return {
        "params_m": _count_params(student),
        "flops_g": _estimate_flops_conv_linear(student, x),
        "fps": _benchmark_fps(student, x, warmup=int(warmup), iters=int(iters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser("benchmark_ad unified eval+latency")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./experiments/benchmark_ad_eval")
    parser.add_argument("--no_heatmap", action="store_true")
    parser.add_argument("--max_heatmaps", type=int, default=50)
    parser.add_argument("--run_latency", action="store_true")
    parser.add_argument("--latency_only", action="store_true")
    parser.add_argument("--student_name", type=str, default="")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--image_topk", type=int, default=100)
    parser.add_argument("--image_blur_kernel", type=int, default=33)
    parser.add_argument("--image_blur_sigma", type=float, default=4.0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Dict[str, float]] = {}

    if not args.latency_only:
        if not args.config:
            raise ValueError("--config is required unless --latency_only is set.")
        cfg = load_yaml(args.config)
        metrics = evaluate(
            config=cfg,
            checkpoint_path=Path(args.checkpoint),
            save_dir=save_dir,
            save_heatmaps=not args.no_heatmap,
            max_heatmaps=int(args.max_heatmaps),
            image_topk=int(args.image_topk),
            image_blur_kernel=int(args.image_blur_kernel),
            image_blur_sigma=float(args.image_blur_sigma),
        )
        out_path = save_dir / "metrics.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        outputs["metrics"] = metrics
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"saved metrics: {out_path}")

    if args.run_latency or args.latency_only:
        latency = evaluate_latency(
            checkpoint_path=Path(args.checkpoint),
            student_name=args.student_name.strip() or None,
            image_size=int(args.image_size),
            device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        latency_path = save_dir / "latency.json"
        with latency_path.open("w", encoding="utf-8") as f:
            json.dump(latency, f, ensure_ascii=False, indent=2)
        outputs["latency"] = latency
        print(json.dumps(latency, ensure_ascii=False, indent=2))
        print(f"saved latency: {latency_path}")

    if len(outputs) > 1:
        summary_path = save_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from benchmark_ad.datasets import UnifiedMVTecDataset
from benchmark_ad.distillation import HeterogeneousDistillationDispatcher
from benchmark_ad.models import VisualADTeacherViTL14, build_student
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
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


def _save_heatmap(anomaly_map: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(anomaly_map, cmap="jet")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


@torch.no_grad()
def evaluate(config: Dict, checkpoint_path: Path, save_dir: Path, save_heatmaps: bool = True, max_heatmaps: int = 50) -> Dict:
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

        img_score = reduce_anomaly_map(amap, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO)
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
    metrics["image_ap"] = metrics["image_map"]
    metrics["pixel_ap"] = metrics["pixel_map"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser("benchmark_ad eval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./experiments/benchmark_ad_eval")
    parser.add_argument("--no_heatmap", action="store_true")
    parser.add_argument("--max_heatmaps", type=int, default=50)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        config=cfg,
        checkpoint_path=Path(args.checkpoint),
        save_dir=save_dir,
        save_heatmaps=not args.no_heatmap,
        max_heatmaps=int(args.max_heatmaps),
    )
    out_path = save_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved metrics: {out_path}")


if __name__ == "__main__":
    main()

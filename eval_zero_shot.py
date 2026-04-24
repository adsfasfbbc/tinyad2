from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from dataset import Dataset as ADDataset
from models import DistillationAdapter, VMambaStudent
from utils.distill_transforms import get_zero_shot_eval_transform, get_zero_shot_mask_transform
from utils.metrics import cal_pro_score
from utils.scoring import reduce_anomaly_map


def _load_prompt_embeddings(path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location=device)
    if isinstance(data, dict):
        if "normal" in data and "anomaly" in data:
            normal = data["normal"].to(device)
            anomaly = data["anomaly"].to(device)
        elif "embeddings" in data and data["embeddings"].shape[0] >= 2:
            normal = data["embeddings"][0].to(device)
            anomaly = data["embeddings"][1].to(device)
        else:
            raise ValueError("Prompt embedding dict must contain normal/anomaly or embeddings.")
    elif torch.is_tensor(data) and data.ndim == 2 and data.shape[0] >= 2:
        normal, anomaly = data[0].to(device), data[1].to(device)
    else:
        raise ValueError("Unsupported prompt embedding file format.")

    return F.normalize(normal, dim=0), F.normalize(anomaly, dim=0)


def _dense_to_map(dense: torch.Tensor, normal: torch.Tensor, anomaly: torch.Tensor, out_hw: int) -> torch.Tensor:
    # dense: [B, C, H, W] -> anomaly logit map [B, out_hw, out_hw]
    dense = F.normalize(dense, dim=1)
    anom = (dense * anomaly.view(1, -1, 1, 1)).sum(dim=1)
    norm = (dense * normal.view(1, -1, 1, 1)).sum(dim=1)
    score = anom - norm
    return F.interpolate(score.unsqueeze(1), size=(out_hw, out_hw), mode="bilinear", align_corners=False).squeeze(1)


def _find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    unique = np.unique(y_score)
    if unique.size > 512:
        qs = np.linspace(0.0, 1.0, 513)
        unique = np.quantile(y_score, qs)
    best_t, best_f1 = 0.0, -1.0
    for t in unique:
        pred = (y_score >= t).astype(np.int32)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser("VMamba-VisualAD Zero-shot Evaluation")
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True, choices=["mvtec", "visa"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True, help=".pt file with normal/anomaly text embeddings")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow_fallback_student", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    preprocess = get_zero_shot_eval_transform(args.image_size)
    target_transform = get_zero_shot_mask_transform(args.image_size)

    ds = ADDataset(
        root=args.test_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.test_dataset,
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint_path, map_location=device)

    student = VMambaStudent(use_fallback_if_unavailable=args.allow_fallback_student).to(device)
    stage_channels = ckpt.get("student_stage_channels", student.stage_channels)
    adapter = DistillationAdapter(stage_channels=stage_channels, output_dim=1024, target_hw=24).to(device)
    student.load_state_dict(ckpt["student"], strict=False)
    adapter.load_state_dict(ckpt["adapter"], strict=False)
    student.eval()
    adapter.eval()

    normal_text, anomaly_text = _load_prompt_embeddings(args.prompt_path, device)

    image_gt: List[int] = []
    image_scores: List[float] = []
    pixel_gt: List[np.ndarray] = []
    pixel_scores: List[np.ndarray] = []

    with torch.no_grad():
        for items in tqdm(dl):
            image = items["img"].to(device)
            masks = items["img_mask"].squeeze(1).cpu().numpy().astype(np.uint8)  # [B, H, W]
            labels = items["anomaly"].cpu().numpy().astype(np.int32)  # [B]

            student_out = student(image)
            adapter_out = adapter(student_out)

            # Use high-level stage fusion (stage3 + stage4)
            dense = 0.5 * (adapter_out.dense[3] + adapter_out.dense[4])
            amap_batch = _dense_to_map(dense, normal_text, anomaly_text, out_hw=args.image_size).cpu().numpy()

            for idx in range(amap_batch.shape[0]):
                amap = gaussian_filter(amap_batch[idx], sigma=args.sigma)
                image_score = float(
                    reduce_anomaly_map(
                        torch.from_numpy(amap).unsqueeze(0),
                        mode="topk_mean",
                        topk_ratio=0.01,
                    )[0]
                )

                image_gt.append(int(labels[idx]))
                image_scores.append(image_score)
                pixel_gt.append(masks[idx].flatten())
                pixel_scores.append(amap.flatten())

    image_gt_np = np.asarray(image_gt)
    image_scores_np = np.asarray(image_scores)
    pixel_gt_np = np.concatenate(pixel_gt)
    pixel_scores_np = np.concatenate(pixel_scores)

    image_auroc = roc_auc_score(image_gt_np, image_scores_np)
    pixel_auroc = roc_auc_score(pixel_gt_np, pixel_scores_np)

    image_th = _find_best_f1_threshold(image_gt_np, image_scores_np)
    pixel_th = _find_best_f1_threshold(pixel_gt_np, pixel_scores_np)

    image_pred = (image_scores_np >= image_th).astype(np.int32)
    pixel_pred = (pixel_scores_np >= pixel_th).astype(np.int32)

    image_acc = accuracy_score(image_gt_np, image_pred)
    pixel_acc = accuracy_score(pixel_gt_np, pixel_pred)
    image_f1 = f1_score(image_gt_np, image_pred, zero_division=0)
    pixel_f1 = f1_score(pixel_gt_np, pixel_pred, zero_division=0)

    # PRO-score
    masks_reshaped = np.stack([x.reshape(args.image_size, args.image_size) for x in pixel_gt])
    amaps = np.stack([x.reshape(args.image_size, args.image_size) for x in pixel_scores])
    pro_score = cal_pro_score(masks=masks_reshaped, amaps=amaps)

    print("=== Zero-shot Evaluation Results ===")
    print(f"Image-AUROC: {image_auroc:.4f}")
    print(f"Pixel-AUROC: {pixel_auroc:.4f}")
    print(f"PRO-score:   {pro_score:.4f}")
    print(f"Image-ACC:   {image_acc:.4f}")
    print(f"Pixel-ACC:   {pixel_acc:.4f}")
    print(f"Image-F1:    {image_f1:.4f}")
    print(f"Pixel-F1:    {pixel_f1:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

from benchmark_ad.datasets import HybridSyntheticAnomaly, UnifiedMVTecDataset
from benchmark_ad.distillation import HeterogeneousDistillationDispatcher
from benchmark_ad.models import VisualADTeacherViTL14, build_student
from utils.transforms import get_transform


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _teacher_to_dict(out) -> Dict:
    return {"cls_token": out.cls_token, "patch_tokens": out.patch_tokens, "attention_maps": out.attention_maps}


def train(cfg: Dict) -> None:
    runtime = cfg.get("runtime", {})
    data_cfg = cfg.get("data", {})
    teacher_cfg = cfg.get("teacher", {})
    student_cfg = cfg.get("student", {})
    distill_cfg = cfg.get("distill", {})

    seed = int(runtime.get("seed", 111))
    setup_seed(seed)
    device = torch.device(runtime.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    image_size = int(runtime.get("image_size", 256))
    batch_size = int(runtime.get("batch_size", 8))
    epochs = int(runtime.get("epochs", 20))
    num_workers = int(runtime.get("num_workers", 4))
    lr = float(runtime.get("learning_rate", 1e-4))
    weight_decay = float(runtime.get("weight_decay", 1e-4))
    log_freq = int(runtime.get("log_freq", 20))
    save_freq = int(runtime.get("save_freq", 5))
    save_dir = Path(runtime.get("save_dir", "./experiments/benchmark_ad"))
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = str(data_cfg.get("root", ""))
    category = str(data_cfg.get("category", "all"))
    synth_prob = float(data_cfg.get("synth_prob", 0.5))
    if not dataset_root or "REPLACE_WITH_YOUR" in dataset_root:
        raise ValueError("Please set data.root to your real dataset path.")

    teacher_layers = [int(v) for v in teacher_cfg.get("features_list", [6, 12, 18, 24])]
    teacher = VisualADTeacherViTL14(features_list=teacher_layers, device=device)

    student_name = str(student_cfg.get("name", "mobilevit_s"))
    student_out_indices = student_cfg.get("out_indices")
    pretrained = bool(student_cfg.get("pretrained", True))
    student = build_student(student_name, out_indices=student_out_indices, pretrained=pretrained).to(device)

    dispatcher = HeterogeneousDistillationDispatcher(
        route=student.spec.route,
        student_stage_channels=student.stage_channels,
        teacher_dim=teacher.embed_dim,
        cfg=distill_cfg,
    ).to(device)

    t_args = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(t_args)
    synth = HybridSyntheticAnomaly(**data_cfg.get("synthetic_anomaly", {}))
    train_set = UnifiedMVTecDataset(
        root=dataset_root,
        transform=preprocess,
        target_transform=target_transform,
        mode="train",
        category=category,
        synth_prob=synth_prob,
        synthetic_anomaly=synth,
    )
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    params = list(student.parameters()) + list(dispatcher.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    logs = []
    for epoch in range(epochs):
        student.train()
        dispatcher.train()
        running = {}
        for step, batch in enumerate(loader, start=1):
            images, masks, _, _, _ = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            masks = (masks > 0.5).float()

            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images)
            loss_dict = dispatcher(_teacher_to_dict(t_out), s_out, masks)
            total = loss_dict["total"]

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())
            if step % log_freq == 0:
                msg = " ".join([f"{k}={running[k]/step:.6f}" for k in sorted(running.keys())])
                print(f"[Epoch {epoch + 1}/{epochs}] step={step}/{len(loader)} {msg}")
        scheduler.step()

        epoch_log = {k: running[k] / max(1, len(loader)) for k in running}
        epoch_log["epoch"] = epoch + 1
        logs.append(epoch_log)

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch + 1,
                "student_name": student_name,
                "student_backbone": student.backbone_name,
                "student_out_indices": student.out_indices,
                "route": student.spec.route,
                "teacher_layers": teacher_layers,
                "student_state_dict": student.state_dict(),
                "dispatcher_state_dict": dispatcher.state_dict(),
                "config": cfg,
                "logs": logs,
            }
            ckpt_path = save_dir / f"{student_name}_epoch_{epoch + 1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"checkpoint saved: {ckpt_path}")

    final_path = save_dir / f"{student_name}_final.pth"
    torch.save(
        {
            "student_name": student_name,
            "student_backbone": student.backbone_name,
            "student_out_indices": student.out_indices,
            "route": student.spec.route,
            "teacher_layers": teacher_layers,
            "student_state_dict": student.state_dict(),
            "dispatcher_state_dict": dispatcher.state_dict(),
            "config": cfg,
            "logs": logs,
        },
        final_path,
    )
    with (save_dir / f"{student_name}_loss_curve.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"training finished: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser("benchmark_ad train")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()


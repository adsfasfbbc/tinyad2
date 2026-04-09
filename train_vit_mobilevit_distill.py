from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from dataset import Dataset
from distill import AttentionMimicryLoss, CLSTokenAlignmentLoss, TokenContrastiveLoss
from student import MobileViTTokenStudent
from utils.anomaly_generator import build_anomaly_generator
from utils.transforms import get_transform
from utils.vit_teacher_adapter import VisualADViTTeacherAdapter


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _align_tokens(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align token counts by interpolating student token map to teacher token map size.
    """
    bs, n_s, d_s = student_tokens.shape
    bt, n_t, d_t = teacher_tokens.shape
    if bs != bt or d_s != d_t:
        raise ValueError(
            f"token batch/dim mismatch: student={tuple(student_tokens.shape)} teacher={tuple(teacher_tokens.shape)}"
        )
    if n_s == n_t:
        return student_tokens, teacher_tokens

    side_s = int(n_s**0.5)
    side_t = int(n_t**0.5)
    if side_s * side_s != n_s or side_t * side_t != n_t:
        n = min(n_s, n_t)
        return student_tokens[:, :n, :], teacher_tokens[:, :n, :]

    s_map = student_tokens.transpose(1, 2).reshape(bs, d_s, side_s, side_s)
    s_map = torch.nn.functional.interpolate(s_map, size=(side_t, side_t), mode="bilinear", align_corners=False)
    s_tokens = s_map.flatten(2).transpose(1, 2)
    return s_tokens, teacher_tokens


def train(cfg: Dict[str, Any]) -> None:
    runtime = cfg.get("runtime", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    seed = int(runtime.get("seed", 111))
    setup_seed(seed)
    device = torch.device(runtime.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")

    image_size = int(runtime.get("image_size", 256))
    batch_size = int(runtime.get("batch_size", 8))
    epochs = int(runtime.get("epochs", 20))
    num_workers = int(runtime.get("num_workers", 4))
    lr = float(runtime.get("learning_rate", 1e-4))
    weight_decay = float(runtime.get("weight_decay", 1e-4))
    save_freq = int(runtime.get("save_freq", 5))
    log_freq = int(runtime.get("log_freq", 20))
    save_path = Path(runtime.get("save_path", "./experiments/vit_mobilevit_distill"))
    save_path.mkdir(parents=True, exist_ok=True)

    teacher_backbone = str(model_cfg.get("teacher_backbone", "ViT-L/14@336px"))
    teacher_layers = [int(v) for v in model_cfg.get("teacher_layers", [6, 12, 18, 24])]
    student_backbone = str(model_cfg.get("student_backbone", "mobilevit_s"))
    student_out_indices = [int(v) for v in model_cfg.get("student_out_indices", [1, 2, 3, 4])]
    projector_hidden_dim = int(model_cfg.get("projector_hidden_dim", 1024))
    projector_dropout = float(model_cfg.get("projector_dropout", 0.1))
    pretrained = bool(model_cfg.get("pretrained", True))

    dataset_name = str(data_cfg.get("dataset_name", "mvtec"))
    train_data_path = str(data_cfg.get("train_data_path", ""))
    online_anomaly_prob = float(data_cfg.get("online_anomaly_prob", 0.7))
    anomaly_gen_cfg = data_cfg.get("anomaly_generator", {})
    if not train_data_path or "REPLACE_WITH_YOUR" in train_data_path:
        raise ValueError("Please set data.train_data_path to real dataset root containing meta.json")

    token_margin = float(loss_cfg.get("token_margin", 0.5))
    attention_temp = float(loss_cfg.get("attention_temperature", 0.07))
    w_token = float(loss_cfg.get("weight_token", 1.0))
    w_attn = float(loss_cfg.get("weight_attention", 1.0))
    w_cls = float(loss_cfg.get("weight_cls", 0.5))

    teacher = VisualADViTTeacherAdapter(
        backbone=teacher_backbone,
        features_list=teacher_layers,
        device=device,
    )
    student = MobileViTTokenStudent(
        backbone_name=student_backbone,
        feature_out_indices=student_out_indices,
        teacher_dim=teacher.embed_dim,
        projector_hidden_dim=projector_hidden_dim,
        projector_dropout=projector_dropout,
        pretrained=pretrained,
    ).to(device)

    anomaly_generator = build_anomaly_generator(anomaly_gen_cfg)
    t_args = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(t_args)
    train_data = Dataset(
        root=train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=dataset_name,
        mode="train",
        anomaly_generator=anomaly_generator,
        online_anomaly_prob=online_anomaly_prob,
    )
    loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    loss_token = TokenContrastiveLoss(margin=token_margin)
    loss_attention = AttentionMimicryLoss(temperature=attention_temp)
    loss_cls = CLSTokenAlignmentLoss()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    print("[Heterogeneous Distill] start")
    print(f"teacher={teacher_backbone}, student={student_backbone}, teacher_layers={teacher_layers}, student_indices={student_out_indices}")
    print(f"dataset={dataset_name}, train_data_path={train_data_path}, online_anomaly_prob={online_anomaly_prob}")

    global_step = 0
    for epoch in range(epochs):
        student.train()
        running_total = 0.0
        running_token = 0.0
        running_attn = 0.0
        running_cls = 0.0

        for step, batch in enumerate(loader, start=1):
            image = batch["img"].to(device, non_blocking=True)
            mask = batch["img_mask"].to(device, non_blocking=True).float()
            mask = (mask > 0.5).float()

            teacher_out = teacher(image)
            student_out = student(image)

            t_stages = teacher_out.stage_tokens
            s_stages = student_out["stage_tokens"]
            stages = min(len(t_stages), len(s_stages))
            if stages == 0:
                raise RuntimeError("no stage tokens available for distillation")

            token_loss_val = torch.tensor(0.0, device=device)
            attn_loss_val = torch.tensor(0.0, device=device)

            for s_tok, t_tok in zip(s_stages[:stages], t_stages[:stages]):
                s_aligned, t_aligned = _align_tokens(s_tok, t_tok)
                token_loss_val = token_loss_val + loss_token(s_aligned, t_aligned, mask=mask)
                attn_loss_val = attn_loss_val + loss_attention(s_aligned, t_aligned)

            token_loss_val = token_loss_val / stages
            attn_loss_val = attn_loss_val / stages
            cls_loss_val = loss_cls(student_out["cls_token"], teacher_out.cls_token)

            total_loss = w_token * token_loss_val + w_attn * attn_loss_val + w_cls * cls_loss_val
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            running_total += float(total_loss.detach().item())
            running_token += float(token_loss_val.detach().item())
            running_attn += float(attn_loss_val.detach().item())
            running_cls += float(cls_loss_val.detach().item())
            global_step += 1

            if step % log_freq == 0:
                denom = float(step)
                print(
                    f"[Epoch {epoch + 1}/{epochs}] step={step}/{len(loader)} "
                    f"total={running_total/denom:.6f} token={running_token/denom:.6f} "
                    f"attn={running_attn/denom:.6f} cls={running_cls/denom:.6f}"
                )

        scheduler.step()

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch + 1,
                "teacher_backbone": teacher_backbone,
                "student_backbone": student_backbone,
                "teacher_layers": teacher_layers,
                "student_out_indices": student_out_indices,
                "teacher_dim": teacher.embed_dim,
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
            }
            ckpt_path = save_path / f"epoch_{epoch + 1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[Heterogeneous Distill] checkpoint saved: {ckpt_path}")

    final_path = save_path / "final_student.pth"
    torch.save(student.state_dict(), final_path)
    with (save_path / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher_backbone": teacher_backbone,
                "teacher_layers": teacher_layers,
                "student_backbone": student_backbone,
                "student_out_indices": student_out_indices,
                "dataset_name": dataset_name,
                "train_data_path": train_data_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "loss_weights": {"token": w_token, "attention": w_attn, "cls": w_cls},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Heterogeneous Distill] training completed. final_student={final_path}")


def main() -> None:
    parser = argparse.ArgumentParser("ViT→MobileViT Heterogeneous Distillation")
    parser.add_argument("--config", type=str, default="configs/vit_mobilevit_distill.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import timm
import torch
import torch.nn.functional as F
import yaml

from dataset import Dataset
from student import TimmStudent, build_anomaly_synthesizer, build_loss_adjuster
from utils.transforms import get_transform


SUPPORTED_STUDENTS = {
    "mobilenetv3_small_100",
    "ghostnet_100",
    "repvgg_a0",
    "repvgg_a1",
    "mobilevit_s",
    "fasternet_t0",
}
DEFAULT_FEATURE_OUT_INDICES = [1, 2, 3, 4]
DEFAULT_TEACHER_CHANNELS = [256, 512, 1024, 2048]


def parse_int_list(raw: str) -> List[int]:
    values = ast.literal_eval(raw)
    if not isinstance(values, (list, tuple)):
        raise ValueError("feature_out_indices/teacher_channels must be a list/tuple")
    return [int(v) for v in values]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataset(train_data_path: str, dataset: str, image_size: int) -> Dataset:
    transform_args = argparse.Namespace(image_size=image_size)
    preprocess, target_transform = get_transform(transform_args)
    try:
        return Dataset(
            root=train_data_path,
            transform=preprocess,
            target_transform=target_transform,
            dataset_name=dataset,
            mode="train",
        )
    except KeyError:
        return Dataset(
            root=train_data_path,
            transform=preprocess,
            target_transform=target_transform,
            dataset_name=dataset,
            mode="test",
        )


def _build_teacher(backbone_name: str, feature_out_indices: List[int], pretrained: bool) -> torch.nn.Module:
    return timm.create_model(
        backbone_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=tuple(feature_out_indices),
    )


def _feature_mse(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
    mask: torch.Tensor | None = None,
    anomaly_region_weight: float = 4.0,
) -> torch.Tensor:
    if len(student_feats) != len(teacher_feats):
        raise ValueError(f"feature map count mismatch: {len(student_feats)} vs {len(teacher_feats)}")
    losses = []
    for s_feat, t_feat in zip(student_feats, teacher_feats):
        if s_feat.shape != t_feat.shape:
            raise ValueError(f"feature shape mismatch: {tuple(s_feat.shape)} vs {tuple(t_feat.shape)}")
        diff = (s_feat - t_feat).pow(2)
        if mask is not None:
            feat_mask = F.interpolate(mask, size=s_feat.shape[-2:], mode="nearest")
            weight = 1.0 + feat_mask * (float(anomaly_region_weight) - 1.0)
            diff = diff * weight
        losses.append(diff.mean())
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser("Train Phase-1 timm student distillation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--teacher_backbone", type=str, default=None)
    parser.add_argument("--student_backbone", type=str, default=None)
    parser.add_argument("--feature_out_indices", type=str, default=None)
    parser.add_argument("--teacher_channels", type=str, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--log_freq", type=int, default=None)
    parser.add_argument("--anomaly_synthesizer", type=str, default=None, choices=["none", "cutpaste", "perlin"])
    parser.add_argument("--loss_adjuster", type=str, default=None, choices=["fixed", "warmup"])
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    setup_seed(int(args.seed if args.seed is not None else cfg.get("seed", 111)))

    teacher_backbone = args.teacher_backbone or cfg.get("teacher_backbone", "wide_resnet50_2")
    student_backbone = args.student_backbone or cfg.get("student_backbone", "mobilenetv3_small_100")
    feature_out_indices = (
        parse_int_list(args.feature_out_indices)
        if args.feature_out_indices is not None
        else list(cfg.get("feature_out_indices", DEFAULT_FEATURE_OUT_INDICES))
    )
    teacher_channels_cfg = (
        parse_int_list(args.teacher_channels)
        if args.teacher_channels is not None
        else list(cfg.get("teacher_channels", DEFAULT_TEACHER_CHANNELS))
    )
    image_size = args.image_size or int(cfg.get("image_size", 256))
    batch_size = args.batch_size or int(cfg.get("batch_size", 8))
    epochs = args.epochs or int(cfg.get("epochs", 20))
    learning_rate = args.learning_rate or float(cfg.get("learning_rate", 1e-4))
    weight_decay = args.weight_decay or float(cfg.get("weight_decay", 1e-4))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 4))
    save_freq = args.save_freq or int(cfg.get("save_freq", 5))
    log_freq = args.log_freq or int(cfg.get("log_freq", 20))
    save_path = args.save_path or cfg.get("save_path", "./experiments/timm_distill")
    dataset = args.dataset or cfg.get("dataset", "mvtec")
    train_data_path = args.train_data_path or cfg.get("train_data_path", "")
    pretrained = cfg.get("pretrained", True) if args.pretrained is None else args.pretrained
    anomaly_synthesizer_name = args.anomaly_synthesizer or cfg.get("anomaly_synthesizer", "cutpaste")
    loss_adjuster_name = args.loss_adjuster or cfg.get("loss_adjuster", "fixed")
    anomaly_synth_cfg = cfg.get("anomaly_synthesizer_config", {})
    loss_adjuster_cfg = cfg.get("loss_adjuster_config", {})
    anomaly_region_weight = float(cfg.get("anomaly_region_weight", 4.0))

    if not train_data_path:
        raise ValueError("train_data_path is empty. Please set a real dataset path.")
    if "REPLACE_WITH_YOUR" in train_data_path:
        raise ValueError("train_data_path still contains placeholder text. Please replace it with a real path.")
    if student_backbone not in SUPPORTED_STUDENTS:
        raise ValueError(f"Unsupported student_backbone: {student_backbone}")
    if student_backbone == "fasternet_t0" and feature_out_indices != [0, 1, 2, 3]:
        raise ValueError("fasternet_t0 only supports feature_out_indices=[0, 1, 2, 3] in this project setup")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    teacher = _build_teacher(teacher_backbone, feature_out_indices, bool(pretrained)).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_channels = list(teacher.feature_info.channels())
    if len(teacher_channels_cfg) != len(teacher_channels):
        print(
            f"[Phase1] teacher_channels cfg length mismatch (cfg={len(teacher_channels_cfg)}, "
            f"model={len(teacher_channels)}), use model channels: {teacher_channels}"
        )
    elif teacher_channels_cfg != teacher_channels:
        print(f"[Phase1] teacher_channels from model: {teacher_channels} (override cfg value: {teacher_channels_cfg})")

    student = TimmStudent(
        backbone_name=student_backbone,
        feature_out_indices=feature_out_indices,
        teacher_channels=teacher_channels,
        pretrained=bool(pretrained),
    ).to(device)

    train_data = _build_dataset(train_data_path=train_data_path, dataset=dataset, image_size=image_size)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    anomaly_synthesizer = build_anomaly_synthesizer(anomaly_synthesizer_name, anomaly_synth_cfg)
    loss_adjuster = build_loss_adjuster(loss_adjuster_name, loss_adjuster_cfg)

    print("[Phase1] Distillation training start")
    print(f"teacher_backbone={teacher_backbone}")
    print(f"student_backbone={student_backbone}")
    print(f"feature_out_indices={feature_out_indices}")
    print(f"teacher_channels={teacher_channels}")
    print(f"dataset={dataset}")
    print(f"train_data_path={train_data_path}")
    print(f"anomaly_synthesizer={anomaly_synthesizer_name}")
    print(f"loss_adjuster={loss_adjuster_name}")
    print(f"save_path={save_path}")

    global_step = 0
    for epoch in range(epochs):
        student.train()
        running_total, running_clean, running_anomaly = 0.0, 0.0, 0.0

        for step, batch in enumerate(train_loader, start=1):
            images = batch["img"].to(device, non_blocking=True)

            student_out = student(images)
            with torch.no_grad():
                teacher_feats = teacher(images)
            clean_loss = _feature_mse(student_out["aligned_features"], teacher_feats, mask=None)

            synth_res = anomaly_synthesizer(images)
            synth_images = synth_res.images
            synth_masks = synth_res.masks
            student_synth = student(synth_images)
            with torch.no_grad():
                teacher_synth = teacher(synth_images)
            anomaly_loss = _feature_mse(
                student_synth["aligned_features"],
                teacher_synth,
                mask=synth_masks,
                anomaly_region_weight=anomaly_region_weight,
            )

            losses = {"clean_distill": clean_loss, "anomaly_distill": anomaly_loss}
            adjusted = loss_adjuster.combine(losses, epoch=epoch, step=global_step)
            total_loss = adjusted.total_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            running_total += float(total_loss.detach().item())
            running_clean += float(clean_loss.detach().item())
            running_anomaly += float(anomaly_loss.detach().item())
            global_step += 1

            if step % log_freq == 0:
                avg_total = running_total / step
                avg_clean = running_clean / step
                avg_anomaly = running_anomaly / step
                print(
                    f"[Epoch {epoch + 1}/{epochs}] step={step}/{len(train_loader)} "
                    f"total={avg_total:.6f} clean={avg_clean:.6f} anomaly={avg_anomaly:.6f} "
                    f"weights={adjusted.weights}"
                )

        scheduler.step()

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch + 1,
                "student_backbone": student_backbone,
                "teacher_backbone": teacher_backbone,
                "feature_out_indices": feature_out_indices,
                "teacher_channels": teacher_channels,
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "anomaly_synthesizer": anomaly_synthesizer_name,
                "loss_adjuster": loss_adjuster_name,
                "config": cfg,
            }
            ckpt_path = Path(save_path) / f"epoch_{epoch + 1}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[Phase1] checkpoint saved: {ckpt_path}")

    final_path = Path(save_path) / "final_student.pth"
    torch.save(student.state_dict(), final_path)
    with (Path(save_path) / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "teacher_backbone": teacher_backbone,
                "student_backbone": student_backbone,
                "feature_out_indices": feature_out_indices,
                "teacher_channels": teacher_channels,
                "dataset": dataset,
                "train_data_path": train_data_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "anomaly_synthesizer": anomaly_synthesizer_name,
                "loss_adjuster": loss_adjuster_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Phase1] training completed. final_student={final_path}")


if __name__ == "__main__":
    main()

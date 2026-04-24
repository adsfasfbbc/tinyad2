from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader

from models import DistillationAdapter, VMambaStudent, VisualADTeacher
from utils.distill_loss import DistillLossConfig, DistillationLoss
from utils.distill_transforms import get_distill_train_transform


class UnlabeledImageDataset(Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, root: str, transform) -> None:
        self.root = root
        self.transform = transform
        self.paths: List[str] = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(self.IMG_EXTS):
                    self.paths.append(os.path.join(dp, fn))
        self.paths.sort()
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = default_loader(path)
        return self.transform(img)


def parse_pairings(value: str):
    # format: "3+4:16,4:24"
    pairings = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        pairings.append((left.strip(), int(right.strip())))
    return pairings


def main() -> None:
    parser = argparse.ArgumentParser("VMamba-VisualAD Dense Distillation")
    parser.add_argument("--train_root", type=str, required=True, help="ImageNet/CC3M image root (labels unused)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_distill")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--teacher_layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--pairings", type=str, default="3+4:16,4:24")
    parser.add_argument("--lambda_dense", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--student_pretrained", type=str, default=None)
    parser.add_argument("--allow_fallback_student", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vmamba-visualad")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tfm = get_distill_train_transform(args.image_size)
    ds = UnlabeledImageDataset(args.train_root, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    teacher = VisualADTeacher(layers=args.teacher_layers, device=device)
    student = VMambaStudent(pretrained=args.student_pretrained, use_fallback_if_unavailable=args.allow_fallback_student).to(device)
    adapter = DistillationAdapter(stage_channels=student.stage_channels, output_dim=1024, target_hw=24).to(device)

    loss_cfg = DistillLossConfig(
        lambda_dense=args.lambda_dense,
        lambda_cls=args.lambda_cls,
        pairings=parse_pairings(args.pairings),
    )
    criterion = DistillationLoss(loss_cfg)

    optim = torch.optim.AdamW(
        list(student.parameters()) + list(adapter.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    step = 0
    for epoch in range(args.epochs):
        nan_skip_count = 0
        student.train()
        adapter.train()

        for images in dl:
            step += 1
            images = images.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                with torch.no_grad():
                    teacher_out = teacher(images)

                student_out = student(images)
                adapter_out = adapter(student_out)
                losses = criterion(teacher_out, adapter_out)

                total_loss = losses["loss_total"]

                if not torch.isfinite(total_loss):
                    nan_skip_count += 1
                    if nan_skip_count == 1 or nan_skip_count % 10 == 0:
                        print(f"[warn] Skipping NaN/Inf loss batch. Total skipped: {nan_skip_count}")
                    continue

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(adapter.parameters()), max_norm=args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(adapter.parameters()), max_norm=args.grad_clip)
                optim.step()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_total": float(losses["loss_total"].detach().item()),
                        "train/loss_dense": float(losses["loss_dense"].detach().item()),
                        "train/loss_cls": float(losses["loss_cls"].detach().item()),
                        "train/step": step,
                        "train/epoch": epoch,
                    }
                )

        print(f"[epoch {epoch + 1}] skipped_nan_inf_batches={nan_skip_count}")

        ckpt = {
            "epoch": epoch + 1,
            "student": student.state_dict(),
            "adapter": adapter.state_dict(),
            "student_stage_channels": student.stage_channels,
            "loss_config": asdict(loss_cfg),
            "teacher_layers": args.teacher_layers,
            "image_size": args.image_size,
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch_{epoch + 1}.pth"))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

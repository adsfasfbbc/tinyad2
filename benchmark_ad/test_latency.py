from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from benchmark_ad.models import build_student


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
def benchmark_fps(model: torch.nn.Module, input_tensor: torch.Tensor, warmup: int = 20, iters: int = 100) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser("benchmark_ad latency")
    parser.add_argument("--student_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    student = build_student(
        model_name=args.student_name,
        out_indices=ckpt.get("student_out_indices"),
        pretrained=False,
    ).to(device)
    student.load_state_dict(ckpt["student_state_dict"], strict=False)
    student.eval()

    x = torch.randn(1, 3, int(args.image_size), int(args.image_size), device=device)
    params_m = _count_params(student)
    flops_g = _estimate_flops_conv_linear(student, x)
    fps = benchmark_fps(student, x, warmup=int(args.warmup), iters=int(args.iters))

    result: Dict[str, float] = {"params_m": params_m, "flops_g": flops_g, "fps": fps}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved latency result: {path}")


if __name__ == "__main__":
    main()


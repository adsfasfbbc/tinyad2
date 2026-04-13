from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_ad.eval import evaluate_latency


def main() -> None:
    parser = argparse.ArgumentParser("benchmark_ad latency (compat wrapper)")
    parser.add_argument("--student_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    result = evaluate_latency(
        checkpoint_path=Path(args.checkpoint),
        student_name=args.student_name,
        image_size=int(args.image_size),
        device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
        warmup=int(args.warmup),
        iters=int(args.iters),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved latency result: {path}")


if __name__ == "__main__":
    main()

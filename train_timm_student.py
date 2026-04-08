from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from student import TimmStudent


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
        raise ValueError("feature_out_indices must be a list/tuple")
    return [int(v) for v in values]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        "Train a lightweight timm student backbone for Phase 1 bootstrap"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--student_backbone", type=str, default=None)
    parser.add_argument("--feature_out_indices", type=str, default=None)
    parser.add_argument("--teacher_channels", type=str, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    student_backbone = args.student_backbone or cfg.get("student_backbone", "mobilenetv3_small_100")
    feature_out_indices = (
        parse_int_list(args.feature_out_indices)
        if args.feature_out_indices is not None
        else list(cfg.get("feature_out_indices", DEFAULT_FEATURE_OUT_INDICES))
    )
    teacher_channels = (
        parse_int_list(args.teacher_channels)
        if args.teacher_channels is not None
        else list(cfg.get("teacher_channels", DEFAULT_TEACHER_CHANNELS))
    )
    image_size = args.image_size or int(cfg.get("image_size", 256))
    save_path = args.save_path or cfg.get("save_path", "./experiments/timm_distill")
    dataset = args.dataset or cfg.get("dataset", "mvtec")
    train_data_path = args.train_data_path or cfg.get("train_data_path", "")
    pretrained = cfg.get("pretrained", True) if args.pretrained is None else args.pretrained

    if student_backbone not in SUPPORTED_STUDENTS:
        raise ValueError(f"Unsupported student_backbone: {student_backbone}")

    # Safety guard for FasterNet-T0 depth.
    if student_backbone == "fasternet_t0" and feature_out_indices != [0, 1, 2, 3]:
        raise ValueError(
            "fasternet_t0 only supports feature_out_indices=[0, 1, 2, 3] in this project setup"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = TimmStudent(
        backbone_name=student_backbone,
        feature_out_indices=feature_out_indices,
        teacher_channels=teacher_channels,
        pretrained=bool(pretrained),
    ).to(device)

    dummy = torch.randn(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        out = model(dummy)

    aligned_shapes = [tuple(feat.shape) for feat in out["aligned_features"]]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print("[Phase1] TimmStudent ready")
    print(f"dataset={dataset}")
    print(f"train_data_path={train_data_path}")
    print(f"student_backbone={student_backbone}")
    print(f"feature_out_indices={feature_out_indices}")
    print(f"student_channels={model.output_channels}")
    print(f"aligned_feature_shapes={aligned_shapes}")


if __name__ == "__main__":
    main()

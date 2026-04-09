from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from student import TimmStudent


class StudentOnnxWrapper(torch.nn.Module):
    def __init__(self, student: TimmStudent) -> None:
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        aligned = self.student(x)["aligned_features"]
        return tuple(aligned)


def _parse_int_list(raw: str) -> List[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser("Export timm student checkpoint to ONNX")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--student_backbone", type=str, required=True)
    parser.add_argument("--feature_out_indices", type=str, default="1,2,3,4")
    parser.add_argument("--teacher_channels", type=str, default="256,512,1024,2048")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    feature_out_indices = _parse_int_list(args.feature_out_indices)
    teacher_channels = _parse_int_list(args.teacher_channels)

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    student = TimmStudent(
        backbone_name=args.student_backbone,
        feature_out_indices=feature_out_indices,
        teacher_channels=teacher_channels,
        pretrained=bool(args.pretrained),
    ).to(device)
    student.eval()

    if "student_state_dict" in checkpoint:
        student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    else:
        student.load_state_dict(checkpoint, strict=False)

    wrapper = StudentOnnxWrapper(student).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[f"aligned_feature_{idx}" for idx in range(len(feature_out_indices))],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .synthetic_anomaly import HybridSyntheticAnomaly


class UnifiedMVTecDataset(data.Dataset):
    """
    Unified MVTec loader:
    - mode=train: supports online synthetic anomaly injection for normal images.
    - always returns (image, mask, label, cls_name, img_path)
    """

    def __init__(
        self,
        root: str,
        transform,
        target_transform,
        mode: str = "train",
        category: str = "all",
        synth_prob: float = 0.5,
        synthetic_anomaly: HybridSyntheticAnomaly | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.category = category
        self.synth_prob = float(max(0.0, min(1.0, synth_prob)))
        self.synthetic_anomaly = synthetic_anomaly if synthetic_anomaly is not None else HybridSyntheticAnomaly()

        with open(os.path.join(root, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)[mode]
        if category == "all":
            cls_names = list(meta.keys())
        else:
            cls_names = [category]
            if category not in meta:
                raise KeyError(f"category={category} not found in meta[{mode}]")

        self.items: List[dict] = []
        for c in cls_names:
            self.items.extend(meta[c])

    def __len__(self) -> int:
        return len(self.items)

    def _load_mask(self, mask_path: str, img_size: Tuple[int, int], anomaly: int) -> Image.Image:
        if anomaly == 0:
            return Image.fromarray(np.zeros((img_size[1], img_size[0]), dtype=np.uint8), mode="L")
        full = os.path.join(self.root, mask_path)
        if os.path.isdir(full):
            return Image.fromarray(np.zeros((img_size[1], img_size[0]), dtype=np.uint8), mode="L")
        m = np.array(Image.open(full).convert("L")) > 0
        return Image.fromarray((m.astype(np.uint8) * 255), mode="L")

    def __getitem__(self, index: int):
        item = self.items[index]
        img_path = os.path.join(self.root, item["img_path"])
        mask_path = item["mask_path"]
        cls_name = item["cls_name"]
        anomaly = int(item["anomaly"])

        img = Image.open(img_path).convert("RGB")
        mask = self._load_mask(mask_path, img.size, anomaly)

        img_t = self.transform(img) if self.transform is not None else img
        mask_t = self.target_transform(mask) if self.target_transform is not None else mask
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.tensor(np.array(mask_t), dtype=torch.float32).unsqueeze(0) / 255.0
        mask_t = (mask_t > 0.5).float()

        if self.mode == "train" and anomaly == 0 and random.random() < self.synth_prob:
            synth_img, synth_mask, synth_label = self.synthetic_anomaly(img_t)
            img_t, mask_t, anomaly = synth_img, synth_mask, int(synth_label)

        return img_t, mask_t, torch.tensor(float(anomaly), dtype=torch.float32), cls_name, img_path


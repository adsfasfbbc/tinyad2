from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from torchvision import transforms


@dataclass
class ResizeWithAspectPad:
    target_size: int
    fill: int = 0

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img.resize((self.target_size, self.target_size), Image.BICUBIC)

        scale = self.target_size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = img.resize((nw, nh), Image.BICUBIC)

        canvas = Image.new("RGB", (self.target_size, self.target_size), color=(self.fill, self.fill, self.fill))
        left = (self.target_size - nw) // 2
        top = (self.target_size - nh) // 2
        canvas.paste(resized, (left, top))
        return canvas


def get_distill_train_transform(image_size: int = 336) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def get_zero_shot_eval_transform(image_size: int = 512) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeWithAspectPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def get_zero_shot_mask_transform(image_size: int = 512) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeWithAspectPad(image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

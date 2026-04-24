from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .contracts import TeacherOutput


class OpenCLIPTeacher(nn.Module):
    """
    Frozen OpenCLIP ViT-L/14@336 teacher extracting multi-layer patch/cls tokens.
    Expected outputs:
      - patch_tokens[layer]: [B, 576, 1024]
      - cls_tokens[layer]: [B, 1024]
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14-336",
        pretrained: str = "openai",
        layers: Iterable[int] = (8, 16, 24),
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for OpenCLIPTeacher. Install open_clip_torch first."
            ) from exc

        self.layers: List[int] = sorted({int(l) for l in layers})
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            force_image_size=336,
        )
        self.visual = self.model.visual
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def _get_transformer_blocks(self) -> nn.ModuleList:
        if hasattr(self.visual, "transformer") and hasattr(self.visual.transformer, "resblocks"):
            return self.visual.transformer.resblocks
        if hasattr(self.visual, "trunk") and hasattr(self.visual.trunk, "blocks"):
            return self.visual.trunk.blocks
        raise RuntimeError("Unsupported OpenCLIP visual transformer structure.")

    def _forward_vit_tokens(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        blocks = self._get_transformer_blocks()

        x = self.visual.conv1(images)  # [B, C, 24, 24] for 336 and patch 14
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, 576, C]

        if hasattr(self.visual, "class_embedding"):
            cls_tok = self.visual.class_embedding.to(x.dtype)
            cls_tok = cls_tok.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tok, x], dim=1)  # [B, 577, C]

        if hasattr(self.visual, "positional_embedding"):
            pos = self.visual.positional_embedding.to(x.dtype)
            if pos.ndim == 2:
                x = x + pos.unsqueeze(0)
            else:
                x = x + pos

        if hasattr(self.visual, "patch_dropout"):
            x = self.visual.patch_dropout(x)
        if hasattr(self.visual, "ln_pre"):
            x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # [N, B, C]
        tokens: Dict[int, torch.Tensor] = {}

        for idx, block in enumerate(blocks, start=1):
            x = block(x)
            if idx in self.layers:
                out = x.permute(1, 0, 2)  # [B, 577, C]
                tokens[idx] = out

        return tokens

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        images = images.to(self.device)
        layer_tokens = self._forward_vit_tokens(images)

        patch_tokens: Dict[int, torch.Tensor] = {}
        cls_tokens: Dict[int, torch.Tensor] = {}

        for layer, token in layer_tokens.items():
            cls_tokens[layer] = token[:, 0, :]
            patch_tokens[layer] = token[:, 1:, :]

        return TeacherOutput(patch_tokens=patch_tokens, cls_tokens=cls_tokens)

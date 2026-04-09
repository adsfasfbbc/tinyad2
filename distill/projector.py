from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjectionHead(nn.Module):
    """Token projector: [B,N,D_in] -> [B,N,D_out]."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        out = self.net(tokens.reshape(-1, d))
        return out.reshape(b, n, -1)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    dim: int
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    force_math_attn: bool = False


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block (ViT-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        force_math_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path = nn.Dropout(dropout)
        self.force_math_attn = bool(force_math_attn)

        self.norm2 = nn.LayerNorm(dim)
        hidden = max(1, int(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, C)
        x_norm = self.norm1(x)
        if (
            self.force_math_attn
            and torch.cuda.is_available()
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
        ):
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
        else:
            ctx = nullcontext()
        with ctx:
            attn_out = self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.drop_path(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=cfg.dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    force_math_attn=cfg.force_math_attn,
                )
                for _ in range(cfg.depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from pointwavelet.layers.pointnet2_modules import (
    PointNetSetAbstractionWavelet,
    SAWaveletConfig,
)


@dataclass
class PointWaveletClsConfig:
    num_classes: int = 40
    input_channels: int = 0  # e.g., normals (3) if present
    # WaveletFormer (per SA layer)
    wf_learnable: bool = True  # PointWavelet-L by default
    wf_beta: float = 0.05
    wf_J: int = 5
    wf_scales: Optional[list[float]] = None
    wf_depth: int = 2
    wf_heads: int = 4
    wf_sigma_mode: str = "mean"
    wf_chunk_size: Optional[int] = None
    wf_force_math_attn: bool = False

    # Universal Measure Correction (UMC)
    wf_use_umc: bool = False
    wf_umc_hidden: tuple[int, int] = (32, 32)
    wf_umc_knn: int = 8
    wf_umc_min_weight: float = 1e-3
    wf_umc_use_inverse: bool = True


class PointWaveletClassifier(nn.Module):
    """PointWavelet / PointWavelet-L classification network.

    Follows the paper's backbone layout:
      SA1 (512, k=32, C=128) + WF1
      SA2 (128, k=32, C=256) + WF2
      SA3 ( 32, k=32, C=512) + WF3
      SA4 (  1, k=32, C=512) + WF4

    Note: SA layers include pointwise MLP; WF is applied on each local patch before pooling.
    """

    def __init__(self, cfg: PointWaveletClsConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.sa1 = PointNetSetAbstractionWavelet(
            in_channels=cfg.input_channels,
            cfg=SAWaveletConfig(
                npoint=512,
                k=32,
                mlp=[64, 64, 128],
                wf_depth=cfg.wf_depth,
                wf_heads=cfg.wf_heads,
                wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales,
                wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta,
                wf_sigma_mode=cfg.wf_sigma_mode,
                wf_chunk_size=cfg.wf_chunk_size,
                wf_force_math_attn=cfg.wf_force_math_attn,
                wf_use_umc=cfg.wf_use_umc,
                wf_umc_hidden=cfg.wf_umc_hidden,
                wf_umc_knn=cfg.wf_umc_knn,
                wf_umc_min_weight=cfg.wf_umc_min_weight,
                wf_umc_use_inverse=cfg.wf_umc_use_inverse,
            ),
        )
        self.sa2 = PointNetSetAbstractionWavelet(
            in_channels=128,
            cfg=SAWaveletConfig(
                npoint=128,
                k=32,
                mlp=[128, 128, 256],
                wf_depth=cfg.wf_depth,
                wf_heads=cfg.wf_heads,
                wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales,
                wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta,
                wf_sigma_mode=cfg.wf_sigma_mode,
                wf_chunk_size=cfg.wf_chunk_size,
                wf_force_math_attn=cfg.wf_force_math_attn,
                wf_use_umc=cfg.wf_use_umc,
                wf_umc_hidden=cfg.wf_umc_hidden,
                wf_umc_knn=cfg.wf_umc_knn,
                wf_umc_min_weight=cfg.wf_umc_min_weight,
                wf_umc_use_inverse=cfg.wf_umc_use_inverse,
            ),
        )
        self.sa3 = PointNetSetAbstractionWavelet(
            in_channels=256,
            cfg=SAWaveletConfig(
                npoint=32,
                k=32,
                mlp=[256, 256, 512],
                wf_depth=cfg.wf_depth,
                wf_heads=cfg.wf_heads,
                wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales,
                wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta,
                wf_sigma_mode=cfg.wf_sigma_mode,
                wf_chunk_size=cfg.wf_chunk_size,
                wf_force_math_attn=cfg.wf_force_math_attn,
                wf_use_umc=cfg.wf_use_umc,
                wf_umc_hidden=cfg.wf_umc_hidden,
                wf_umc_knn=cfg.wf_umc_knn,
                wf_umc_min_weight=cfg.wf_umc_min_weight,
                wf_umc_use_inverse=cfg.wf_umc_use_inverse,
            ),
        )
        self.sa4 = PointNetSetAbstractionWavelet(
            in_channels=512,
            cfg=SAWaveletConfig(
                npoint=1,
                k=32,
                mlp=[512, 512, 512],
                wf_depth=cfg.wf_depth,
                wf_heads=cfg.wf_heads,
                wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales,
                wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta,
                wf_sigma_mode=cfg.wf_sigma_mode,
                wf_chunk_size=cfg.wf_chunk_size,
                wf_force_math_attn=cfg.wf_force_math_attn,
                wf_use_umc=cfg.wf_use_umc,
                wf_umc_hidden=cfg.wf_umc_hidden,
                wf_umc_knn=cfg.wf_umc_knn,
                wf_umc_min_weight=cfg.wf_umc_min_weight,
                wf_umc_use_inverse=cfg.wf_umc_use_inverse,
            ),
        )

        # Classification head (PointNet++-style)
        self.fc1 = nn.Linear(512, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, cfg.num_classes)

    def regularization_loss(self) -> torch.Tensor:
        # Sum L1 penalties from WaveletFormer layers (only nonzero for learnable version)
        return self.sa1.regularization_loss() + self.sa2.regularization_loss() + self.sa3.regularization_loss() + self.sa4.regularization_loss()

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward.

        Args:
            xyz: (B, N, 3)
            feats: (B, N, C_in) or None (e.g., normals)

        Returns:
            logits: (B, num_classes)
        """
        l1_xyz, l1_points = self.sa1(xyz, feats)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B,1,512)

        x = l4_points.squeeze(1)  # (B,512)
        x = self.drop1(torch.relu(self.bn1(self.fc1(x))))
        x = self.drop2(torch.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

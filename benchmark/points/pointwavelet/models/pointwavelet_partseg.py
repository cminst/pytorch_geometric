from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from pointwavelet.layers.pointnet2_modules import (
    PointNetFeaturePropagation,
    PointNetSetAbstractionWavelet,
    SAWaveletConfig,
)


@dataclass
class PointWaveletPartSegConfig:
    num_parts: int = 50
    input_channels: int = 0  # e.g., normals
    category_embed_dim: int = 16  # optional class conditioning (ShapeNetPart)
    use_category_label: bool = True

    wf_learnable: bool = True
    wf_beta: float = 0.05
    wf_J: int = 5
    wf_scales: Optional[list[float]] = None
    wf_depth: int = 2
    wf_heads: int = 4
    wf_sigma_mode: str = "mean"


class PointWaveletPartSeg(nn.Module):
    """PointWavelet / PointWavelet-L part segmentation network.

    Uses PointNet++ style SA + FP with WaveletFormer inside each SA.
    """

    def __init__(self, cfg: PointWaveletPartSegConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Optional category conditioning MLP
        if cfg.use_category_label:
            self.cat_mlp = nn.Sequential(
                nn.Linear(cfg.category_embed_dim, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            extra_in = 64
        else:
            self.cat_mlp = None
            extra_in = 0

        self.sa1 = PointNetSetAbstractionWavelet(
            in_channels=cfg.input_channels + extra_in,
            cfg=SAWaveletConfig(
                npoint=512, k=32, mlp=[64, 64, 128],
                wf_depth=cfg.wf_depth, wf_heads=cfg.wf_heads, wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales, wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta, wf_sigma_mode=cfg.wf_sigma_mode,
            ),
        )
        self.sa2 = PointNetSetAbstractionWavelet(
            in_channels=128,
            cfg=SAWaveletConfig(
                npoint=128, k=32, mlp=[128, 128, 256],
                wf_depth=cfg.wf_depth, wf_heads=cfg.wf_heads, wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales, wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta, wf_sigma_mode=cfg.wf_sigma_mode,
            ),
        )
        self.sa3 = PointNetSetAbstractionWavelet(
            in_channels=256,
            cfg=SAWaveletConfig(
                npoint=32, k=32, mlp=[256, 256, 512],
                wf_depth=cfg.wf_depth, wf_heads=cfg.wf_heads, wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales, wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta, wf_sigma_mode=cfg.wf_sigma_mode,
            ),
        )
        self.sa4 = PointNetSetAbstractionWavelet(
            in_channels=512,
            cfg=SAWaveletConfig(
                npoint=1, k=32, mlp=[512, 512, 512],
                wf_depth=cfg.wf_depth, wf_heads=cfg.wf_heads, wf_J=cfg.wf_J,
                wf_scales=cfg.wf_scales, wf_learnable=cfg.wf_learnable,
                wf_beta=cfg.wf_beta, wf_sigma_mode=cfg.wf_sigma_mode,
            ),
        )

        self.fp4 = PointNetFeaturePropagation(in_channels=512 + 512, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channels=256 + 512, mlp=[512, 512])
        self.fp2 = PointNetFeaturePropagation(in_channels=128 + 512, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channels=256 + extra_in + cfg.input_channels, mlp=[128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, cfg.num_parts, kernel_size=1)

    def regularization_loss(self) -> torch.Tensor:
        return self.sa1.regularization_loss() + self.sa2.regularization_loss() + self.sa3.regularization_loss() + self.sa4.regularization_loss()

    def forward(
        self,
        xyz: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        category_onehot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward.

        Args:
            xyz: (B, N, 3)
            feats: (B, N, C_in) or None
            category_onehot: (B, category_embed_dim) if use_category_label

        Returns:
            logits: (B, N, num_parts)
        """
        B, N, _ = xyz.shape

        if self.cfg.use_category_label:
            if category_onehot is None:
                raise ValueError("category_onehot must be provided when use_category_label=True")
            cat_feat = self.cat_mlp(category_onehot)  # (B,64)
            cat_feat = cat_feat.view(B, 1, -1).repeat(1, N, 1)  # (B,N,64)
            if feats is None:
                feats0 = cat_feat
            else:
                feats0 = torch.cat([feats, cat_feat], dim=-1)
        else:
            feats0 = feats

        l1_xyz, l1_points = self.sa1(xyz, feats0)      # (B,512,128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B,128,256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 32,512)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B,  1,512)

        l3_points_up = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)  # (B,32,512)
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_up)  # (B,128,512)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)  # (B,512,256)

        # For fp1, skip is original features (feats0); source is l1_points_up (B,512,256)
        l0_points_up = self.fp1(xyz, l1_xyz, feats0, l1_points_up)  # (B,N,128)

        x = l0_points_up.permute(0, 2, 1).contiguous()  # (B,128,N)
        x = self.drop1(torch.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)  # (B,num_parts,N)
        return x.permute(0, 2, 1).contiguous()

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pointwavelet.utils.pointnet2_utils import farthest_point_sample, index_points, knn_point, three_interpolate, three_nn
from pointwavelet.layers.waveletformer import WaveletFormer, WaveletFormerConfig


def _make_conv2d_mlp(channels: List[int], bn: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False))
        if bn:
            layers.append(nn.BatchNorm2d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_conv1d_mlp(channels: List[int], bn: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=False))
        if bn:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@dataclass
class SAWaveletConfig:
    npoint: int
    k: int
    mlp: List[int]  # pointwise MLP output channels; last is C_out
    wf_depth: int = 2
    wf_heads: int = 4
    wf_J: int = 5
    wf_scales: Optional[List[float]] = None
    wf_learnable: bool = False
    wf_beta: float = 0.05
    wf_sigma_mode: str = "mean"
    wf_chunk_size: Optional[int] = None
    wf_force_math_attn: bool = False

    # Universal Measure Correction (UMC)
    wf_use_umc: bool = False
    wf_umc_hidden: tuple[int, int] = (32, 32)
    wf_umc_knn: int = 8
    wf_umc_min_weight: float = 1e-3
    wf_umc_use_inverse: bool = True


class PointNetSetAbstractionWavelet(nn.Module):
    """PointNet++ Set Abstraction (sampling + grouping) + WaveletFormer spectral block.

    Pipeline (per centroid):
      - FPS to sample npoint centroids
      - kNN to group k neighbors
      - pointwise shared MLP on grouped (relative_xyz || features)
      - WaveletFormer on the local patch (k nodes)
      - max-pool over k to get centroid feature

    Inputs:
      xyz: (B, N, 3)
      points: (B, N, C_in) or None

    Outputs:
      new_xyz: (B, npoint, 3)
      new_points: (B, npoint, C_out)
    """

    def __init__(self, in_channels: int, cfg: SAWaveletConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if len(cfg.mlp) < 1:
            raise ValueError("mlp must have at least one output channel")
        self.npoint = cfg.npoint
        self.k = cfg.k

        # MLP processes (relative_xyz + features)
        mlp_in = in_channels + 3
        mlp_channels = [mlp_in] + cfg.mlp
        self.mlp = _make_conv2d_mlp(mlp_channels, bn=True)
        self.out_channels = cfg.mlp[-1]

        self.wf = WaveletFormer(
            WaveletFormerConfig(
                n_nodes=cfg.k,
                dim=self.out_channels,
                J=cfg.wf_J,
                scales=cfg.wf_scales,
                sigma_mode=cfg.wf_sigma_mode,
                learnable=cfg.wf_learnable,
                beta=cfg.wf_beta,
                chunk_size=cfg.wf_chunk_size,
                depth=cfg.wf_depth,
                num_heads=cfg.wf_heads,
                force_math_attn=cfg.wf_force_math_attn,
                use_umc=cfg.wf_use_umc,
                umc_hidden=cfg.wf_umc_hidden,
                umc_knn=cfg.wf_umc_knn,
                umc_min_weight=cfg.wf_umc_min_weight,
                umc_use_inverse=cfg.wf_umc_use_inverse,
            )
        )

    def regularization_loss(self) -> torch.Tensor:
        return self.wf.regularization_loss()

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if xyz.dim() != 3 or xyz.size(-1) != 3:
            raise ValueError(f"xyz must be (B,N,3), got {xyz.shape}")
        B, N, _ = xyz.shape

        # Sample centroids
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # Group neighbors
        _, idx = knn_point(self.k, xyz, new_xyz)  # (B, npoint, k)
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, k, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # relative coords

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, npoint, k, C_in)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, k, 3+C_in)
        else:
            new_points = grouped_xyz_norm  # (B, npoint, k, 3)

        # Pointwise shared MLP: (B, npoint, k, C_in+3) -> (B, C_out, npoint, k)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)  # (B, C_out, npoint, k)
        new_points = new_points.permute(0, 2, 3, 1).contiguous()  # (B, npoint, k, C_out)

        # WaveletFormer per patch: flatten patches
        P = B * self.npoint
        patch_feats = new_points.view(P, self.k, self.out_channels)
        patch_xyz = grouped_xyz_norm.view(P, self.k, 3)

        # Always pass patch_xyz; WaveletFormer will ignore it unless needed.
        patch_out = self.wf(patch_feats, xyz=patch_xyz)  # (P,k,C_out)
        patch_out = patch_out.view(B, self.npoint, self.k, self.out_channels)

        # Pool over neighbors
        new_points_out = patch_out.max(dim=2).values  # (B, npoint, C_out)
        return new_xyz, new_points_out


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation layer (upsampling).

    Args:
        in_channels: channels of concatenated (interpolated + skip) features
        mlp: list of output channels for conv1d MLP
    """

    def __init__(self, in_channels: int, mlp: List[int]) -> None:
        super().__init__()
        self.mlp = _make_conv1d_mlp([in_channels] + mlp, bn=True)
        self.out_channels = mlp[-1]

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate features from (xyz2, points2) to xyz1.

        Args:
            xyz1: (B, N, 3) target (denser) points
            xyz2: (B, S, 3) source (sparser) points
            points1: (B, N, C1) skip features (may be None)
            points2: (B, S, C2) source features

        Returns:
            new_points: (B, N, C_out)
        """
        if xyz1.dim() != 3 or xyz2.dim() != 3:
            raise ValueError("xyz1/xyz2 must be 3D")
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)  # (B,N,C2)
        else:
            dists, idx = three_nn(xyz1, xyz2)  # (B,N,3)
            interpolated = three_interpolate(points2, idx, dists)  # (B,N,C2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)  # (B,N,C1+C2)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,C,N)
        new_points = self.mlp(new_points)  # (B,C_out,N)
        return new_points.permute(0, 2, 1).contiguous()

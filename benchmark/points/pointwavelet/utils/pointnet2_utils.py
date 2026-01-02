from __future__ import annotations

"""PointNet++ utilities implemented in pure PyTorch (no custom CUDA ops)."""

from typing import Tuple

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance.

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        dist2: (B, N, M)
    """
    if src.dim() != 3 or dst.dim() != 3:
        raise ValueError(f"src and dst must be 3D tensors, got {src.shape=} {dst.shape=}")
    src2 = (src ** 2).sum(dim=-1, keepdim=True)  # (B,N,1)
    dst2 = (dst ** 2).sum(dim=-1, keepdim=True).transpose(1, 2)  # (B,1,M)
    inner = torch.bmm(src, dst.transpose(1, 2))  # (B,N,M)
    dist2 = src2 + dst2 - 2.0 * inner
    return torch.clamp(dist2, min=0.0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features with batched indices.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        gathered: (B, S, C) or (B, S, K, C)
    """
    if points.dim() != 3:
        raise ValueError(f"points must be (B,N,C), got {points.shape}")
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    batch_idx = torch.arange(B, dtype=torch.long, device=device).view(*view_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling (FPS).

    Args:
        xyz: (B, N, 3)
        npoint: number of samples (<= N)

    Returns:
        centroids_idx: (B, npoint) long
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must be (B,N,3), got {xyz.shape}")
    B, N, _ = xyz.shape
    if npoint > N:
        raise ValueError(f"npoint ({npoint}) must be <= N ({N})")

    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), float('inf'), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid_xyz) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=1)[1]
    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """kNN search (sorted by distance ascending).

    Args:
        k: number of neighbors
        xyz: (B, N, 3) database points
        new_xyz: (B, S, 3) query points

    Returns:
        dists2: (B, S, k) squared distances
        idx: (B, S, k) indices into xyz
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    if k > N:
        raise ValueError(f"k ({k}) must be <= N ({N})")
    dist2 = square_distance(new_xyz, xyz)  # (B,S,N)
    dists2, idx = torch.topk(dist2, k=k, dim=-1, largest=False, sorted=True)
    return dists2, idx


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """3-NN of unknown in known (for FP).

    Args:
        unknown: (B, N, 3)
        known: (B, M, 3)

    Returns:
        dists: (B, N, 3) euclidean distances
        idx: (B, N, 3) indices into known
    """
    dist2 = square_distance(unknown, known)
    dists2, idx = torch.topk(dist2, k=3, dim=-1, largest=False, sorted=True)
    dists = torch.sqrt(torch.clamp(dists2, min=1e-10))
    return dists, idx


def three_interpolate(known_feats: torch.Tensor, idx: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
    """Inverse-distance weighted interpolation.

    Args:
        known_feats: (B, M, C)
        idx: (B, N, 3)
        dists: (B, N, 3) euclidean distances

    Returns:
        unknown_feats: (B, N, C)
    """
    inv_d = 1.0 / torch.clamp(dists, min=1e-10)
    norm = inv_d.sum(dim=-1, keepdim=True)
    w = inv_d / norm  # (B,N,3)

    neigh = index_points(known_feats, idx)  # (B,N,3,C)
    return (neigh * w.unsqueeze(-1)).sum(dim=2)

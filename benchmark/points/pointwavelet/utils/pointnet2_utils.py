from __future__ import annotations

"""PointNet++ utilities implemented in pure PyTorch (no custom CUDA ops)."""

from typing import Tuple

import torch
from torch_geometric.nn import fps as tg_fps, knn as tg_knn
from torch_geometric.typing import WITH_TORCH_CLUSTER


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

    if not WITH_TORCH_CLUSTER:
        raise RuntimeError("farthest_point_sample requires torch-cluster for optimized FPS.")

    device = xyz.device
    x = xyz.reshape(B * N, 3)
    batch = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(N)
    ratio = float(npoint) / float(N)
    idx = tg_fps(x, batch=batch, ratio=ratio, random_start=True)

    out = torch.empty(B, npoint, dtype=torch.long, device=device)
    for b in range(B):
        idx_b = idx[batch[idx] == b]
        if idx_b.numel() < npoint:
            idx_b = tg_fps(xyz[b], ratio=1.0, random_start=True)
        if idx_b.numel() > npoint:
            idx_b = idx_b[:npoint]
        out[b] = idx_b - (b * N)
    return out


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
    if not WITH_TORCH_CLUSTER:
        raise RuntimeError("knn_point requires torch-cluster for optimized kNN.")

    device = xyz.device
    x = xyz.reshape(B * N, 3)
    y = new_xyz.reshape(B * S, 3)
    batch_x = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(N)
    batch_y = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(S)
    row, col = tg_knn(x, y, k, batch_x, batch_y)

    order = torch.argsort(row, stable=True)
    row = row[order]
    col = col[order]

    col_batch = batch_x[col]
    col_local = col - col_batch * N
    idx = col_local.view(B, S, k)

    neigh = index_points(xyz, idx)  # (B,S,k,3)
    dists2 = ((neigh - new_xyz.unsqueeze(2)) ** 2).sum(dim=-1)
    dists2, sort_idx = torch.sort(dists2, dim=-1)
    idx = torch.gather(idx, -1, sort_idx)
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

"""Utilities + models for:
- spectral projection classifier on ModelNet{10,40} and ShapeNet
- fixed / heuristic / learned quadrature weights
- orthogonality/conditioning regularizer (corr-normalized Gram)
- robustness stress test under IrregularResample bias
- capacity-matched control (same weight MLP, but not used as weights).

Design choices:
1) Basis:
   Compute U from L_sym = I - D^{-1/2} A D^{-1/2} via eigh
   Define phi = D^{-1/2} U  so phi^T D phi = I (degree-orthonormal)
2) Projection:
   f_hat = phi^T (w ⊙ x)
3) Ortho loss:
   Use corr-normalized Gram:
      G = phi^T diag(w) phi
      C_ij = G_ij / sqrt(G_ii G_jj)
      L_ortho = mean ||C - I||_F^2  (diagonal becomes 0 automatically)
4) Capacity control:
   Same weight net; but projection uses w=1.
   We project the predicted weights as a *signal*:
      g_hat = phi^T w_pred
   then inject it as an additive bias in spectral domain:
      f_hat' = f_hat + g_hat (broadcast to channels)
   This uses the extra capacity, but not as quadrature weighting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_dense_adj

try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None


def get_BN(data: Data) -> tuple[int, int]:
    """Returns (B, N) where:
    - B: Number of batches/graphs in the data
         For Batch objects (from DataLoader): uses data.num_graphs
         For single Data objects: defaults to B=1
    - N: Number of nodes per batch/graph
         Calculated as total_nodes // B
         If division is not exact, falls back to B=1 and N=total_nodes.

    Args:
        data: A PyG Data or Batch object

    Returns:
        tuple: (B, N) where B is the number of graphs and N is the number of nodes per graph
    """
    total_nodes = data.num_nodes
    if total_nodes is None:
        total_nodes = data.pos.size(0)
    total_nodes = int(total_nodes)

    B = getattr(data, "num_graphs", 1)
    B = int(B) if B is not None else 1
    if B < 1:
        B = 1

    N = total_nodes // B
    # If something weird happens, fall back safely:
    if N * B != total_nodes:
        B = 1
        N = total_nodes

    return B, N

def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable minutes/hours string."""
    total_seconds = int(round(seconds))
    if total_seconds >= 3600:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}:{minutes:02d} hours"
    if total_seconds >= 60:
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes}:{secs:02d} minutes"
    return f"{total_seconds} seconds"


# ----------------------------
# Reproducibility
# ----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # If you want determinism at the cost of speed:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Core ops / metrics
# ----------------------------

def _batched_view(x: torch.Tensor, B: int, N: int) -> torch.Tensor:
    return x.view(B, N, *x.shape[1:])

def normalize_weights_per_graph(w: torch.Tensor, B: int, N: int, eps: float = 1e-12) -> torch.Tensor:
    """Normalize weights so mean(w)=1 per graph."""
    wb = w.view(B, N)
    wb = wb * (float(N) / (wb.sum(dim=1, keepdim=True) + eps))
    return wb.view(B * N)

def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return (a @ b) / denom

@torch.no_grad()
def batch_weight_stats(w: torch.Tensor, B: int, N: int, eps: float = 1e-12) -> Dict[str, float]:
    wb = w.view(B, N)
    w_min = wb.min(dim=1).values
    w_mean = wb.mean(dim=1)
    w_max = wb.max(dim=1).values
    ratio = w_max / (w_min + eps)

    w_sum = wb.sum(dim=1)
    w2_sum = (wb * wb).sum(dim=1)
    ess = (w_sum * w_sum) / (w2_sum + eps)
    ess_frac = ess / float(N)

    return {
        "w_min": float(w_min.mean().item()),
        "w_mean": float(w_mean.mean().item()),
        "w_max": float(w_max.mean().item()),
        "w_max_over_min": float(ratio.mean().item()),
        "ess_frac": float(ess_frac.mean().item()),
    }

def _normalize_gram_corr(gram: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """gram: [B,K,K]
    corr normalize:
      C_ij = G_ij / sqrt(G_ii G_jj).
    """
    d = torch.diagonal(gram, dim1=1, dim2=2).clamp_min(eps)      # [B,K]
    denom = torch.sqrt(d.unsqueeze(2) * d.unsqueeze(1) + eps)     # [B,K,K]
    return gram / denom

def orthogonality_loss_corr(
    phi: torch.Tensor,
    w: torch.Tensor,
    B: int,
    N: int,
    K: int,
    eps: float = 1e-12,
    normalize: bool = True,
) -> torch.Tensor:
    """L = mean ||C - I||_F^2 where C is corr-normalized Phi^T diag(w) Phi.
    With corr norm, diagonal is ~1 automatically, so this is effectively off-diagonal energy.
    """
    phi_b = phi.view(B, N, K)
    w_b = w.view(B, N)

    gram = torch.bmm(phi_b.transpose(1, 2), phi_b * w_b.unsqueeze(-1))  # [B,K,K]
    C = _normalize_gram_corr(gram, eps=eps)

    I = torch.eye(K, device=phi.device).unsqueeze(0).expand(B, K, K)
    err = (C - I).pow(2).sum(dim=(1, 2)).mean()

    if normalize:
        err = err / float(K * (K - 1) + 1e-12)
    return err


def corr_gram_error_single(phi: torch.Tensor, w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Single-graph corr Gram error ||C - I||_F^2 (not normalized).
    phi: [N,K], w: [N].
    """
    K = phi.shape[1]
    G = phi.T @ (phi * w.view(-1, 1))
    d = torch.diagonal(G).clamp_min(eps)
    C = G / torch.sqrt(d.view(K, 1) * d.view(1, K) + eps)
    I = torch.eye(K, device=phi.device)
    return (C - I).pow(2).sum()


# ----------------------------
# Density features
# ----------------------------

def density_features(pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns:
      mean_dist: [num_nodes]
      log_mean_dist: [num_nodes]
      log_deg: [num_nodes].

    Requires torch_scatter.
    """
    if scatter_mean is None:
        raise ImportError("torch_scatter is required for density features.")

    row, col = edge_index[0], edge_index[1]
    dist = (pos[row] - pos[col]).norm(dim=1)                      # [E]
    mean_dist = scatter_mean(dist, row, dim=0, dim_size=num_nodes) # [N]
    log_mean_dist = torch.log(mean_dist + 1e-6)

    deg = degree(row, num_nodes=num_nodes, dtype=pos.dtype)
    log_deg = torch.log(deg + 1.0)

    return mean_dist, log_mean_dist, log_deg


# ----------------------------
# Networks
# ----------------------------

class WeightEstimator(nn.Module):
    """Predict per-point positive weights with per-graph normalization mean(w)=1."""
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (128, 64), eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        h1, h2 = hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.LeakyReLU(0.2),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.2),
            nn.Linear(h2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(5.0))

    def forward(self, feat: torch.Tensor, B: int, N: int) -> torch.Tensor:
        raw = self.mlp(feat).view(B, N)
        w = F.softplus(raw * self.output_scale) + self.eps
        w = w * (float(N) / (w.sum(dim=1, keepdim=True) + self.eps))
        return w.view(B * N)


class SpectralHead(nn.Module):
    """Core spectral projection head:
      project -> filter -> abs -> flatten -> MLP -> log_softmax.

    Provides feature extraction for stability analysis.
    """
    def __init__(self, in_channels: int, num_classes: int, K: int):
        super().__init__()
        self.K = int(K)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        self.spectral_filter = nn.Parameter(torch.ones(1, self.K, self.in_channels))

        self.mlp = nn.Sequential(
            nn.Linear(self.K * self.in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

    def project(self, x: torch.Tensor, phi: torch.Tensor, w: torch.Tensor, B: int) -> torch.Tensor:
        """x:   [B*N, C]
        phi: [B*N, K]
        w:   [B*N]
        returns f_hat: [B, K, C].
        """
        C = x.shape[1]
        N = x.shape[0] // B
        x_b = x.view(B, N, C)
        phi_b = phi.view(B, N, self.K)
        w_b = w.view(B, N)

        weighted_x = x_b * w_b.unsqueeze(-1)
        f_hat = torch.bmm(phi_b.transpose(1, 2), weighted_x)   # [B,K,C]
        return f_hat

    def features_from_fhat(self, f_hat: torch.Tensor) -> torch.Tensor:
        """f_hat: [B,K,C] -> y: [B, K*C]."""
        y = torch.abs(f_hat * self.spectral_filter)
        return y.reshape(y.shape[0], -1)

    def logits_from_features(self, y: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.mlp(y), dim=1)


# ----------------------------
# Model variants
# ----------------------------

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: Data, return_features: bool = False):
        raise NotImplementedError


class NoWeightClassifier(BaseModel):
    def __init__(self, K: int, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.K = int(K)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi

        B, N = get_BN(data)
        total_nodes = B * N

        w = torch.ones(total_nodes, device=x.device, dtype=x.dtype)

        f_hat = self.core.project(x=x, phi=phi, w=w, B=B)
        y = self.core.features_from_fhat(f_hat)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": False, "uses_weights_for_projection": False}
        if return_features:
            return logp, w, aux, y
        return logp, w, aux


class FixedDegreeClassifier(BaseModel):
    def __init__(self, K: int, num_classes: int, in_channels: int = 3, eps: float = 1e-12):
        super().__init__()
        self.K = int(K)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)
        self.eps = float(eps)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi
        deg = data.deg.to(x.device).to(x.dtype)

        B, N = get_BN(data)
        total_nodes = B * N

        w = normalize_weights_per_graph(deg, B=B, N=N, eps=self.eps)

        f_hat = self.core.project(x=x, phi=phi, w=w, B=B)
        y = self.core.features_from_fhat(f_hat)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": False, "uses_weights_for_projection": True}
        if return_features:
            return logp, w, aux, y
        return logp, w, aux


class InvDegreeHeuristicClassifier(BaseModel):
    def __init__(self, K: int, num_classes: int, in_channels: int = 3, eps: float = 1e-6):
        super().__init__()
        self.K = int(K)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)
        self.eps = float(eps)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi
        deg = data.deg.to(x.device).to(x.dtype)

        B, N = get_BN(data)
        total_nodes = B * N

        w = 1.0 / (deg + self.eps)
        w = normalize_weights_per_graph(w, B=B, N=N, eps=self.eps)

        f_hat = self.core.project(x=x, phi=phi, w=w, B=B)
        y = self.core.features_from_fhat(f_hat)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": False, "uses_weights_for_projection": True}
        if return_features:
            return logp, w, aux, y
        return logp, w, aux


class MeanDistHeuristicClassifier(BaseModel):
    """Heuristic: w ∝ mean_kNN_distance (bigger in sparse regions, smaller in dense clusters)."""
    def __init__(self, K: int, num_classes: int, in_channels: int = 3, eps: float = 1e-12):
        super().__init__()
        if scatter_mean is None:
            raise ImportError("torch_scatter required for MeanDistHeuristicClassifier.")
        self.K = int(K)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)
        self.eps = float(eps)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi

        B, N = get_BN(data)
        total_nodes = B * N

        mean_dist, _, _ = density_features(data.pos, data.edge_index, num_nodes=total_nodes)
        w = mean_dist.to(x.dtype)
        w = normalize_weights_per_graph(w, B=B, N=N, eps=self.eps)

        f_hat = self.core.project(x=x, phi=phi, w=w, B=B)
        y = self.core.features_from_fhat(f_hat)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": False, "uses_weights_for_projection": True}
        if return_features:
            return logp, w, aux, y
        return logp, w, aux


class UMCClassifier(BaseModel):
    """Learned weights w_pred used as quadrature weights in projection: f_hat = phi^T (w_pred ⊙ x)."""
    def __init__(
        self,
        K: int,
        num_classes: int,
        in_channels: int = 3,
        use_pos: bool = True,
        use_density: bool = True,
        weight_hidden: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        if use_density and scatter_mean is None:
            raise ImportError("torch_scatter required for use_density=True.")

        self.K = int(K)
        self.use_pos = bool(use_pos)
        self.use_density = bool(use_density)

        in_dim = 0
        if self.use_pos:
            in_dim += 3
        if self.use_density:
            in_dim += 3  # mean_dist, log_mean_dist, log_deg

        self.weight_net = WeightEstimator(in_dim=in_dim, hidden=weight_hidden)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)

    def _weight_features(self, data: Data) -> torch.Tensor:
        parts = []
        if self.use_pos:
            parts.append(data.pos)

        if self.use_density:
            md, log_md, log_deg = density_features(data.pos, data.edge_index, num_nodes=data.num_nodes)
            parts.append(md.unsqueeze(1))
            parts.append(log_md.unsqueeze(1))
            parts.append(log_deg.unsqueeze(1))

        return torch.cat(parts, dim=1)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi

        B, N = get_BN(data)
        total_nodes = B * N

        feat = self._weight_features(data)
        w_pred = self.weight_net(feat, B=B, N=N)

        f_hat = self.core.project(x=x, phi=phi, w=w_pred, B=B)
        y = self.core.features_from_fhat(f_hat)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": True, "uses_weights_for_projection": True}
        if return_features:
            return logp, w_pred, aux, y
        return logp, w_pred, aux


class ExtraCapacityControl(BaseModel):
    """Capacity-matched control:
      - same weight_net produces w_pred
      - projection uses w=1 (NO quadrature weighting)
      - we still let w_pred influence representation by projecting it as a signal:
          g_hat = phi^T (w_pred)
        then:
          f_hat' = f_hat + g_hat (broadcast over channels).

    This keeps the same weight_net parameters but removes "importance weighting" mechanism.
    """
    def __init__(
        self,
        K: int,
        num_classes: int,
        in_channels: int = 3,
        use_pos: bool = True,
        use_density: bool = True,
        weight_hidden: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        if use_density and scatter_mean is None:
            raise ImportError("torch_scatter required for use_density=True.")
        self.K = int(K)
        self.use_pos = bool(use_pos)
        self.use_density = bool(use_density)

        in_dim = 0
        if self.use_pos:
            in_dim += 3
        if self.use_density:
            in_dim += 3

        self.weight_net = WeightEstimator(in_dim=in_dim, hidden=weight_hidden)
        self.core = SpectralHead(in_channels=in_channels, num_classes=num_classes, K=K)

    def _weight_features(self, data: Data) -> torch.Tensor:
        parts = []
        if self.use_pos:
            parts.append(data.pos)
        if self.use_density:
            md, log_md, log_deg = density_features(data.pos, data.edge_index, num_nodes=data.num_nodes)
            parts.append(md.unsqueeze(1))
            parts.append(log_md.unsqueeze(1))
            parts.append(log_deg.unsqueeze(1))
        return torch.cat(parts, dim=1)

    def forward(self, data: Data, return_features: bool = False):
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi

        B, N = get_BN(data)
        total_nodes = B * N

        feat = self._weight_features(data)
        w_pred = self.weight_net(feat, B=B, N=N)

        w_used = torch.ones_like(w_pred)

        # uniform projection of x
        f_hat = self.core.project(x=x, phi=phi, w=w_used, B=B)

        # project w_pred as a scalar signal, then broadcast to channels
        g_hat = self.core.project(
            x=w_pred.view(B * N, 1),  # [B*N,1]
            phi=phi,
            w=w_used,
            B=B,
        )  # [B,K,1]

        f_hat2 = f_hat + g_hat.expand(-1, -1, f_hat.shape[2])

        y = self.core.features_from_fhat(f_hat2)
        logp = self.core.logits_from_features(y)

        aux = {"B": B, "N": N, "has_weight_net": True, "uses_weights_for_projection": False}
        if return_features:
            return logp, w_pred, aux, y
        return logp, w_pred, aux

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
from typing import Any, Dict, List, Optional, Tuple

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
    """Returns (B, N) for both:
    - Batch objects (from DataLoader): has num_graphs
    - Single Data objects: no num_graphs -> treat as B=1.
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
# Transforms
# ----------------------------

class MakeUndirected(BaseTransform):
    """Ensure edge_index is undirected by adding reverse edges (duplicates allowed)."""
    def forward(self, data: Data) -> Data:
        ei = data.edge_index
        data.edge_index = torch.cat([ei, ei.flip(0)], dim=1)
        return data


class CopyCategoryToY(BaseTransform):
    """For ShapeNet: copy the 'category' field to 'y' for consistent label access.

    ShapeNet stores shape category in data.category (shape [1]) and per-point
    segmentation labels in data.y. For classification tasks, we want data.y
    to be the category label.

    This transform is idempotent - safe to apply multiple times.
    """
    def forward(self, data: Data) -> Data:
        if hasattr(data, 'category') and data.category is not None:
            data.y = data.category.clone()
        return data


class PointMLPAffine(BaseTransform):
    def __init__(self,
                 scale_low: float = 2.0/3.0,
                 scale_high: float = 3.0/2.0,
                 translate_low: float = -0.2,
                 translate_high: float = 0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_low = translate_low
        self.translate_high = translate_high

    def forward(self, data):
        # data.pos: [N,3]
        pos = data.pos

        # Per-axis scale factors (3-dim)
        scales = (self.scale_low
                  + (self.scale_high - self.scale_low)
                  * torch.rand(3, device=pos.device))
        # Per-axis translation
        shifts = (self.translate_low
                  + (self.translate_high - self.translate_low)
                  * torch.rand(3, device=pos.device))

        pos = pos * scales + shifts
        data.pos = pos
        return data


class IrregularResample(BaseTransform):
    """Resample points with optional exponential bias along a random focus direction.

    - bias_strength = 0 => uniform downsample/upsample to num_points
    - bias_strength > 0 => sample without replacement using softmax weights exp(bias * proj)

    IMPORTANT: wipes edge_index/phi/deg so downstream transforms recompute them.
    """
    def __init__(self, num_points: int = 512, bias_strength: float = 0.0):
        self.num_points = int(num_points)
        self.bias = float(bias_strength)

    def forward(self, data: Data) -> Data:
        pos = data.pos
        device = pos.device
        N_curr = pos.size(0)

        if self.bias > 1e-6:
            focus = torch.randn(1, 3, device=device)
            focus = focus / (focus.norm() + 1e-12)
            proj = (pos @ focus.T).squeeze()     # [N]
            proj = proj - proj.max()             # stability
            weights = torch.exp(self.bias * proj)
            weights = weights / (weights.sum() + 1e-12)
            idx = torch.multinomial(weights, self.num_points, replacement=False)
            data.pos = pos[idx]
        else:
            if N_curr >= self.num_points:
                idx = torch.randperm(N_curr, device=device)[:self.num_points]
                data.pos = pos[idx]
            else:
                idx = torch.randint(0, N_curr, (self.num_points,), device=device)
                data.pos = pos[idx]

        # wipe derived fields
        for key in ["edge_index", "edge_attr", "phi", "phi_evals", "deg", "batch"]:
            if hasattr(data, key):
                delattr(data, key)

        data.num_nodes = self.num_points
        return data


class RandomIrregularResample(BaseTransform):
    """Sample a random bias in [0, max_bias] for each example."""
    def __init__(self, num_points: int = 512, max_bias: float = 3.0):
        self.num_points = int(num_points)
        self.max_bias = float(max_bias)

    def forward(self, data: Data) -> Data:
        bias = float(torch.rand(1).item() * self.max_bias)
        return IrregularResample(num_points=self.num_points, bias_strength=bias)(data)


class ComputePhiRWFromSym(BaseTransform):
    """Compute phi = D^{-1/2} U where U are eigenvectors of L_sym.

    A is built from edge_index -> dense adjacency (binarized), diag=0.
    deg is computed from A (consistent with phi construction) and stored in data.deg.
    """
    def __init__(self, K: int = 64, eps: float = 1e-12, store_aux: bool = True):
        self.K = int(K)
        self.eps = float(eps)
        self.store_aux = bool(store_aux)

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        N = int(data.num_nodes)
        device = data.pos.device

        A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).to(device)
        A = (A > 0).to(dtype=torch.float32)
        A.fill_diagonal_(0.0)

        deg = A.sum(dim=1).clamp_min(self.eps)  # [N]
        inv_sqrt_deg = deg.rsqrt()

        L_sym = torch.eye(N, device=device) - (inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :])

        evals, U = torch.linalg.eigh(L_sym)  # ascending
        K = min(self.K, N)
        U = U[:, :K]                         # [N,K]
        phi = inv_sqrt_deg[:, None] * U      # [N,K]

        data.phi = phi.to(torch.float32)
        if self.store_aux:
            data.phi_evals = evals[:K].to(torch.float32)
            data.deg = deg.to(torch.float32)

        return data


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
    phi: [N,K], w: [N]
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


# ----------------------------
# Training / evaluation helpers
# ----------------------------

@torch.no_grad()
def eval_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        logp, _, _ = model(data)
        pred = logp.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += int(data.y.numel())
    return correct / max(total, 1)

@torch.no_grad()
def eval_accuracy_with_metrics(
    model: nn.Module,
    loader,
    device: torch.device,
    K: int,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    ortho_accum = 0.0
    batches = 0
    wstats_accum = None
    correct_per_class = None
    counts_per_class = None
    if num_classes is not None and num_classes > 0:
        correct_per_class = torch.zeros(num_classes, device=device)
        counts_per_class = torch.zeros(num_classes, device=device)

    for data in loader:
        data = data.to(device)
        logp, w, aux = model(data)
        pred = logp.argmax(dim=1)
        correct += int((pred == data.y).sum().item())
        total += int(data.y.numel())
        if counts_per_class is not None:
            y = data.y
            counts_per_class += torch.bincount(y, minlength=num_classes).to(counts_per_class.dtype)
            correct_per_class += torch.bincount(y[pred.eq(y)], minlength=num_classes).to(correct_per_class.dtype)

        B, N = aux["B"], aux["N"]
        o = float(orthogonality_loss_corr(data.phi, w, B=B, N=N, K=K).item())
        ortho_accum += o

        stats = batch_weight_stats(w, B=B, N=N)
        if wstats_accum is None:
            wstats_accum = stats
        else:
            for k in wstats_accum:
                wstats_accum[k] += stats[k]

        batches += 1

    acc = correct / max(total, 1)
    ortho = ortho_accum / max(batches, 1)
    for k in wstats_accum:
        wstats_accum[k] /= max(batches, 1)

    metrics = {"acc": acc, "ortho_corr": ortho, **wstats_accum}
    if counts_per_class is not None:
        valid = counts_per_class > 0
        if valid.any():
            macc = (correct_per_class[valid] / counts_per_class[valid]).mean().item()
        else:
            macc = 0.0
        metrics["macc"] = float(macc)
    return metrics

@torch.no_grad()
def eval_weight_correlations(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    """Average Pearson correlations between predicted/fixed w and density proxies.
    Works for all models; for fixed-weight baselines it will show near-constant values.
    """
    if scatter_mean is None:
        raise ImportError("torch_scatter required for eval_weight_correlations.")

    model.eval()
    acc = {"corr_w_deg": 0.0, "corr_w_invdeg": 0.0, "corr_w_meandist": 0.0, "corr_w_invmeandist": 0.0}
    seen = 0

    for b, data in enumerate(loader):
        if b >= max_batches:
            break
        data = data.to(device)
        _, w, aux = model(data)
        B, N = aux["B"], aux["N"]

        # use stored deg if available
        if hasattr(data, "deg"):
            deg = data.deg.to(w.dtype)
        else:
            row = data.edge_index[0]
            deg = degree(row, num_nodes=data.num_nodes, dtype=w.dtype)

        mean_dist, _, _ = density_features(data.pos, data.edge_index, num_nodes=data.num_nodes)
        mean_dist = mean_dist.to(w.dtype)

        # compute correlation per graph, then average
        wb = w.view(B, N).detach()
        db = deg.view(B, N).detach()
        mb = mean_dist.view(B, N).detach()

        for i in range(B):
            wi = wb[i]
            di = db[i]
            mi = mb[i]
            acc["corr_w_deg"] += float(pearson_corr(wi, di).item())
            acc["corr_w_invdeg"] += float(pearson_corr(wi, 1.0 / (di + 1e-6)).item())
            acc["corr_w_meandist"] += float(pearson_corr(wi, mi).item())
            acc["corr_w_invmeandist"] += float(pearson_corr(wi, 1.0 / (mi + 1e-6)).item())
            seen += 1

    for k in acc:
        acc[k] /= max(seen, 1)
    return acc

@torch.no_grad()
def eval_feature_stability(
    model: nn.Module,
    data_list_dense: List[Data],
    transform_bias0,
    transform_bias1,
    device: torch.device,
    max_items: int = 200,
) -> float:
    """Cosine similarity between feature vectors y for the same shape under:
      - bias0 resample
      - bias1 resample
    We wrap each Data into a Batch of size 1 so num_graphs exists.
    """
    model.eval()
    sims = []
    n = min(max_items, len(data_list_dense))

    for i in range(n):
        base = data_list_dense[i]

        d0 = transform_bias0(base.clone())
        d1 = transform_bias1(base.clone())

        # Wrap to Batch(1)
        d0 = Batch.from_data_list([d0]).to(device)
        d1 = Batch.from_data_list([d1]).to(device)

        _, _, _, y0 = model(d0, return_features=True)
        _, _, _, y1 = model(d1, return_features=True)

        v0 = y0[0]
        v1 = y1[0]
        sim = float(F.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0), dim=1).item())
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_ortho: float = 0.0
    lambda_w_reg: float = 0.0
    clip_grad: float = 0.0
    ortho_normalize: bool = True


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    K: int,
    cfg: TrainConfig,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Train with validation selection (best val acc). Returns final test metrics + best checkpoint stats."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            logp, w, aux = model(data)

            cls_loss = F.nll_loss(logp, data.y)

            B, N = aux["B"], aux["N"]
            ortho = orthogonality_loss_corr(data.phi, w, B=B, N=N, K=K, normalize=cfg.ortho_normalize)

            w_reg = (w - 1.0).pow(2).mean()

            loss = cls_loss
            if cfg.lambda_ortho > 0:
                loss = loss + cfg.lambda_ortho * ortho
            if cfg.lambda_w_reg > 0 and aux.get("has_weight_net", False):
                loss = loss + cfg.lambda_w_reg * w_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad and cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()

        # validate
        val_acc = eval_accuracy(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_metrics = eval_accuracy_with_metrics(model, test_loader, device, K=K, num_classes=num_classes)
    test_macc = float(test_metrics.get("macc", 0.0))

    return {
        "best_val_acc": float(best_val),
        "test_acc": float(test_metrics["acc"]),
        "test_macc": float(test_macc),
        "test_ortho_corr": float(test_metrics["ortho_corr"]),
        "test_w_mean": float(test_metrics["w_mean"]),
        "test_w_min": float(test_metrics["w_min"]),
        "test_w_max": float(test_metrics["w_max"]),
        "test_ess_frac": float(test_metrics["ess_frac"]),
    }

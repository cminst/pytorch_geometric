# umc_modelnet10_utils.py
"""
UMC-style spectral projection utilities for ModelNet10 with learned geometry-conditioned weights.

Key idea:
- Precompute a spectral basis phi per point cloud (Data.phi).
- Learn point weights w = psi_theta(geometry) jointly with classification.
- Regularize weights using an orthogonality/conditioning loss.

IMPORTANT UPDATE (Fix 1b):
Instead of penalizing ||Phi^T diag(w) Phi - I|| directly (which is dominated by diagonal scale),
we use a CORRELATION-NORMALIZED Gram matrix:

    G = Phi^T diag(w) Phi
    C_ij = G_ij / sqrt(G_ii G_jj)

Then minimize ||C - I||_F^2 (optionally normalized). This focuses on cross-mode coupling ("leakage")
and is invariant to per-mode scaling.

Notes:
- For tractability, use num_points=256 or 512 and K<=64-ish.
- Precomputing eigenvectors is expensive; we do it once via dataset pre_transform.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, degree

try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None


# ----------------------------
# Reproducibility helpers
# ----------------------------

def seed_everything(seed: int = 0) -> None:
    import random
    import os
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Graph / spectral precompute
# ----------------------------

class MakeUndirected(BaseTransform):
    """Ensure graph is undirected by adding reverse edges."""
    def forward(self, data: Data) -> Data:
        ei = data.edge_index
        rev = ei.flip(0)
        data.edge_index = torch.cat([ei, rev], dim=1)
        return data

class RandomIrregularResample(BaseTransform):
    """
    Applies IrregularResample with a random bias strength 
    uniformly sampled from [0, max_bias].
    """
    def __init__(self, num_points=512, max_bias=3.0):
        self.num_points = num_points
        self.max_bias = max_bias

    def forward(self, data):
        # Pick a random bias for this specific example
        bias = float(torch.rand(1).item() * self.max_bias)
        
        # Reuse your existing logic
        resampler = IrregularResample(num_points=self.num_points, bias_strength=bias)
        return resampler(data)

class IrregularResample(BaseTransform):
    """
    Simulates sensor bias/occlusion.
    1. Picks a random focus direction.
    2. Concentrates points along that direction using exponential weighting.
    3. Replaces the point cloud with these new samples.
    4. WIPES out existing edges/phi so they can be recomputed.
    """
    def __init__(self, num_points=512, bias_strength=0.0):
        self.num_points = num_points
        self.bias = bias_strength

    def forward(self, data):
        pos = data.pos
        device = pos.device
        N_curr = pos.size(0)
        
        if self.bias > 0.001:
            # 1. Random focus direction
            focus = torch.randn(1, 3, device=device)
            focus = focus / (focus.norm() + 1e-12)
            
            # 2. Project points: higher val = closer to focus direction
            proj = (pos @ focus.T).squeeze() # [N]
            
            # 3. Compute sampling weights: exp(bias * proj)
            # Shift proj so max is 0 for numerical stability
            proj = proj - proj.max()
            weights = torch.exp(self.bias * proj)
            weights = weights / weights.sum()
            
            # 4. Sample without replacement to get exactly num_points
            idx = torch.multinomial(weights, self.num_points, replacement=False)
            new_pos = pos[idx]
            
            data.pos = new_pos
        else:
            # If bias is 0, just downsample uniformly if needed (standard behavior)
            if N_curr >= self.num_points:
                idx = torch.randperm(N_curr)[:self.num_points]
                data.pos = pos[idx]
            else:
                # Upsample with replacement
                idx = torch.randint(0, N_curr, (self.num_points,))
                data.pos = pos[idx]

        # CRITICAL: The old graph (edge_index) and basis (phi) are now WRONG.
        # We must delete them so the subsequent transforms (KNN, ComputePhi) rebuild them.
        for key in ['edge_index', 'edge_attr', 'phi', 'phi_evals', 'deg', 'batch']:
            if hasattr(data, key):
                delattr(data, key)
                
        data.num_nodes = self.num_points
        return data


class ComputePhiRWFromSym(BaseTransform):
    """
    Compute a random-walk-like basis phi from a symmetric normalized Laplacian basis.

    For undirected adjacency A with degrees D:
      L_sym = I - D^{-1/2} A D^{-1/2} (symmetric)
    Compute eigenvectors U of L_sym (orthonormal in Euclidean inner product).

    Then define:
      phi = D^{-1/2} U

    This yields a basis that is generally NOT Euclidean-orthonormal:
      phi^T phi != I when degrees vary
    but it is "degree-orthonormal":
      phi^T D phi = I

    That mismatch is exactly what a diagonal W can attempt to correct/approximate.
    """

    def __init__(self, K: int = 64, eps: float = 1e-12, store_aux: bool = True):
        self.K = int(K)
        self.eps = float(eps)
        self.store_aux = bool(store_aux)

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        N = int(data.num_nodes)
        device = data.pos.device

        # Dense adjacency A (unweighted)
        A = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(0).to(device)
        # Binarize in case edge_index contains duplicates (common after symmetrization)
        A = (A > 0).to(dtype=torch.float32)
        A.fill_diagonal_(0.0)

        deg = A.sum(dim=1).clamp_min(self.eps)  # [N]
        inv_sqrt_deg = deg.rsqrt()

        L_sym = torch.eye(N, device=device) - (inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :])

        evals, U = torch.linalg.eigh(L_sym)  # ascending
        K = min(self.K, N)
        uK = U[:, :K]
        phi = inv_sqrt_deg[:, None] * uK

        data.phi = phi.to(torch.float32)

        if self.store_aux:
            data.phi_evals = evals[:K].to(torch.float32)
            data.deg = deg.to(torch.float32)

        return data


# ----------------------------
# Model components
# ----------------------------

class SpectralProjectionNet(nn.Module):
    """Spectral projection network (as provided, with minor safety)."""
    def __init__(self, in_channels: int, num_classes: int, K: int):
        super().__init__()
        self.K = int(K)
        self.spectral_filter = nn.Parameter(torch.ones(1, self.K, in_channels))

        self.mlp = nn.Sequential(
            nn.Linear(self.K * in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, phi: torch.Tensor, w: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        x:   [B*N, C]
        phi: [B*N, K]
        w:   [B*N]
        """
        B = int(batch_size)
        C = x.shape[1]
        N = x.shape[0] // B
        if N * B != x.shape[0]:
            raise ValueError("x not divisible by batch_size; ensure fixed num_points.")

        x = x.view(B, N, C)
        phi = phi.view(B, N, self.K)
        w = w.view(B, N)

        weighted_x = x * w.unsqueeze(-1)  # [B, N, C]
        f_hat = torch.bmm(phi.transpose(1, 2), weighted_x)  # [B, K, C]

        y = f_hat * self.spectral_filter
        y = torch.abs(y)  # sign invariance
        y = y.reshape(B, -1)
        return F.log_softmax(self.mlp(y), dim=1)

class WeightEstimator(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, int] = (128, 64),
        eps: float = 1e-6,
    ):
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
        # Learnable gain to help break out of the "1.0" equilibrium
        self.output_scale = nn.Parameter(torch.tensor(5.0)) 

    def forward(self, feat: torch.Tensor, batch_size: int, num_points: int) -> torch.Tensor:
        B = int(batch_size)
        N = int(num_points)
        
        raw = self.mlp(feat).view(B, N)
        
        # Apply gain and Softplus
        # This allows the network to easily output 0.01 or 10.0
        w = F.softplus(raw * self.output_scale) + self.eps
        
        # Normalize
        w = w * (N / (w.sum(dim=1, keepdim=True) + self.eps)) 
        return w.reshape(B * N)
        

class UMCClassifier(nn.Module):
    """
    End-to-end model:
      - predicts w from geometry features
      - performs weighted spectral projection + classification
    """
    def __init__(
        self,
        K: int,
        num_classes: int,
        in_channels: int = 3,
        use_pos: bool = True,
        use_density: bool = True,
        weight_hidden: Tuple[int, int] = (64, 32),
    ):
        super().__init__()
        self.K = int(K)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.use_pos = bool(use_pos)
        self.use_density = bool(use_density)

        in_dim = 0
        if self.use_pos:
            in_dim += 3
        if self.use_density:
            # CHANGED: We now use 3 density features: mean_dist, log_dist, log_deg
            in_dim += 3  
        
        if in_dim == 0:
            raise ValueError("WeightEstimator in_dim would be 0; enable use_pos and/or use_density.")

        if self.use_density and scatter_mean is None:
            raise ImportError("torch_scatter required for density features; install or set use_density=False.")

        self.weight_net = WeightEstimator(in_dim=in_dim, hidden=weight_hidden)
        self.classifier = SpectralProjectionNet(in_channels=self.in_channels, num_classes=self.num_classes, K=self.K)

    def _density_features(self, pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        row, col = edge_index[0], edge_index[1]
        dist = (pos[row] - pos[col]).norm(dim=1)
        mean_dist = scatter_mean(dist, row, dim=0, dim_size=num_nodes)
        
        # CHANGED: Add log distance to handle massive scale differences in clusters
        log_dist = torch.log(mean_dist + 1e-6)
        
        deg = degree(row, num_nodes=num_nodes, dtype=pos.dtype)
        log_deg = torch.log(deg + 1.0)
        
        # Return 3 values
        return mean_dist, log_dist, log_deg

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi

        B = int(data.num_graphs)
        total_nodes = int(data.num_nodes)
        N = total_nodes // B
        if N * B != total_nodes:
            raise ValueError("Variable N in batch; ensure fixed SamplePoints(num_points).")

        feat_parts = []
        if self.use_pos:
            feat_parts.append(data.pos)

        mean_dist = log_deg = None
        if self.use_density:
            # CHANGED: Unpack 3 values here
            mean_dist, log_dist, log_deg = self._density_features(data.pos, data.edge_index, num_nodes=total_nodes)
            
            feat_parts.append(mean_dist.unsqueeze(1))
            feat_parts.append(log_dist.unsqueeze(1)) # Add the new feature
            feat_parts.append(log_deg.unsqueeze(1))

        feat = torch.cat(feat_parts, dim=1)

        w = self.weight_net(feat, batch_size=B, num_points=N)
        log_probs = self.classifier(x=x, phi=phi, w=w, batch_size=B)

        aux = {"B": B, "N": N, "mean_dist": mean_dist, "log_deg": log_deg}
        return log_probs, w, aux


class NoWeightClassifier(nn.Module):
    """
    Baseline: identical spectral classifier, but weights are disabled (w ≡ 1).
    """
    def __init__(self, K: int, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.K = int(K)
        self.classifier = SpectralProjectionNet(in_channels=in_channels, num_classes=num_classes, K=K)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi
        B = int(data.num_graphs)
        total_nodes = int(data.num_nodes)
        N = total_nodes // B
        if N * B != total_nodes:
            raise ValueError("Variable N in batch; ensure fixed SamplePoints(num_points).")
        w = torch.ones(total_nodes, device=x.device, dtype=x.dtype)
        log_probs = self.classifier(x=x, phi=phi, w=w, batch_size=B)
        aux = {"B": B, "N": N, "mean_dist": None, "log_deg": None}
        return log_probs, w, aux


class FixedDegreeClassifier(nn.Module):
    """
    Baseline: identical spectral classifier, but weights are initialized to the degree.
    """
    def __init__(self, K: int, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.K = int(K)
        self.classifier = SpectralProjectionNet(in_channels=in_channels, num_classes=num_classes, K=K)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = data.pos if getattr(data, "x", None) is None else data.x
        phi = data.phi
        B = int(data.num_graphs)
        total_nodes = int(data.num_nodes)
        N = total_nodes // B
        if N * B != total_nodes:
            raise ValueError("Variable N in batch; ensure fixed SamplePoints(num_points).")
        w = data.deg.to(x.device).to(x.dtype)
        log_probs = self.classifier(x=x, phi=phi, w=w, batch_size=B)
        aux = {"B": B, "N": N, "mean_dist": None, "log_deg": None}
        return log_probs, w, aux


# ----------------------------
# Losses / metrics
# ----------------------------

def _normalize_gram(gram: torch.Tensor, mode: str, eps: float = 1e-12) -> torch.Tensor:
    """
    gram: [B, K, K]
    mode:
      - "corr": C_ij = G_ij / sqrt(G_ii G_jj)   (Fix 1b, recommended)
      - "trace": G / (tr(G)/K)                  (Fix 1a)
      - "none": no normalization
    """
    mode = mode.lower()
    if mode == "none":
        return gram

    B, K, _ = gram.shape

    if mode == "trace":
        tr = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)  # [B]
        scale = (tr / float(K)).clamp_min(eps)               # [B]
        return gram / scale.view(B, 1, 1)

    if mode == "corr":
        d = torch.diagonal(gram, dim1=1, dim2=2).clamp_min(eps)  # [B, K]
        denom = torch.sqrt(d.unsqueeze(2) * d.unsqueeze(1) + eps) # [B, K, K]
        return gram / denom

    raise ValueError(f"Unknown gram normalization mode: {mode}")


def orthogonality_loss(
    phi: torch.Tensor,
    w: torch.Tensor,
    B: int,
    N: int,
    K: int,
    mode: str = "corr",
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Mean over batch of ||G_norm - I||_F^2, where
      G = Phi^T diag(w) Phi
      G_norm = normalize(G) based on mode

    mode="corr" is recommended: focuses on cross-mode coupling/leakage.
    """
    phi_b = phi.view(B, N, K)
    w_b = w.view(B, N)

    gram = torch.bmm(phi_b.transpose(1, 2), phi_b * w_b.unsqueeze(-1))  # [B, K, K]
    gram_n = _normalize_gram(gram, mode=mode, eps=eps)

    I = torch.eye(K, device=phi.device).unsqueeze(0).expand(B, K, K)
    err = (gram_n - I).pow(2).sum(dim=(1, 2))  # [B]
    loss = err.mean()

    if normalize:
        # average per-entry error (you can also use K*(K-1) to focus strictly off-diagonal)
        loss = loss / float(K * K)
    return loss


@torch.no_grad()
def batch_weight_stats(w: torch.Tensor, B: int, N: int, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    w_b = w.view(B, N)
    w_min = w_b.min(dim=1).values
    w_mean = w_b.mean(dim=1)
    w_max = w_b.max(dim=1).values
    ratio = w_max / (w_min + eps)

    w_sum = w_b.sum(dim=1)
    w2_sum = (w_b * w_b).sum(dim=1)
    ess = (w_sum * w_sum) / (w2_sum + eps)
    ess_frac = ess / float(N)

    return {
        "w_min": w_min.mean(),
        "w_mean": w_mean.mean(),
        "w_max": w_max.mean(),
        "w_max_over_min": ratio.mean(),
        "ess_frac": ess_frac.mean(),
    }


@torch.no_grad()
def gram_error(phi: torch.Tensor, w: torch.Tensor, K: int) -> torch.Tensor:
    """
    Raw ||Phi^T diag(w) Phi - I||_F^2 for a single graph (phi: [N,K], w: [N]).
    """
    N = phi.shape[0]
    I = torch.eye(K, device=phi.device)
    G = phi.T @ (phi * w.view(N, 1))
    return (G - I).pow(2).sum()


@torch.no_grad()
def corr_gram_error(phi: torch.Tensor, w: torch.Tensor, K: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Correlation-normalized ||C - I||_F^2 for a single graph.
      C_ij = G_ij / sqrt(G_ii G_jj)
    """
    N = phi.shape[0]
    G = phi.T @ (phi * w.view(N, 1))  # [K,K]
    d = torch.diagonal(G).clamp_min(eps)
    denom = torch.sqrt(d.view(K, 1) * d.view(1, K) + eps)
    C = G / denom
    I = torch.eye(K, device=phi.device)
    return (C - I).pow(2).sum()


@torch.no_grad()
def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return (a @ b) / denom

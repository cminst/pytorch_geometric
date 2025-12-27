from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pointwavelet.utils.pointnet2_utils import square_distance

from pointwavelet.layers.graph_wavelet import (
    LearnableSpectralBasis,
    WaveletSpec,
    _default_scales,
    build_normalized_laplacian,
    mexican_hat_scaling,
    mexican_hat_wavelet,
)
from pointwavelet.layers.transformer import TransformerConfig, TransformerEncoder


def _einsum_ut_x(U: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute U^T x with support for batched or unbatched U.

    U: (..., n, n)
    x: (P, n, C)
    returns: (P, n, C)
    """
    if U.dim() == 2:
        return torch.einsum("ij,pjc->pic", U.transpose(0, 1), x)
    elif U.dim() == 3:
        return torch.einsum("pij,pjc->pic", U.transpose(-1, -2), x)
    else:
        raise ValueError(f"U must be 2D or 3D, got {U.shape}")


def _einsum_u_x(U: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute U x with support for batched or unbatched U."""
    if U.dim() == 2:
        return torch.einsum("ij,pjc->pic", U, x)
    elif U.dim() == 3:
        return torch.einsum("pij,pjc->pic", U, x)
    else:
        raise ValueError(f"U must be 2D or 3D, got {U.shape}")


@dataclass
class WaveletFormerConfig:
    n_nodes: int
    dim: int
    wavelet: str = "mexican_hat"
    J: int = 5
    scales: Optional[List[float]] = None
    sigma_mode: str = "mean"  # for adjacency in eigendecomp version
    learnable: bool = False
    beta: float = 0.05  # L1 weight for q_eps in learnable basis
    eps: float = 1e-8

    # Transformer
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Universal Measure Correction (UMC) style quadrature weights
    # If enabled, WaveletFormer will compute coefficients as Psi_s (W x)
    # and reconstruct back with an additional W^{-1} factor.
    use_umc: bool = False
    umc_hidden: Tuple[int, int] = (32, 32)
    umc_knn: int = 8  # neighbors for mean-distance density proxy inside each patch
    umc_min_weight: float = 1e-3
    umc_use_inverse: bool = True


class UMCWeightNet(nn.Module):
    """Predict per-node nonnegative quadrature weights from local geometry."""

    def __init__(self, in_dim: int = 6, hidden: Tuple[int, int] = (32, 32)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
            nn.Softplus(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (P,n,in_dim) -> (P,n)
        return self.net(feats).squeeze(-1)


class WaveletFormer(nn.Module):
    """WaveletFormer layer (PointWavelet / PointWavelet-L core).

    Input is a batch of local patches, each patch being a graph with n_nodes nodes.
    The layer:
      1) computes graph wavelet transform at multiple scales,
      2) treats scales as tokens and runs self-attention across scales per node,
      3) reconstructs back to the vertex domain using the pseudoinverse described in the paper.

    Shapes:
      x: (P, n, C)
      xyz: (P, n, 3) is required for eigendecomposition version; ignored for learnable version.
      output: (P, n, C)
    """

    def __init__(self, cfg: WaveletFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.last_umc: Optional[dict[str, torch.Tensor]] = None
        self.last_umc_raw: Optional[dict[str, torch.Tensor]] = None
        if cfg.scales is None:
            self.scales = _default_scales(cfg.J)
        else:
            if len(cfg.scales) != cfg.J:
                raise ValueError("len(scales) must equal J")
            self.scales = [float(s) for s in cfg.scales]

        if cfg.wavelet != "mexican_hat":
            raise NotImplementedError(
                "This implementation currently supports Mexican hat wavelets as used in the paper. "
                "(Meyer wavelets can be added if needed.)"
            )

        self.transformer = TransformerEncoder(
            TransformerConfig(
                dim=cfg.dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attn_dropout=cfg.attn_dropout,
            )
        )

        self.learnable_basis: Optional[LearnableSpectralBasis] = None
        if cfg.learnable:
            self.learnable_basis = LearnableSpectralBasis(n=cfg.n_nodes, beta=cfg.beta)

        self.umc: Optional[UMCWeightNet] = None
        if cfg.use_umc:
            self.umc = UMCWeightNet(in_dim=6, hidden=cfg.umc_hidden)

    def regularization_loss(self) -> torch.Tensor:
        if self.learnable_basis is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.learnable_basis.regularization_loss()

    def _get_basis(self, xyz: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (U, lambdas).

        - learnable: (n,n), (n,)
        - eigendecomp: (P,n,n), (P,n)
        """
        if self.learnable_basis is not None:
            U, lambdas = self.learnable_basis()
            return U, lambdas
        if xyz is None:
            raise ValueError("xyz is required when learnable=False")
        L = build_normalized_laplacian(xyz, eps=self.cfg.eps, sigma_mode=self.cfg.sigma_mode)  # (P,n,n)
        lambdas, U = torch.linalg.eigh(L)  # lambdas: (P,n), U: (P,n,n)
        return U, lambdas

    def forward(self, x: torch.Tensor, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be (P,n,C), got {x.shape}")
        P, n, C = x.shape
        if n != self.cfg.n_nodes:
            raise ValueError(f"Expected n_nodes={self.cfg.n_nodes}, got n={n}")
        if C != self.cfg.dim:
            raise ValueError(f"Expected dim={self.cfg.dim}, got C={C}")
        # xyz is required for the eigendecomposition basis, and also for UMC weight prediction.
        if (self.learnable_basis is None) and (xyz is None):
            raise ValueError("xyz must be provided for eigendecomposition version")
        if self.cfg.use_umc and xyz is None:
            raise ValueError("xyz must be provided when use_umc=True")

        U, lambdas = self._get_basis(xyz)  # U: (P,n,n) or (n,n)

        # Optional UMC weighting in the vertex domain.
        w: Optional[torch.Tensor] = None
        if self.umc is not None:
            # xyz: (P,n,3)
            assert xyz is not None
            # Pairwise squared distances in the patch.
            dist2 = square_distance(xyz, xyz)  # (P,n,n)
            big = torch.eye(n, device=dist2.device, dtype=dist2.dtype).view(1, n, n) * 1e6
            dist2 = dist2 + big  # mask diagonal

            k = int(min(max(self.cfg.umc_knn, 1), n - 1))
            nn2 = dist2.topk(k, dim=-1, largest=False).values  # (P,n,k)
            md = torch.sqrt(torch.clamp(nn2, min=0.0)).mean(dim=-1)  # (P,n)

            r = torch.sqrt(torch.clamp((xyz ** 2).sum(dim=-1), min=self.cfg.eps))  # (P,n)
            feats = torch.cat(
                [
                    xyz,
                    r.unsqueeze(-1),
                    md.unsqueeze(-1),
                    torch.log(md.unsqueeze(-1) + self.cfg.eps),
                ],
                dim=-1,
            )  # (P,n,6)

            w = self.umc(feats)  # (P,n)
            # Normalize per patch so sum(w)=n.
            w = w * (float(n) / torch.clamp(w.sum(dim=-1, keepdim=True), min=self.cfg.eps))
            w = torch.clamp(w, min=self.cfg.umc_min_weight)
            self.last_umc_raw = {
                "w": w,
                "mean_dist": md,
            }
            self.last_umc = {
                "w": w.detach(),
                "mean_dist": md.detach(),
            }
            x = x * w.unsqueeze(-1)
        else:
            self.last_umc = None
            self.last_umc_raw = None

        # Compute spectral transform of (possibly reweighted) input once.
        x_freq = _einsum_ut_x(U, x)  # (P,n,C)

        # Multi-scale wavelet coefficients (still in vertex domain after filtering)
        h = mexican_hat_scaling(lambdas)  # (P,n) or (n,)
        coeffs = []
        coeffs.append(_einsum_u_x(U, x_freq * h.unsqueeze(-1)))
        g_list = []
        for s in self.scales:
            g = mexican_hat_wavelet(lambdas * float(s))
            g_list.append(g)
            coeffs.append(_einsum_u_x(U, x_freq * g.unsqueeze(-1)))

        # Stack as (P, S, n, C)
        x_stack = torch.stack(coeffs, dim=1)
        S = x_stack.shape[1]

        # Transformer across scales per node:
        # (P,S,n,C) -> (P,n,S,C) -> (P*n,S,C)
        tokens = x_stack.permute(0, 2, 1, 3).reshape(P * n, S, C)
        tokens = self.transformer(tokens)
        x_out_stack = tokens.reshape(P, n, S, C).permute(0, 2, 1, 3)  # (P,S,n,C)

        # Reconstruction via pseudoinverse:
        # In frequency domain: f = U ( p^{-1} * sum_s (w_s * U^T y_s) )
        # where p = h^2 + sum_j g_j^2.
        p = h * h
        for g in g_list:
            p = p + g * g
        p_inv = 1.0 / torch.clamp(p, min=self.cfg.eps)

        # y_s are the transformer outputs per scale.
        y0 = x_out_stack[:, 0]  # (P,n,C)
        y0_freq = _einsum_ut_x(U, y0)
        agg_freq = y0_freq * h.unsqueeze(-1)

        for j, g in enumerate(g_list, start=1):
            yj = x_out_stack[:, j]
            yj_freq = _einsum_ut_x(U, yj)
            agg_freq = agg_freq + yj_freq * g.unsqueeze(-1)

        out_freq = agg_freq * p_inv.unsqueeze(-1)
        out = _einsum_u_x(U, out_freq)

        # If we used coefficients Psi_s (W x), reconstruct with an additional W^{-1}.
        if (w is not None) and self.cfg.umc_use_inverse:
            out = out / torch.clamp(w.unsqueeze(-1), min=self.cfg.umc_min_weight)
        return out

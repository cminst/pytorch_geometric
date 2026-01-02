from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from pointwavelet.utils.pointnet2_utils import square_distance


def mexican_hat_scaling(lambdas: torch.Tensor) -> torch.Tensor:
    # h(x) = e^{-x^4}
    return torch.exp(-(lambdas ** 4))


def mexican_hat_wavelet(x: torch.Tensor) -> torch.Tensor:
    # g(x) = x e^{-x}
    return x * torch.exp(-x)


@dataclass
class WaveletSpec:
    name: str = "mexican_hat"
    J: int = 5
    scales: Optional[List[float]] = None  # if None, defaults to [1,2,4,...]


def _default_scales(J: int) -> List[float]:
    # A common choice for graph wavelets is dyadic scales.
    # The paper does not explicitly list the scales; dyadic scales are used by default.
    return [float(2 ** j) for j in range(J)]  # 1,2,4,...,2^{J-1}


def build_normalized_laplacian(
    xyz: torch.Tensor,
    eps: float = 1e-8,
    sigma_mode: str = "mean",
) -> torch.Tensor:
    """Construct a dense normalized Laplacian for each local patch.

    Uses the normalized Laplacian definition in the paper:
        L = I - D^{-1/2} A D^{-1/2}

    Adjacency weights A_{ij} are constructed with an RBF kernel from pairwise distances.

    Args:
        xyz: (P, n, 3) patch coordinates (translation doesn't matter; relative coords are fine)
        eps: numerical stability
        sigma_mode: how to set the RBF bandwidth per patch.
            - "mean": sigma^2 = mean(dist^2) over i,j
            - "median": sigma^2 = median(dist^2) over i,j (robust)

    Returns:
        L: (P, n, n)
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"xyz must be (P,n,3), got {xyz.shape}")
    P, n, _ = xyz.shape

    dist2 = square_distance(xyz, xyz)  # (P,n,n)

    if sigma_mode == "mean":
        sigma2 = dist2.mean(dim=(-1, -2), keepdim=True)
    elif sigma_mode == "median":
        sigma2 = dist2.flatten(1).median(dim=-1).values.view(P, 1, 1)
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    sigma2 = torch.clamp(sigma2, min=eps)

    A = torch.exp(-dist2 / sigma2)  # (P,n,n)
    # Remove self-loops in adjacency for Laplacian construction (common choice)
    A = A * (1.0 - torch.eye(n, device=xyz.device, dtype=xyz.dtype).view(1, n, n))

    deg = A.sum(dim=-1)  # (P,n)
    deg_inv_sqrt = torch.rsqrt(torch.clamp(deg, min=eps))  # (P,n)
    # normalized adjacency: D^{-1/2} A D^{-1/2}
    An = A * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    I = torch.eye(n, device=xyz.device, dtype=xyz.dtype).view(1, n, n)
    L = I - An
    # Enforce symmetry (helps numerical stability)
    L = 0.5 * (L + L.transpose(-1, -2))
    return L


class LearnableSpectralBasis(nn.Module):
    """Learnable spectral basis (PointWavelet-L).

    Implements Eq. (13)-(14) to construct an orthogonal matrix U from a vector q,
    and learns nonnegative eigenvalues lambda_i via tanh-parameterization.

    The paper uses:
        q = q_ini + q_eps, with q_ini=(c,...,c) and L1 penalty on q_eps.
        lambda_1 = 0, lambda_i = tanh(lambda_theta_i) + 1  (i>=2) so lambda_i > 0.
    """

    def __init__(self, n: int, beta: float = 0.05, init_c: float = 1.0, init_eps_std: float = 1e-2) -> None:
        super().__init__()
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.beta = beta

        self.c_theta = nn.Parameter(torch.tensor(float(init_c)))
        self.q_eps = nn.Parameter(torch.randn(n) * float(init_eps_std))

        self.lambda_theta = nn.Parameter(torch.randn(n - 1) * 0.1)  # for lambda_2..lambda_n

    def q(self) -> torch.Tensor:
        q_ini = self.c_theta * torch.ones(self.n, device=self.c_theta.device, dtype=self.c_theta.dtype)
        q = q_ini + self.q_eps
        # Normalize to ||q||_2 = 1
        q = q / torch.clamp(q.norm(p=2), min=1e-8)
        return q

    def U(self) -> torch.Tensor:
        # Eq. (13)-(14)
        q = self.q()  # (n,)
        n = self.n
        denom = torch.sum(torch.abs(q[1:]) ** 2)
        denom = torch.clamp(denom, min=1e-8)

        # F matrix for i,j>=2
        qi = q[1:].unsqueeze(1)  # (n-1,1)
        qj = q[1:].unsqueeze(0)  # (1,n-1)
        F = (qi * qj) / denom * (q[0] - 1.0)  # (n-1,n-1)

        U = torch.zeros((n, n), device=q.device, dtype=q.dtype)
        U[:, 0] = q
        U[0, 1:] = -q[1:]
        U[1:, 1:] = F + torch.eye(n - 1, device=q.device, dtype=q.dtype)
        return U

    def lambdas(self) -> torch.Tensor:
        # lambda_1 = 0; lambda_i = tanh(theta) + 1 for i>=2
        lam_rest = torch.tanh(self.lambda_theta) + 1.0  # (n-1,)
        lam = torch.cat([torch.zeros(1, device=lam_rest.device, dtype=lam_rest.dtype), lam_rest], dim=0)
        return lam

    def regularization_loss(self) -> torch.Tensor:
        # beta * ||q_eps||_1
        return self.beta * torch.sum(torch.abs(self.q_eps))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.U(), self.lambdas()

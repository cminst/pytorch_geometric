import torch
from torch_geometric.data import Batch, Data
from typing import Any, Dict, List, Optional, Tuple, Union
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_dense_adj


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
    def __init__(
        self,
        K: int = 64,
        eps: float = 1e-12,
        store_aux: bool = True,
        eig_device: Optional[Union[torch.device, str]] = None,
    ):
        self.K = int(K)
        self.eps = float(eps)
        self.store_aux = bool(store_aux)
        self.eig_device = torch.device(eig_device) if eig_device is not None else None
        self._warned_eig_fallback = False

    def _resolve_eig_device(self, default_device: torch.device) -> torch.device:
        if self.eig_device is None:
            return default_device

        if self.eig_device.type == "cuda" and not torch.cuda.is_available():
            if not self._warned_eig_fallback:
                print("ComputePhiRWFromSym: requested CUDA eig_device but CUDA is not available; falling back to CPU.")
                self._warned_eig_fallback = True
            return torch.device("cpu")
        return self.eig_device

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        N = int(data.num_nodes)
        out_device = data.pos.device
        eig_device = self._resolve_eig_device(out_device)

        edge_index = data.edge_index.to(eig_device)
        A = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0)
        A = (A > 0).to(dtype=torch.float32)
        A.fill_diagonal_(0.0)

        deg = A.sum(dim=1).clamp_min(self.eps)  # [N]
        inv_sqrt_deg = deg.rsqrt()

        L_sym = torch.eye(N, device=eig_device) - (inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :])

        # Debugging
        # print(L_sym.shape)

        evals, U = torch.linalg.eigh(L_sym)  # ascending
        K = min(self.K, N)
        U = U[:, :K]                         # [N,K]
        phi = inv_sqrt_deg[:, None] * U      # [N,K]

        data.phi = phi.to(out_device, dtype=torch.float32)
        if self.store_aux:
            data.phi_evals = evals[:K].to(out_device, dtype=torch.float32)
            data.deg = deg.to(out_device, dtype=torch.float32)

        return data

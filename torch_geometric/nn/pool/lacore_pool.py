from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter_mean

from torch_geometric.utils import to_undirected
from torch_geometric.typing import OptTensor


class LaCorePooling(torch.nn.Module):
    r"""Pools nodes according to a precomputed clustering assignment
    (``cluster``) and the number of clusters per graph (``num_clusters``).

    This layer expects static cluster ids, e.g., computed via the LaCore
    cover routine. It aggregates node features into cluster features and
    coarsens the edge_index accordingly.
    """
    def __init__(self, aggregate: str = 'mean'):
        super().__init__()
        if aggregate != 'mean':
            raise ValueError("Only 'mean' aggregation is supported.")
        self.aggregate = aggregate

    @torch.no_grad()
    def _coarsen_edges(
        self,
        edge_index: Tensor,
        batch: Tensor,
        cluster: Tensor,
        num_clusters: Tensor,
    ) -> Tuple[Tensor, OptTensor]:
        # Build global cluster ids per node across the mini-batch so that
        # coarsened edges are unique in the flattened space.
        offsets = torch.zeros_like(num_clusters)
        if num_clusters.numel() > 1:
            offsets[1:] = torch.cumsum(num_clusters[:-1], dim=0)
        global_cluster = cluster + offsets[batch]

        row, col = edge_index
        cu = global_cluster[row]
        cv = global_cluster[col]

        mask = cu != cv
        cu = cu[mask]
        cv = cv[mask]

        pooled = torch.stack([cu, cv], dim=0)
        pooled = to_undirected(pooled)
        pooled = torch.unique(pooled, dim=1)
        return pooled, None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        cluster: Tensor,
        num_clusters: Tensor,
        edge_attr: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, Tensor, Tensor, OptTensor]:
        r"""Forward pass.

        Args:
            x (Tensor): Node feature matrix of shape ``[N, F]``.
            edge_index (Tensor): Edge indices of shape ``[2, E]``.
            batch (Tensor): Batch vector assigning each node to a graph.
            cluster (Tensor): Local cluster assignment per node.
            num_clusters (Tensor): Number of clusters per graph in the batch.
            edge_attr (Tensor, optional): Edge attributes. Currently ignored.
        """
        if cluster.dim() != 1 or cluster.numel() != x.size(0):
            raise ValueError("cluster must be a 1D tensor with one entry per node.")
        if num_clusters.dim() != 1:
            raise ValueError("num_clusters must be a 1D tensor with one entry per graph.")

        total_clusters = int(num_clusters.sum().item())
        if total_clusters == 0:
            raise ValueError("num_clusters sums to zero; nothing to pool.")

        offsets = torch.zeros_like(num_clusters)
        if num_clusters.numel() > 1:
            offsets[1:] = torch.cumsum(num_clusters[:-1], dim=0)
        global_cluster = cluster + offsets[batch]

        if self.aggregate == 'mean':
            x_pool = scatter_mean(x, global_cluster, dim=0, dim_size=total_clusters)
        else:
            raise RuntimeError("Unsupported aggregation.")

        edge_index_pooled, edge_attr_pooled = self._coarsen_edges(
            edge_index, batch, cluster, num_clusters)
        batch_pooled = torch.arange(num_clusters.numel(), device=batch.device).repeat_interleave(num_clusters)

        # Return signature mirrors other pooling ops:
        # (x, edge_index, edge_attr, batch, perm/assignment, score)
        return (x_pool, edge_index_pooled, edge_attr_pooled,
                batch_pooled, cluster, None)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(aggregate={self.aggregate})'

    def reset_parameters(self):
        # No learnable parameters, but defined for API consistency.
        pass

"""
Graph Classification Utilities (graph_classif_utils.py)

This module provides utilities for graph classification tasks using PyTorch Geometric.
It includes methods for generating LaCore-based graph covers, handling datasets,
and various helper functions for graph manipulation and seeding.

Key Components:
- LaCoreWrapped: A wrapper for creating in-memory datasets from PyG data.
- compute_lacore_cover_for_graph: Computes a LaCore-based "cover" of a graph by iteratively finding dense subgraphs.
- Helper functions: For edge list processing, subgraph extraction, and deterministic seeding.
"""

from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from generate_lacore_seeds import generate_lacore_cluster
import random
import numpy as np
import torch

class LaCoreWrapped(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)


def edge_list_from_pyg(data: Data) -> List[Tuple[int, int]]:
    """Return an undirected, deduplicated 0-indexed edge list with no self-loops."""
    assert data.edge_index is not None
    ei = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    u, v = ei[0].tolist(), ei[1].tolist()
    E = set()
    for a, b in zip(u, v):
        if a == b:
            continue
        x, y = (a, b) if a < b else (b, a)
        E.add((x, y))
    return list(E)


def induced_subgraph_edges(all_edges: List[Tuple[int, int]], nodes: List[int]) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    """
    Restrict `all_edges` to the induced subgraph on `nodes`.
    Returns:
      - edge list in the subgraph with 0..(k-1) node ids
      - map old_id -> sub_id
    """
    nodes_sorted = sorted(nodes)
    idmap = {old: i for i, old in enumerate(nodes_sorted)}
    S = set(nodes_sorted)
    sub_edges = []
    for u, v in all_edges:
        if u in S and v in S:
            sub_edges.append((idmap[u], idmap[v]))
    return sub_edges, idmap


def assign_singletons(unassigned: List[int], existing_clusters: List[List[int]]) -> List[List[int]]:
    for u in unassigned:
        existing_clusters.append([u])
    return existing_clusters


def compute_lacore_cover_for_graph(
    data: Data,
    epsilon: float,
    target_ratio: float = 0.5,
    min_size: int = 4,
    max_clusters: Optional[int] = None,
) -> Tuple[torch.Tensor, int, List[List[int]]]:
    """
    Compute a LaCore "cover" for a single PyG graph. It runs LaCore repeatedly until the target ratio is reached. (or the maximum number of clusters is reached)

    Args:
        data: A PyG Data object representing the graph.
        epsilon: The expansion parameter for LaCore detection; lower means more stringent.
        target_ratio: Fraction of nodes to cover (default: 0.5).
        min_size: Minimum number of nodes required for a cluster to be included (default: 4).
        max_clusters: Maximum number of clusters to generate (default: None, no limit).

    Returns:
        A tuple of:
          - cluster_id: A tensor of length N assigning each node to a cluster ID (or -1 for none).
          - num_clusters: Total number of clusters generated.
          - clusters: A list of lists, each containing the node IDs in each cluster.
    """
    N = data.num_nodes
    assert N is not None

    all_edges = edge_list_from_pyg(data)
    remaining = set(range(N))
    clusters: List[List[int]] = []

    covered_target = int(target_ratio * N)
    while len(clusters) < (max_clusters or 10**9) and len(remaining) > 0:
        if N - len(remaining) >= covered_target:
            break

        sub_edges, idmap = induced_subgraph_edges(all_edges, sorted(remaining))
        if len(sub_edges) == 0:
            break

        res = generate_lacore_cluster(sub_edges, epsilon=epsilon)
        sub_seed_nodes = res.get("seed_nodes", [])

        cluster_nodes = [node for node, sid in idmap.items() if sid in set(sub_seed_nodes)]

        if len(cluster_nodes) < min_size:
            break

        clusters.append(cluster_nodes)
        remaining -= set(cluster_nodes)

    if len(remaining) > 0:
        clusters = assign_singletons(sorted(list(remaining)), clusters)

    cluster_id = torch.empty(N, dtype=torch.long)
    for cid, nodes in enumerate(clusters):
        cluster_id[torch.tensor(nodes, dtype=torch.long)] = cid
    num_clusters = len(clusters)
    return cluster_id, num_clusters, clusters


class Accumulator:
    def __init__(self):
        self.value = -float("inf")

    def update_best(self, value):
        if value > self.value:
            self.value = value


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seed_list(seed_value) -> List[int]:
    """
    Normalize a seed argument (int/str/list) into a non-empty list of ints.

    Args:
        seed_value: An integer, string (comma-separated), or list/tuple of seeds.

    Returns:
        A list of integers representing the parsed seeds.

    Raises:
        ValueError: If no seeds are provided or parsing fails.
    """
    if seed_value is None:
        return []
    if isinstance(seed_value, (list, tuple)):
        seeds = list(seed_value)
    elif isinstance(seed_value, str):
        parts = [p.strip() for p in seed_value.split(",") if p.strip()]
        seeds = [int(p) for p in parts] if parts else []
    else:
        seeds = [int(seed_value)]

    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds

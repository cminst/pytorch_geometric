import argparse
from rich import print
import glob
import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple, Iterable, Set
import random
from iv2_utils.iv2 import json_read, json_write
import tempfile

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_edgelist(path: str) -> Iterable[Tuple[int, int]]:
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            yield a, b

def write_edgelist(path: str, edges: Iterable[Tuple[int, int]]):
    with open(path, 'w') as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

def parse_seeds(path: str) -> Tuple[Dict[int, int], List[int]]:
    """
    Return (node_to_cluster_index, sorted_cluster_ids).
    The cluster indices are 0..C-1, sorted by cluster_id.
    - On overlapping membership, choose the cluster with higher 'score', then smaller cluster_id.
    """
    js = json_read(path)
    clusters = js.get('clusters', [])
    clusters_sorted = sorted(clusters, key=lambda c: c.get('cluster_id', 0))
    cluster_id_list = [c.get('cluster_id', i) for i, c in enumerate(clusters_sorted)]
    cluster_id_to_idx = {cid: i for i, cid in enumerate(cluster_id_list)}

    node_choice: Dict[int, Tuple[int, float, int]] = {}

    for c in clusters_sorted:
        cid = c.get('cluster_id', None)
        if cid is None:
            continue
        idx = cluster_id_to_idx[cid]
        members = c.get('members')
        if members is None:
            members = c.get('seed_nodes', [])
        score = float(c.get('score', 0.0))
        for u in members:
            prev = node_choice.get(u, None)
            if prev is None or (score > prev[1]) or (score == prev[1] and cid < prev[2]):
                node_choice[u] = (idx, score, cid)

    node_to_cluster = {u: idx for u, (idx, score, cid) in node_choice.items()}
    return node_to_cluster, cluster_id_list

def coarsen_edgelist(prev_edgelist: str, seeds_json: str, out_edgelist: str) -> int:
    node_to_cluster, cluster_id_list = parse_seeds(seeds_json)
    edges_set: Set[Tuple[int, int]] = set()
    missing_nodes = 0
    for u, v in read_edgelist(prev_edgelist):
        cu = node_to_cluster.get(u, None)
        cv = node_to_cluster.get(v, None)
        if cu is None or cv is None:
            missing_nodes += 1
            continue
        if cu == cv:
            continue
        a, b = (cu, cv) if cu < cv else (cv, cu)
        edges_set.add((a, b))

    write_edgelist(out_edgelist, sorted(edges_set))
    return missing_nodes

def run_java(
    java_exec: str,
    class_name: str,
    edgelist_path: str,
    out_json_path: str,
    epsilon: str,
    java_opts: List[str],
    quiet: bool = False,
) -> None:
    """Run an existing compiled Java class.

    Note: Compilation has been removed to avoid overhead. Ensure classes are
    already compiled or on the classpath if you still use this function.
    """
    cmd = [java_exec] + (java_opts or []) + [class_name, edgelist_path, out_json_path, epsilon]
    if not quiet:
        print("[blue]Running:[/blue]", " ".join(cmd))

    run_kwargs = {'check': True}
    if quiet:
        run_kwargs['stdout'] = subprocess.DEVNULL
        run_kwargs['stderr'] = subprocess.DEVNULL
    subprocess.run(cmd, **run_kwargs)


# -----------------------------
# Pure-Python LaCore (RMC) port
# -----------------------------

class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n + 1))
        self.size = [0] * (n + 1)
        self.made = [False] * (n + 1)
        self.Q = [0.0] * (n + 1)

    def make_if_needed(self, v: int):
        if not self.made[v]:
            self.made[v] = True
            self.parent[v] = v
            self.size[v] = 1
            self.Q[v] = 0.0

    def find(self, v: int) -> int:
        if not self.made[v]:
            return v
        while self.parent[v] != v:
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v

    def union(self, a: int, b: int) -> int:
        self.make_if_needed(a)
        self.make_if_needed(b)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.Q[ra] += self.Q[rb]
        return ra


def _rmc_single_cluster_from_adj(adj1: List[List[int]], epsilon: float) -> Tuple[Set[int], float]:
    """
    Python port of synthetic.clique2_mk_benchmark_accuracy.runLaplacianRMC.
    Input adjacency is 1-based: adj1[1..n] neighbors are 1-based.
    Returns: (best_component_nodes_1based, bestSL)
    """
    import heapq

    n = len(adj1) - 1

    # Phase 1: peeling to get reverse order (addition order)
    deg0 = [0] * (n + 1)
    for i in range(1, n + 1):
        deg0[i] = len(adj1[i])
    pq = [(deg0[i], i) for i in range(1, n + 1)]
    heapq.heapify(pq)
    peel_stack: List[int] = []

    while pq:
        d, u = heapq.heappop(pq)
        if d != deg0[u]:
            continue  # stale entry
        peel_stack.append(u)
        for v in adj1[u]:
            if deg0[v] > 0:
                deg0[v] -= 1
                heapq.heappush(pq, (deg0[v], v))
        deg0[u] = 0

    # Build add order and index
    add_order = [0] * n
    idx = [0] * (n + 1)
    for t in range(n):
        u = peel_stack.pop()
        add_order[t] = u
        idx[u] = t

    # Phase 1.5: orient edges and sort successors by idx
    succ = [[] for _ in range(n + 1)]
    pred = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in adj1[u]:
            if u < v:  # handle undirected once
                if idx[u] < idx[v]:
                    succ[u].append(v)
                    pred[v].append(u)
                else:
                    succ[v].append(u)
                    pred[u].append(v)
    for v in range(1, n + 1):
        if len(succ[v]) > 1:
            succ[v].sort(key=lambda w: idx[w])

    # Phase 2: reverse reconstruction with O(k) per edge
    dsu = _DSU(n)
    deg = [0] * (n + 1)       # dynamic degree
    pred_sum = [0] * (n + 1)  # sum of degrees of predecessors

    bestSL = 0.0
    bestRoot = 0
    bestComponent: Set[int] = set()

    def sum_succ_until(v: int, T: int) -> int:
        s = 0
        for w in succ[v]:
            if idx[w] >= T:
                break
            s += deg[w]
        return s

    def snapshot_component(root: int) -> Set[int]:
        comp = set()
        for i in range(1, n + 1):
            if dsu.made[i] and dsu.find(i) == root:
                comp.add(i)
        return comp

    for u in add_order:
        dsu.make_if_needed(u)

        ru = dsu.find(u)
        sL = dsu.size[ru] / (dsu.Q[ru] + epsilon)
        if sL > bestSL:
            bestSL = sL
            bestRoot = ru
            bestComponent = snapshot_component(ru)

        Su = 0
        Tu = idx[u]

        for v in pred[u]:
            a = deg[u]
            b = deg[v]

            Sv = pred_sum[v] + sum_succ_until(v, Tu)

            dQu = 2 * a * a - 2 * Su + a
            dQv = 2 * b * b - 2 * Sv + b
            edgeTerm = (a - b) * (a - b)

            ru = dsu.find(u)
            rv = dsu.find(v)

            dsu.Q[ru] += float(dQu)
            dsu.Q[rv] += float(dQv)

            if ru != rv:
                r = dsu.union(ru, rv)
                dsu.Q[r] += float(edgeTerm)
            else:
                r = ru
                dsu.Q[r] += float(edgeTerm)

            sL = dsu.size[r] / (dsu.Q[r] + epsilon)
            if sL > bestSL:
                bestSL = sL
                bestComponent = snapshot_component(r)

            deg[u] += 1
            deg[v] += 1

            for y in succ[u]:
                pred_sum[y] += 1
            for y in succ[v]:
                pred_sum[y] += 1

            Su += deg[v]

    return bestComponent, float(bestSL)

def generate_lacore_cluster(
    edges: List[Tuple[int, int]],
    epsilon: float
) -> Dict:
    """
    Generate a single LaCore cluster directly in Python.

    Args:
        edges: Undirected 0-indexed edges (u, v) with u != v.
        epsilon: The epsilon value for the LaCore algorithm (float or str).

    Returns:
        dict with keys:
          - 'seed_nodes': list[int] (0-indexed nodes in the best component)
          - 'score': float (best size/(Q+epsilon))
    """
    # Accept epsilon as str for backwards-compat callers
    try:
        eps = float(epsilon)
    except Exception:
        eps = float(str(epsilon))

    if not edges:
        return {"seed_nodes": [], "score": 0.0}

    # Build 1-based adjacency
    max_node = -1
    for u, v in edges:
        if u == v:
            continue
        if u > max_node:
            max_node = u
        if v > max_node:
            max_node = v
    n = max_node + 1

    adj1: List[List[int]] = [[] for _ in range(n + 1)]
    seen: Set[Tuple[int, int]] = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        aa, bb = a + 1, b + 1
        adj1[aa].append(bb)
        adj1[bb].append(aa)

    best_comp_1b, bestSL = _rmc_single_cluster_from_adj(adj1, eps)
    seed_nodes_0b = sorted([u - 1 for u in best_comp_1b])
    return {"seed_nodes": seed_nodes_0b, "score": bestSL}


def generate_random_seeds(edgelist_path: str, num_nodes: int, out_json_path: str):
    """Generates a seeds file with a single cluster of randomly selected nodes."""
    nodes = set()
    for u, v in read_edgelist(edgelist_path):
        nodes.add(u)
        nodes.add(v)

    if num_nodes > len(nodes):
        print(f"[yellow]Warning: requested {num_nodes} random nodes, but only {len(nodes)} unique nodes exist. Selecting all nodes.[/yellow]")
        selected_nodes = list(nodes)
    else:
        selected_nodes = random.sample(list(nodes), num_nodes)

    selected_nodes_1_indexed = [n + 1 for n in selected_nodes]

    seed_data = {
        "clusters": [
            {
                "cluster_id": 1,
                "seed_nodes": selected_nodes_1_indexed,
                "score": 1.0
            }
        ]
    }

    json_write(seed_data, out_json_path)
    print(f"[green]Generated random seeds with {len(selected_nodes)} nodes at {out_json_path}[/green]")

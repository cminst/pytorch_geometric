import argparse
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Tuple

import numpy as np
import torch.multiprocessing as tmp
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
from tqdm import tqdm

import os
import torch
try:
    import wandb
except ImportError:
    wandb = None


from graph_classif_utils import (
    compute_lacore_cover_for_graph,
    LaCoreWrapped,
    Accumulator,
    seed_all,
    parse_seed_list,
)
from pooling_adapter import make_pool, apply_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb_run(run_name: str, config: dict, group: Optional[str] = None):
    if wandb is None:
        raise ImportError("wandb is required unless --save_only is set. Install wandb or rerun with --save_only.")
    return wandb.init(project="lacore_pooling", name=run_name, config=config, group=group)


def wandb_log(run, metrics: dict, prefix: Optional[str] = None, step: Optional[int] = None):
    if run is None:
        return
    payload = metrics if prefix is None else {f"{prefix}/{k}": v for k, v in metrics.items()}
    run.log(payload, step=step)


def _format_percent(ratio: Optional[float]) -> str:
    return f"{ratio * 100:g}%" if ratio is not None else "unknown%"


def make_wandb_labels(config) -> Tuple[str, Optional[str]]:
    """Build human-friendly run name and group for wandb."""
    pool = (getattr(config, "pool", None) or "lacore").lower()
    group_map = {"lacore": "LaCore-Exp", "asap": "ASAP-Exp", "topk": "TopK-Exp", "sag": "SAG-Exp", "edge": "Edge-Exp"}

    if pool == "lacore":
        eps_str = f"{config.epsilon:g}" if getattr(config, "epsilon", None) is not None else "unknown"
        run_name = f"LaCore-eps{eps_str}-{_format_percent(getattr(config, 'target_ratio', None))}"
    elif pool == "topk":
        run_name = f"TopK-{_format_percent(getattr(config, 'pool_ratio', None))}"
    elif pool == "sag":
        run_name = f"SAG-{_format_percent(getattr(config, 'pool_ratio', None))}"
    elif pool == "asap":
        run_name = f"ASAP-{_format_percent(getattr(config, 'pool_ratio', None))}"
    elif pool == "edge":
        run_name = "EdgePool"
    else:
        run_name = f"{getattr(config, 'dataset_name', 'run')}-{pool}"

    return run_name, group_map.get(pool)

def _compute_lacore_for_graph(args):
    """Worker to compute LaCore cover for a single graph."""
    data, epsilon, target_ratio, min_size, max_clusters = args
    d = data.clone()
    cluster, m, cluster_list = compute_lacore_cover_for_graph(d, epsilon, target_ratio, min_size, max_clusters)
    covered = sum(len(c) for c in cluster_list if len(c) > 1)
    N = cluster.shape[0]

    d.cluster = cluster
    d.num_clusters = torch.tensor([m], dtype=torch.long)
    return d, covered, N


def precompute_lacore_assignments(
    dataset,
    epsilon: float,
    target_ratio: float = 0.5,
    min_size: int = 4,
    max_clusters: Optional[int] = None,
    nproc: int = 2,
):
    """
    Adds:
      data.cluster : LongTensor [num_nodes]  (local cluster id)
      data.num_clusters : LongTensor [1]     (number of clusters)
    to each Data in the dataset.
    """
    if len(dataset) == 0:
        return LaCoreWrapped([])

    # Use file-based sharing to avoid torch shm permission issues when spawning.
    try:
        tmp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    ctx = None
    for method in ("spawn", "fork"):
        try:
            ctx = mp.get_context(method)
            break
        except ValueError:
            continue
    if ctx is None:
        ctx = mp.get_context()

    num_workers = max(1, min(len(dataset), nproc))
    tasks = ((dataset[i], epsilon, target_ratio, min_size, max_clusters) for i in range(len(dataset)))

    data_list = []
    pbar = tqdm(total=len(dataset), desc="Precomputing LaCore assignments", unit="graph")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        for d, covered, N in executor.map(_compute_lacore_for_graph, tasks, chunksize=1):
            pbar.set_postfix_str(f"Coverage (non-singleton): {covered}/{N} ({covered/N:.1%})")
            data_list.append(d)
            pbar.update(1)
    pbar.close()

    return LaCoreWrapped(data_list)


def get_cache_filename(
    dataset_name: str, split_name: str, epsilon: float, target_ratio: float, min_size: int, max_clusters: Optional[int]
) -> str:
    """Generate a human-readable cache filename."""
    max_clusters_str = str(max_clusters) if max_clusters is not None else "inf"
    split_suffix = f"_{split_name}" if split_name else ""
    return f"lacore_{dataset_name}{split_suffix}_eps{epsilon}_ratio{target_ratio}_min{min_size}_max{max_clusters_str}.pkl"


def save_lacore_cache(dataset, cache_path: str, config: dict):
    """Save LaCore-processed dataset to cache."""
    cache_data = {"dataset": dataset, "lacore_config": config, "version": "1.0"}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved LaCore cache to: {cache_path}")


def load_lacore_cache(cache_path: str):
    """Load LaCore-processed dataset from cache."""
    print(f"Loading LaCore cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    return cache_data["dataset"], cache_data["lacore_config"]


def validate_cache_config(cached_config: dict, current_config: dict) -> bool:
    """Check if cached config matches current LaCore config."""
    excluded = {
        "dataset_root",
        "batch_size",
        "hidden",
        "lr",
        "wd",
        "epochs",
        "dropout",
        "save_only",
        "force_recompute",
        "cache_dir",
        "cache_file_train",
        "cache_file_test",
        "val_ratio",
        "no_cache",
        "nproc",
    }
    for key in cached_config:
        if cached_config.get(key) != current_config.get(key) and (key not in excluded):
            print(f"\nCache config mismatch for {key}: cached={cached_config.get(key)}, current={current_config.get(key)}\n")
            return False
    return True


def get_or_compute_lacore_dataset(
    original_dataset,
    config,
    split_name: str,
    cache_file_override: Optional[str] = None,
):
    """Get LaCore dataset from cache or compute it for a specific split."""

    validation_config = dict(config)
    validation_config["split_name"] = split_name

    cache_path = None
    if not config.no_cache:
        if cache_file_override:
            if not os.path.exists(cache_file_override):
                raise FileNotFoundError(f"Specified cache file not found: {cache_file_override}")
            cache_path = cache_file_override
        elif config.cache_dir:
            os.makedirs(config.cache_dir, exist_ok=True)
            cache_filename = get_cache_filename(
                config.dataset_name, split_name, config.epsilon, config.target_ratio, config.min_size, config.max_clusters
            )
            cache_path = os.path.join(config.cache_dir, cache_filename)

    if cache_path and os.path.exists(cache_path) and not config.force_recompute:
        try:
            dataset, cached_config = load_lacore_cache(cache_path)
            if validate_cache_config(cached_config, validation_config):
                return dataset
            print("Cache config mismatch, recomputing...")
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}, recomputing...")

    print(f"Computing LaCore assignments for split '{split_name}'...")
    dataset = precompute_lacore_assignments(
        original_dataset,
        epsilon=config.epsilon,
        target_ratio=config.target_ratio,
        min_size=config.min_size,
        max_clusters=config.max_clusters,
        nproc=config.nproc,
    )

    if cache_path:
        cache_config = {
            "dataset_name": config.dataset_name,
            "split_name": split_name,
            "epsilon": config.epsilon,
            "target_ratio": config.target_ratio,
            "min_size": config.min_size,
            "max_clusters": config.max_clusters,
            "num_points": config.num_points,
            "knn_k": config.knn_k,
            "pool": config.pool,
            "pool_ratio": getattr(config, "pool_ratio", None),
        }
        save_lacore_cache(dataset, cache_path, cache_config)

    return dataset


class LaCorePool(nn.Module):
    def __init__(self, aggregate: str = "mean"):
        super().__init__()
        assert aggregate in ("mean",)
        self.aggregate = aggregate

    @torch.no_grad()
    def _build_pooled_edges(
        self, edge_index: torch.Tensor, batch: torch.Tensor, cluster: torch.Tensor, num_clusters_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Map original edges (u,v) to (Cu, Cv) using global cluster ids.
        Returns pooled edge_index [2, E'] (undirected, deduped) and optional edge_weight (None here).
        """
        B = int(num_clusters_vec.numel())
        offsets = torch.zeros(B, dtype=torch.long, device=cluster.device)
        offsets[1:] = torch.cumsum(num_clusters_vec[:-1], dim=0)

        global_cluster = cluster + offsets[batch]

        u, v = edge_index
        cu = global_cluster[u]
        cv = global_cluster[v]

        mask = cu != cv
        cu = cu[mask]
        cv = cv[mask]

        pooled = torch.stack([cu, cv], dim=0)
        pooled = to_undirected(pooled)
        pooled = torch.unique(pooled, dim=1)

        return pooled, None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        cluster: torch.Tensor,
        num_clusters_vec: torch.Tensor,
    ):
        """
        x: [N, F], edge_index: [2, E], batch: [N] (graph id per node)
        cluster: [N] local cluster id per node
        num_clusters_vec: [B] number of clusters for each graph in the batch
        """
        device = x.device
        B = int(num_clusters_vec.numel())
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(num_clusters_vec[:-1], dim=0)
        global_cluster = cluster + offsets[batch]

        M = int(num_clusters_vec.sum().item())
        if self.aggregate == "mean":
            x_pool = scatter_mean(x, global_cluster, dim=0, dim_size=M)

        edge_index_pooled, _ = self._build_pooled_edges(edge_index, batch, cluster, num_clusters_vec)
        batch_pooled = torch.arange(B, device=device).repeat_interleave(num_clusters_vec)

        return x_pool, edge_index_pooled, batch_pooled


class LaCorePoolNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        dropout: float = 0.2,
        pool_name: str = "lacore",
        pool_ratio: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.pool_name = (pool_name or "lacore").lower()
        if self.pool_name == "lacore":
            self.pool = LaCorePool(aggregate="mean")
        else:
            self.pool = make_pool(self.pool_name, in_channels=hidden, ratio=pool_ratio)

        self.lin = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x0 = F.relu(self.conv1(x, ei))
        x0 = F.dropout(x0, p=self.dropout, training=self.training)

        g0_mean = global_mean_pool(x0, batch)
        g0_max = global_max_pool(x0, batch)

        x1, ei1, batch1 = x0, ei, batch
        if self.pool_name == "lacore":
            if getattr(data, "cluster", None) is None or getattr(data, "num_clusters", None) is None:
                raise ValueError("LaCore pooling requires precomputed `cluster` and `num_clusters` fields.")
            x1, ei1, batch1 = self.pool(x0, ei, batch, cluster=data.cluster, num_clusters_vec=data.num_clusters)
        else:
            x1, ei1, batch1 = apply_pool(self.pool, data, x0, ei, batch)
            if batch1 is None:
                batch1 = batch

        x1 = F.relu(self.conv2(x1, ei1))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        g1_mean = global_mean_pool(x1, batch1)
        g1_max = global_max_pool(x1, batch1)

        g = torch.cat([g0_mean, g0_max, g1_mean, g1_max], dim=-1)
        out = self.lin(g)
        return out


def run_fold(
    train_loader,
    val_loader,
    test_loader,
    in_dim,
    num_classes,
    hidden=64,
    lr=1e-3,
    wd=5e-4,
    epochs=200,
    dropout=0.2,
    pool_name: str = "lacore",
    pool_ratio: float = 0.5,
    log_progress=True,
    wandb_run=None,
    log_prefix: Optional[str] = None,
    step_offset: int = 0,
):
    model = LaCorePoolNet(
        in_dim=in_dim,
        hidden=hidden,
        num_classes=num_classes,
        dropout=dropout,
        pool_name=pool_name,
        pool_ratio=pool_ratio,
    ).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0.0
    best_epoch = 0
    final_acc = Accumulator()
    final_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data)
            loss = cross_entropy(out, data.y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs

        model.eval()

        def _eval(loader, compute_loss=False):
            correct = 0
            total = 0
            total_loss_val = 0.0
            with torch.no_grad():
                for d in loader:
                    d = d.to(device)
                    logits = model(d)
                    if compute_loss:
                        loss = cross_entropy(logits, d.y)
                        total_loss_val += loss.item() * d.num_graphs

                    pred = logits.argmax(dim=-1)
                    correct += int((pred == d.y).sum())
                    total += d.num_graphs

            acc = correct / total
            return acc, total_loss_val / len(loader.dataset) if compute_loss else acc

        train_acc, _ = _eval(train_loader)
        val_acc, val_loss = _eval(val_loader, compute_loss=True)
        test_acc, _ = _eval(test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            final_acc.update_best(test_acc)
            final_metrics = {"train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc, "epoch": epoch}

        if log_progress and (epoch % 20 == 0 or epoch == 1):
            print(
                f"[{epoch:03d}] loss={total_loss/len(train_loader.dataset):.4f} "
                f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% test_acc={test_acc*100:.2f}% "
                f"(test_at_best_val={final_acc.value*100:.2f}%)"
            )
        wandb_log(
            wandb_run,
            {
                "epoch": step_offset + epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "test_acc": test_acc,
            },
            prefix=log_prefix,
            step=step_offset + epoch,
        )

    wandb_log(
        wandb_run,
        {
            "test_acc@val": final_acc.value,
            "test_acc@val%": 100 * final_acc.value,
            "best_epoch": best_epoch,
        },
        prefix=log_prefix,
        step=step_offset + epochs,
    )

    return final_acc.value, best_epoch, final_metrics


def make_modelnet40_graph_datasets(root: str, num_points: int = 1024, k: int = 16):
    """
    Build ModelNet40 train/test datasets as kNN graphs with x = pos.
    """

    class PosToX(object):
        def __call__(self, data):
            data.x = data.pos
            return data

    transform = T.Compose(
        [
            T.SamplePoints(num_points),
            T.NormalizeScale(),
            T.KNNGraph(k=k, force_undirected=True),
            PosToX(),
        ]
    )

    train_ds = ModelNet(root=root, name="40", train=True, transform=transform)
    test_ds = ModelNet(root=root, name="40", train=False, transform=transform)
    return train_ds, test_ds


def stratified_train_val_split(labels: torch.Tensor, val_ratio: float, seed: int):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, stratify=labels.cpu().numpy(), random_state=seed)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


def run_modelnet_experiment(
    dataset_root: str = "data/ModelNet",
    dataset_name: str = "ModelNet40",
    num_points: int = 1024,
    knn_k: int = 16,
    seed: int = 42,
    epsilon: float = 1e3,
    target_ratio: float = 0.5,
    min_size: int = 4,
    max_clusters: Optional[int] = None,
    nproc: int = 2,
    batch_size: int = 32,
    hidden: int = 64,
    lr: float = 0.0001,
    wd: float = 0.001,
    epochs: int = 300,
    dropout: float = 0.2,
    pool: str = "lacore",
    pool_ratio: float = 0.5,
    val_ratio: float = 0.1,
    cache_dir: Optional[str] = "./lacore_cache",
    cache_file_train: Optional[str] = None,
    cache_file_test: Optional[str] = None,
    save_only: bool = False,
    force_recompute: bool = False,
    no_cache: bool = False,
):
    """
    Run LaCore pooling on ModelNet40 using the canonical train/test split.
    """
    config_dict = locals()
    config = edict(config_dict)
    config.pool = (config.pool or "lacore").lower()
    use_lacore_pool = config.pool == "lacore"
    if not use_lacore_pool:
        config.no_cache = True
        if config.save_only:
            print("Non-LaCore pooling selected; ignoring --save_only because no LaCore caches are produced.")
            config.save_only = False

    print("Using configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nLoading ModelNet40 (train/test splits)...")
    train_raw, test_raw = make_modelnet40_graph_datasets(config.dataset_root, num_points=config.num_points, k=config.knn_k)
    print(f"Train graphs: {len(train_raw)} | Test graphs: {len(test_raw)}")

    if use_lacore_pool:
        train_dataset = get_or_compute_lacore_dataset(
            train_raw, config, split_name="train", cache_file_override=config.cache_file_train
        )
        test_dataset = get_or_compute_lacore_dataset(
            test_raw, config, split_name="test", cache_file_override=config.cache_file_test
        )
    else:
        print(f"Using pool='{config.pool}'. Skipping LaCore preprocessing and disk caches.")
        train_dataset = LaCoreWrapped([train_raw[i] for i in range(len(train_raw))])
        test_dataset = LaCoreWrapped([test_raw[i] for i in range(len(test_raw))])

    if config.save_only:
        print("LaCore assignments computed and saved. Exiting.")
        return

    num_classes = int(train_dataset.num_classes)
    assert train_dataset.num_features > 0, "ModelNet graphs must carry node features."
    in_dim = train_dataset.num_features

    labels = torch.tensor([data.y.item() for data in train_dataset])
    seed_values = parse_seed_list(config.seed)
    run_name_base, run_group = make_wandb_labels(config)

    results = []
    for seed_value in seed_values:
        seed_all(seed_value)
        run = None
        if not config.save_only:
            wandb_config = dict(config)
            wandb_config.update({"seed": seed_value, "in_dim": in_dim, "num_classes": num_classes})
            if len(seed_values) > 1:
                run_name = f"{run_name_base}-seed{seed_value}"
            else:
                run_name = run_name_base
            run = init_wandb_run(run_name=run_name, config=wandb_config, group=run_group)

        train_idx, val_idx = stratified_train_val_split(labels, config.val_ratio, seed_value)

        train_loader = DataLoader(train_dataset.index_select(train_idx), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset.index_select(val_idx), batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        best, best_epoch, metrics = run_fold(
            train_loader,
            val_loader,
            test_loader,
            in_dim,
            num_classes,
            hidden=config.hidden,
            lr=config.lr,
            wd=config.wd,
            epochs=config.epochs,
            dropout=config.dropout,
            pool_name=config.pool,
            pool_ratio=config.pool_ratio,
            log_progress=len(seed_values) == 1,
            wandb_run=run,
        )
        results.append(best)

        print(f"\nSeed {seed_value}: test_acc@val={100*best:.2f}% at epoch {best_epoch}")
        print(f"  train_acc={100*metrics.get('train_acc', 0):.2f}% val_acc={100*metrics.get('val_acc', 0):.2f}%")

        if run is not None:
            wandb_log(
                run,
                {
                    "final_seed_test_acc@val": best,
                    "final_seed_test_acc@val%": 100 * best,
                    "best_epoch": best_epoch,
                },
                step=config.epochs + 1,
            )
            run.finish()

    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results)) if len(results) > 1 else 0.0
    print(f"\nTest accuracy (mean over seeds): {100*mean_acc:.2f}% ± {100*std_acc:.2f}%")
    return mean_acc, std_acc


def main():
    default_config = {
        "dataset_root": "data/ModelNet",
        "dataset_name": "ModelNet40",
        "num_points": 1024,
        "knn_k": 16,
        "seed": 42,
        "epsilon": 1e3,
        "target_ratio": 0.5,
        "min_size": 4,
        "max_clusters": None,
        "nproc": 2,
        "batch_size": 32,
        "hidden": 64,
        "lr": 0.0001,
        "wd": 0.001,
        "epochs": 500,
        "dropout": 0.2,
        "pool": "lacore",
        "pool_ratio": 0.5,
        "val_ratio": 0.1,
        "cache_dir": "./lacore_cache",
        "cache_file_train": None,
        "cache_file_test": None,
        "save_only": False,
        "force_recompute": False,
        "no_cache": False,
    }

    parser = argparse.ArgumentParser(description="LaCore Pooling on ModelNet40 (canonical split)")

    parser.add_argument("--dataset_root", type=str, default=default_config["dataset_root"], help="Root directory for ModelNet")
    parser.add_argument("--dataset_name", type=str, default=default_config["dataset_name"], help="Dataset name (kept for logging)")
    parser.add_argument("--num_points", type=int, default=default_config["num_points"], help="Number of points sampled per mesh")
    parser.add_argument("--knn_k", type=int, default=default_config["knn_k"], help="k for KNN graph construction")
    parser.add_argument("--seed", type=str, default=default_config["seed"], help="Random seed(s), comma-separated for multiple")
    parser.add_argument("--epsilon", type=float, default=default_config["epsilon"], help="Epsilon parameter for LaCore clustering")
    parser.add_argument(
        "--target_ratio", type=float, default=default_config["target_ratio"], help="Target ratio of nodes covered by multi-node clusters"
    )
    parser.add_argument("--min_size", type=int, default=default_config["min_size"], help="Minimum cluster size to keep")
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=default_config["max_clusters"],
        nargs="?",
        help="Maximum clusters per graph (use None to disable limit)",
    )
    parser.add_argument("--nproc", type=int, default=default_config["nproc"], help="Worker processes for LaCore preprocessing")
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"], help="Batch size for training")
    parser.add_argument("--hidden", type=int, default=default_config["hidden"], help="Hidden dimension for GCN layers")
    parser.add_argument("--lr", type=float, default=default_config["lr"], help="Learning rate")
    parser.add_argument("--wd", type=float, default=default_config["wd"], help="Weight decay")
    parser.add_argument("--epochs", type=int, default=default_config["epochs"], help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=default_config["dropout"], help="Dropout probability")
    parser.add_argument(
        "--pool",
        type=str,
        default=default_config["pool"],
        choices=["lacore", "topk", "sag", "asap", "edge"],
        help="Pooling method; non-LaCore options skip cache usage and LaCore preprocessing",
    )
    parser.add_argument(
        "--pool_ratio",
        type=float,
        default=default_config["pool_ratio"],
        help="Pooling ratio for TopK/SAG/ASAP (ignored for LaCore/Edge pooling)",
    )
    parser.add_argument("--val_ratio", type=float, default=default_config["val_ratio"], help="Validation ratio taken from the train split")

    parser.add_argument("--cache_dir", type=str, default=default_config["cache_dir"], help="Directory to store/load LaCore caches")
    parser.add_argument(
        "--cache_file_train",
        type=str,
        default=default_config["cache_file_train"],
        help="Specific cache file to load for the train split (overrides auto-generated cache)",
    )
    parser.add_argument(
        "--cache_file_test",
        type=str,
        default=default_config["cache_file_test"],
        help="Specific cache file to load for the test split (overrides auto-generated cache)",
    )
    parser.add_argument("--save_only", action="store_true", help="Only compute and save LaCore assignments, then exit")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation even if cache exists")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching (compute fresh each time)")

    args = parser.parse_args()

    config_dict = {}
    for key, default_val in default_config.items():
        val = getattr(args, key)
        if isinstance(default_val, int) and val == "None":
            val = None
        elif isinstance(default_val, float) and val == "None":
            val = None
        elif isinstance(default_val, type(None)) and val == "None":
            val = None
        elif val is not None and isinstance(default_val, int) and isinstance(val, str):
            try:
                val = int(val)
            except ValueError:
                pass
        elif val is not None and isinstance(default_val, float) and isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        config_dict[key] = val

    run_modelnet_experiment(**config_dict)


if __name__ == "__main__":
    main()

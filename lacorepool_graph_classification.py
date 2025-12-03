import torch
from datetime import timedelta
from typing import Tuple, Optional
import argparse
import hashlib
import pickle
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as tmp
from epsilon_seed_sweep import mean
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import numpy as np
from torch_geometric.datasets import TUDataset
from easydict import EasyDict as edict
from datetime import datetime
import json
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_wandb_run(run_name: str, config: dict):
    if wandb is None:
        raise ImportError("wandb is required unless --save_only is set. Install wandb or rerun with --save_only.")
    return wandb.init(project="lacore_pooling", name=run_name, config=config)


def wandb_log(run, metrics: dict, prefix: Optional[str] = None, step: Optional[int] = None):
    if run is None:
        return
    payload = metrics if prefix is None else {f"{prefix}/{k}": v for k, v in metrics.items()}
    run.log(payload, step=step)


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
    nproc: int = 2
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

def get_lacore_cache_key(dataset_name: str, epsilon: float, target_ratio: float, min_size: int, max_clusters: Optional[int]) -> str:
    """Generate a unique cache key based on LaCore parameters."""
    key_data = {
        'dataset_name': dataset_name,
        'epsilon': epsilon,
        'target_ratio': target_ratio,
        'min_size': min_size,
        'max_clusters': max_clusters,
    }
    key_str = str(sorted(key_data.items()))
    return hashlib.md5(key_str.encode()).hexdigest()[:16]

def save_lacore_cache(dataset, cache_path: str, config: dict):
    """Save LaCore-processed dataset to cache."""
    cache_data = {
        'dataset': dataset,
        'lacore_config': config,
        'version': '1.0'
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Saved LaCore cache to: {cache_path}")

def load_lacore_cache(cache_path: str):
    """Load LaCore-processed dataset from cache."""
    print(f"Loading LaCore cache from: {cache_path}")
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    return cache_data['dataset'], cache_data['lacore_config']

def get_cache_filename(dataset_name: str, epsilon: int, target_ratio: float, min_size: int, max_clusters: Optional[int]) -> str:
    """Generate a human-readable cache filename."""
    max_clusters_str = str(max_clusters) if max_clusters is not None else "inf"
    return f"lacore_{dataset_name}_eps{epsilon}_ratio{target_ratio}_min{min_size}_max{max_clusters_str}.pkl"

def validate_cache_config(cached_config: dict, current_config: dict) -> bool:
    """Check if cached config matches current LaCore config."""
    excluded = {'dataset_root', 'batch_size', 'hidden', 'lr', 'wd', 'epochs', 'dropout', 'save_only', 'force_recompute', 'cache_dir', 'cache_file', 'protocol', 'use_node_attr', 'num_proc', 'nproc'}
    for key in cached_config:
        if cached_config.get(key) != current_config.get(key) and (key not in excluded):
            print(f"\nCache config mismatch for {key}: cached={cached_config.get(key)}, current={current_config.get(key)}")
            print(f"Cached Config: {cached_config}")
            print(f"Current Config: {current_config}\n\n")
            return False
    return True

def get_or_compute_lacore_dataset(original_dataset, config):
    """Get LaCore dataset from cache or compute it."""

    if config.cache_file:
        if not os.path.exists(config.cache_file):
            raise FileNotFoundError(f"Specified cache file not found: {config.cache_file}")

        dataset, cached_config = load_lacore_cache(config.cache_file)
        if not validate_cache_config(cached_config, config):
            if not config.force_recompute:
                raise ValueError("Cache config mismatch. Use --force_recompute to ignore.")
            print("Config mismatch detected, but --force_recompute not set. Using cached data anyway.")
        return dataset

    if config.cache_dir:
        os.makedirs(config.cache_dir, exist_ok=True)
        cache_filename = get_cache_filename(
            config.dataset_name, config.epsilon, config.target_ratio,
            config.min_size, config.max_clusters
        )
        cache_path = os.path.join(config.cache_dir, cache_filename)

        if os.path.exists(cache_path) and not config.force_recompute:
            try:
                dataset, cached_config = load_lacore_cache(cache_path)
                if validate_cache_config(cached_config, config):
                    return dataset
                else:
                    print("Cache config mismatch, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}, recomputing...")

    print("Computing LaCore assignments...")
    dataset = precompute_lacore_assignments(
        original_dataset,
        epsilon=config.epsilon,
        target_ratio=config.target_ratio,
        min_size=config.min_size,
        max_clusters=config.max_clusters,
        nproc=config.nproc
    )

    if config.cache_dir:
        save_lacore_cache(dataset, cache_path, config)

    return dataset


class LaCorePool(nn.Module):
    def __init__(self, aggregate: str = "mean"):
        super().__init__()
        assert aggregate in ("mean",)
        self.aggregate = aggregate

    @torch.no_grad()
    def _build_pooled_edges(
        self, edge_index: torch.Tensor, batch: torch.Tensor,
        cluster: torch.Tensor, num_clusters_vec: torch.Tensor
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

        mask = (cu != cv)
        cu = cu[mask]; cv = cv[mask]

        pooled = torch.stack([cu, cv], dim=0)

        pooled = to_undirected(pooled)
        pooled = torch.unique(pooled, dim=1)

        return pooled, None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
        cluster: torch.Tensor, num_clusters_vec: torch.Tensor
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
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.pool = LaCorePool(aggregate="mean")
        self.dropout = dropout

        self.lin = nn.Sequential(
            nn.Linear(4*hidden, 2*hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden, num_classes),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x0 = F.relu(self.conv1(x, ei))
        x0 = F.dropout(x0, p=self.dropout, training=self.training)

        g0_mean = global_mean_pool(x0, batch)
        g0_max  = global_max_pool(x0, batch)

        x1, ei1, batch1 = self.pool(
            x0, ei, batch, cluster=data.cluster, num_clusters_vec=data.num_clusters
        )

        x1 = F.relu(self.conv2(x1, ei1))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        g1_mean = global_mean_pool(x1, batch1)
        g1_max  = global_max_pool(x1, batch1)

        g = torch.cat([g0_mean, g0_max, g1_mean, g1_max], dim=-1)
        out = self.lin(g)
        return out


def make_loaders(dataset, train_idx, test_idx, batch_size=64):
    train_ds = dataset.index_select(train_idx)
    test_ds  = dataset.index_select(test_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


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
    is_multilabel=False,
    log_progress=True,
    wandb_run=None,
    log_prefix: Optional[str] = None,
    step_offset: int = 0,
):
    model = LaCorePoolNet(in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0.0
    best_epoch = 0
    final_acc = Accumulator()
    final_metrics = {}

    for epoch in range(1, epochs+1):
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
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for d in loader:
                    d = d.to(device)
                    logits = model(d)
                    if compute_loss:
                        loss = cross_entropy(logits, d.y)
                        total_loss_val += loss.item() * d.num_graphs

                    if is_multilabel:
                        preds = (logits > 0).float()
                        all_preds.append(preds.cpu())
                        all_labels.append(d.y.cpu())
                    else:
                        pred = logits.argmax(dim=-1)
                        correct += int((pred == d.y).sum())
                        total += d.num_graphs

            if is_multilabel:
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                ap = average_precision_score(all_labels.numpy(), all_preds.numpy(), average='macro')
                return ap, total_loss_val / len(loader.dataset) if compute_loss else ap
            else:
                acc = correct / total
                return acc, total_loss_val / len(loader.dataset) if compute_loss else acc

        train_acc, _ = _eval(train_loader)
        val_acc, val_loss = _eval(val_loader, compute_loss=True)
        test_acc, _ = _eval(test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if is_multilabel:
                final_acc.value = test_acc
            else:
                final_acc.update_best(test_acc)

            final_metrics = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
                'epoch': epoch
            }

        if log_progress and (epoch % 20 == 0 or epoch == 1):
            if is_multilabel:
                print(f"[{epoch:03d}] loss={total_loss/len(train_loader.dataset):.4f} "
                      f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% test_ap={test_acc*100:.2f}% (test_at_best_val={final_acc.value*100:.2f}%)")
            else:
                print(f"[{epoch:03d}] loss={total_loss/len(train_loader.dataset):.4f} "
                      f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% test_acc={test_acc*100:.2f}% (test_at_best_val={final_acc.value*100:.2f}%)")

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


def _run_single_fold_job(args):
    (
        fold_id,
        seed_value,
        train_idx,
        test_idx,
        labels,
        dataset,
        worker_cfg,
        in_dim,
        num_classes,
        is_multilabel,
        log_progress,
    ) = args

    seed_all(seed_value)

    train_labels = labels[train_idx]
    skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    train_sub_idx, val_sub_idx = next(skf_val.split(np.arange(len(train_idx)), train_labels))

    fold_train_idx = train_idx[train_sub_idx]
    fold_val_idx = train_idx[val_sub_idx]

    train_loader = DataLoader(dataset.index_select(fold_train_idx), batch_size=worker_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset.index_select(fold_val_idx), batch_size=worker_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset.index_select(test_idx), batch_size=worker_cfg['batch_size'], shuffle=False)

    best, best_epoch, _ = run_fold(
        train_loader, val_loader, test_loader, in_dim, num_classes,
        hidden=worker_cfg['hidden'], lr=worker_cfg['lr'], wd=worker_cfg['wd'],
        epochs=worker_cfg['epochs'], dropout=worker_cfg['dropout'], is_multilabel=is_multilabel,
        log_progress=log_progress
    )

    return {
        'fold_id': fold_id,
        'score': best,
        'best_epoch': best_epoch,
    }

def get_gpn24_hyperparams(dataset_name: str) -> dict:
    """Hyperparameters from Table 5"""
    gpn24_params = {
        'DD': {'hidden': 128, 'lr': 1e-4, 'dropout': 0.15, 'batch_size': 64, 'epochs': 500},
        'PROTEINS': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.1, 'batch_size': 128, 'epochs': 500},
        'NCI1': {'hidden': 256, 'lr': 5e-4, 'dropout': 0.4, 'batch_size': 256, 'epochs': 500},
        'NCI109': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.2, 'batch_size': 64, 'epochs': 500},
        'FRANKENSTEIN': {'hidden': 128, 'lr': 1e-3, 'dropout': 0.2, 'batch_size': 128, 'epochs': 500},
    }
    return gpn24_params.get(dataset_name, gpn24_params['PROTEINS'])

def run_lacore_pool_experiment(
    dataset_root: str = "data",
    dataset_name: str = "PROTEINS",
    kfold_splits: int = 10,
    num_proc: int = 1,
    seed: int = 42,
    epsilon: float = 1e3,
    target_ratio: float = 0.5,
    min_size: int = 4,
    max_clusters: Optional[int] = None,
    nproc: int = 2,
    batch_size: int = 64,
    hidden: int = 64,
    lr: float = 0.0001,
    wd: float = 0.001,
    epochs: int = 500,
    dropout: float = 0.2,
    cache_dir: Optional[str] = "./lacore_cache",
    cache_file: Optional[str] = None,
    save_only: bool = False,
    force_recompute: bool = False,
    no_cache: bool = False,
    use_node_attr: bool = False,
    protocol: Optional[str] = None,
    log_dir: Optional[str] = "./gpn24_logs",
):
    """
    Runs the LaCore pooling experiment with the given configuration.

    If protocol='gpn24', runs GPN ICLR'24 evaluation protocol (20 seeds x 10 folds).

    Returns:
        A tuple of (mean_accuracy, std_accuracy) across folds if not in save_only mode.
        For gpn24 protocol, returns aggregated results across all seeds and folds.
    """
    config_dict = locals()
    if config_dict.pop('no_cache'):
        config_dict['cache_dir'] = None

    config = edict(config_dict)

    print("Using configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    use_node_attr = config.use_node_attr or (config.dataset_name.upper() == "ENZYMES")

    original_dataset = TUDataset(root=config.dataset_root, name=config.dataset_name, use_node_attr=use_node_attr)
    print(original_dataset)

    dataset = get_or_compute_lacore_dataset(original_dataset, config)

    if config.save_only:
        print("LaCore assignments computed and saved. Exiting.")
        return

    is_multilabel = False

    if is_multilabel:
        labels = torch.stack([data.y for data in dataset])
        if labels.dim() > 1:
            simple_labels = (labels.sum(dim=1) > 0).long()
        else:
            simple_labels = labels.long()
    else:
        labels = torch.tensor([data.y.item() for data in dataset])
        simple_labels = labels

    num_classes = int(dataset.num_classes)
    assert dataset.num_features > 0, \
        "No node features found. For ENZYMES pass --use_node_attr. Otherwise add a degree-based transform."
    in_dim = dataset.num_features

    if config.protocol == 'gpn24':
        return run_gpn24_protocol(dataset, simple_labels, in_dim, num_classes, config, is_multilabel)

    seed_values = parse_seed_list(config.seed)
    log_folds_verbose = (len(seed_values) == 1)
    results = []
    config.seed = seed_values if len(seed_values) > 1 else seed_values[0]

    worker_cfg = {
        'batch_size': config.batch_size,
        'hidden': config.hidden,
        'lr': config.lr,
        'wd': config.wd,
        'epochs': config.epochs,
        'dropout': config.dropout,
    }

    for seed_value in seed_values:
        seed_all(seed_value)
        wandb_run = None
        wandb_base_config = dict(config)
        wandb_base_config.update({"seed": seed_value, "in_dim": in_dim, "num_classes": num_classes, "is_multilabel": is_multilabel})
        run_name = f"{config.dataset_name}-seed{seed_value}"
        wandb_run = init_wandb_run(run_name=run_name, config=wandb_base_config)

        skf = StratifiedKFold(n_splits=config.kfold_splits, shuffle=True, random_state=seed_value)
        fold_indices = [(train_idx, test_idx) for train_idx, test_idx in skf.split(torch.arange(len(dataset)), simple_labels)]

        fold_scores = []
        num_workers = max(1, int(config.num_proc))
        num_workers = min(num_workers, len(fold_indices))
        use_parallel = num_workers > 1 and wandb_run is None

        if use_parallel:
            jobs = []
            for k, (train_idx, test_idx) in enumerate(fold_indices, 1):
                if log_folds_verbose:
                    print(f"\n=== Fold {k}/{config.kfold_splits} === (queued)")
                jobs.append((
                    k, seed_value, train_idx, test_idx, labels, dataset,
                    worker_cfg, in_dim, num_classes, is_multilabel, False
                ))

            with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as executor:
                for res in executor.map(_run_single_fold_job, jobs):
                    fold_scores.append(res)
                    if log_folds_verbose:
                        print(f"\n=== Fold {res['fold_id']}/{config.kfold_splits} === completed "
                              f"(test_at_best_val={100*res['score']:.2f}%)")
                    else:
                        print(f"\n=== Fold {res['fold_id']}/{config.kfold_splits} === "
                              f"(test_at_best_val={100*res['score']:.2f}%)")

            ordered_scores = [r['score'] for r in sorted(fold_scores, key=lambda x: x['fold_id'])]
        else:
            for k, (train_idx, test_idx) in enumerate(fold_indices, 1):
                fold_header = f"=== Fold {k}/{config.kfold_splits} ==="
                if log_folds_verbose:
                    print(f"\n{fold_header}")

                train_labels = labels[train_idx]
                skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
                train_sub_idx, val_sub_idx = next(skf_val.split(np.arange(len(train_idx)), train_labels))

                fold_train_idx = train_idx[train_sub_idx]
                fold_val_idx = train_idx[val_sub_idx]

                train_loader = DataLoader(dataset.index_select(fold_train_idx), batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(dataset.index_select(fold_val_idx), batch_size=config.batch_size, shuffle=False)
                test_loader = DataLoader(dataset.index_select(test_idx), batch_size=config.batch_size, shuffle=False)

                best, best_epoch, _ = run_fold(
                    train_loader, val_loader, test_loader, in_dim, num_classes,
                    hidden=config.hidden, lr=config.lr, wd=config.wd,
                    epochs=config.epochs, dropout=config.dropout, is_multilabel=is_multilabel,
                    log_progress=log_folds_verbose,
                    wandb_run=wandb_run,
                    log_prefix=f"fold{k}",
                    step_offset=(k - 1) * config.epochs,
                )
                fold_scores.append({'fold_id': k, 'score': best})

                if not log_folds_verbose:
                    print(f"\n{fold_header} (test_at_best_val={100*best:.2f}%)")

            ordered_scores = [r['score'] for r in fold_scores]

        mean_acc = mean(ordered_scores)
        std_acc = np.std(ordered_scores)
        print(f"\nSeed {seed_value} | epsilon={config.epsilon} | CV mean acc: {100*mean_acc:.2f}% ± {100*std_acc:.2f}%")
        results.append({'seed': seed_value, 'mean_acc': mean_acc, 'std_acc': std_acc})

        if wandb_run is not None:
            wandb_log(
                wandb_run,
                {
                    "cv_mean_acc": mean_acc,
                    "cv_std_acc": std_acc,
                },
                step=config.epochs * config.kfold_splits,
            )
            wandb_run.finish()

    if len(results) == 1:
        return results[0]['mean_acc'], results[0]['std_acc']
    return results

def run_gpn24_protocol(dataset, labels, in_dim, num_classes, config, is_multilabel):
    """Run GPN ICLR'24 evaluation protocol: 20 seeds x 10 folds"""
    print("\n=== Running GPN ICLR'24 Protocol ===")
    print(f"Dataset: {config.dataset_name}")
    print("20 seeds × 10 folds = 200 total runs")
    print(f"Multilabel: {is_multilabel}")

    gpn24_params = get_gpn24_hyperparams(config.dataset_name)
    print(f"Using GPN24 hyperparameters: {gpn24_params}")

    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        log_file = os.path.join(config.log_dir, f"gpn24_{config.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    else:
        log_file = None

    all_results = []
    seed_results = []
    start_time = datetime.now()

    for seed in range(20):
        print(f"\n{'='*60}")
        print(f"Seed {seed+1}/20 - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        seed_all(seed)
        run_config = dict(config)
        run_config.update(gpn24_params)
        run_config.update({"seed": seed, "in_dim": in_dim, "num_classes": num_classes, "is_multilabel": is_multilabel})
        wandb_run = init_wandb_run(run_name=f"{config.dataset_name}-gpn24-seed{seed}", config=run_config)
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_indices = list(folds.split(torch.arange(len(dataset)), labels))

        fold_scores = []
        fold_details = []

        for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
            print(f"\nFold {fold_id+1}/10")

            train_labels = labels[train_idx]
            skf_val = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            train_sub_idx, val_sub_idx = next(skf_val.split(np.arange(len(train_idx)), train_labels))

            fold_train_idx = train_idx[train_sub_idx]
            fold_val_idx = train_idx[val_sub_idx]

            train_loader = DataLoader(dataset.index_select(fold_train_idx), batch_size=gpn24_params['batch_size'], shuffle=True)
            val_loader = DataLoader(dataset.index_select(fold_val_idx), batch_size=gpn24_params['batch_size'], shuffle=False)
            test_loader = DataLoader(dataset.index_select(test_idx), batch_size=gpn24_params['batch_size'], shuffle=False)

            best_score, best_epoch, metrics = run_fold(
                train_loader, val_loader, test_loader,
                in_dim, num_classes,
                hidden=gpn24_params['hidden'],
                lr=gpn24_params['lr'],
                wd=config.wd,
                epochs=gpn24_params['epochs'],
                dropout=gpn24_params['dropout'],
                is_multilabel=is_multilabel,
                wandb_run=wandb_run,
                log_prefix=f"fold{fold_id + 1}",
                step_offset=fold_id * gpn24_params['epochs'],
            )

            fold_scores.append(best_score)
            fold_details.append({
                'fold': fold_id + 1,
                'best_epoch': best_epoch,
                'test_score': float(best_score),
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'val_idx': val_sub_idx.tolist(),
                'metrics': metrics
            })

            print(f"Fold {fold_id+1} best: epoch {best_epoch}, score: {best_score:.4f}")

        seed_mean = np.mean(fold_scores)
        seed_std = np.std(fold_scores)
        seed_results.append(seed_mean)

        print(f"\nSeed {seed+1} mean: {seed_mean:.4f} ± {seed_std:.4f}")

        seed_data = {
            'seed': seed,
            'dataset': config.dataset_name,
            'fold_scores': [float(s) for s in fold_scores],
            'seed_mean': float(seed_mean),
            'seed_std': float(seed_std),
            'hyperparams': gpn24_params,
            'fold_details': fold_details,
            'timestamp': datetime.now().isoformat()
        }
        all_results.append(seed_data)

        if log_file:
            with open(log_file, 'a') as f:
                f.write(json.dumps(seed_data) + '\n')

        elapsed = (datetime.now() - start_time).total_seconds()
        avg_time_per_seed = elapsed / (seed + 1)
        remaining_seeds = 19 - seed
        eta_seconds = avg_time_per_seed * remaining_seeds

        print(f"Progress: {seed+1}/20 seeds completed")
        print(f"Time per seed: {avg_time_per_seed:.1f}s")
        if remaining_seeds > 0:
            eta_seconds = avg_time_per_seed * remaining_seeds
            print(f"ETA: {timedelta(seconds=int(eta_seconds))}")
        else:
            print("ETA: 00:00:00")

        wandb_log(
            wandb_run,
            {
                "cv_mean_acc": seed_mean,
                "cv_std_acc": seed_std,
            },
            step=gpn24_params['epochs'] * 10,
        )
        wandb_run.finish()

    final_mean = np.mean(seed_results)
    final_std = np.std(seed_results)

    print(f"\n{'='*60}")
    print(f"GPN24 Protocol Results for {config.dataset_name}")
    print(f"{'='*60}")
    print(f"Final mean: {final_mean:.4f} ± {final_std:.4f}")
    if is_multilabel:
        print("Metric: Average Precision (AP)")
    else:
        print("Metric: Accuracy")
    print("Total runs: 200 (20 seeds × 10 folds)")
    print(f"Total time: {(datetime.now() - start_time).total_seconds():.1f}s")

    return final_mean, final_std


def main():
    default_config = {
        "dataset_root": "data",
        "dataset_name": "PROTEINS",
        "kfold_splits": 10,
        "num_proc": 1,
        "seed": 42,
        "epsilon": 1e3,
        "target_ratio": 0.5,
        "min_size": 4,
        "max_clusters": None,
        "nproc": 2,
        "batch_size": 64,
        "hidden": 64,
        "lr": 0.0001,
        "wd": 0.001,
        "epochs": 500,
        "dropout": 0.2,
        "cache_dir": "./lacore_cache",
        "cache_file": None,
        "save_only": False,
        "force_recompute": False,
        "use_node_attr": False,
        "protocol": None,
        "log_dir": "./gpn24_logs",
    }

    parser = argparse.ArgumentParser(description="LaCore Pooling Graph Classifier")

    parser.add_argument("--dataset_root", type=str, default=default_config["dataset_root"],
                        help="Root directory for datasets")
    parser.add_argument("--dataset_name", type=str, default=default_config["dataset_name"],
                        help="Name of the TUDataset to use (e.g., PROTEINS, MUTAG)")
    parser.add_argument("--kfold_splits", type=int, default=default_config["kfold_splits"],
                        help="Number of folds for cross-validation")
    parser.add_argument("--num-proc", dest="num_proc", type=int, default=default_config["num_proc"],
                        help="Number of folds to train in parallel")
    parser.add_argument("--seed", type=str, default=default_config["seed"],
                        help="Random seed for reproducibility (comma-separated for multiple)")
    parser.add_argument("--epsilon", type=float, default=default_config["epsilon"],
                        help="Epsilon parameter for LaCore clustering")
    parser.add_argument("--target_ratio", type=float, default=default_config["target_ratio"],
                        help="Target ratio of nodes to cover with multi-node clusters")
    parser.add_argument("--min_size", type=int, default=default_config["min_size"],
                        help="Minimum cluster size to consider (ignore smaller ones)")
    parser.add_argument("--max_clusters", type=int, default=default_config["max_clusters"], nargs="?",
                        help="Maximum number of clusters per graph (use 'None' to disable limit)")
    parser.add_argument("--nproc", type=int, default=default_config["nproc"],
                        help="Worker processes for LaCore preprocessing")
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"],
                        help="Batch size for training")
    parser.add_argument("--hidden", type=int, default=default_config["hidden"],
                        help="Hidden dimension size for GCN layers")
    parser.add_argument("--lr", type=float, default=default_config["lr"],
                        help="Learning rate")
    parser.add_argument("--wd", type=float, default=default_config["wd"],
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=default_config["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=default_config["dropout"],
                        help="Dropout probability")

    parser.add_argument("--cache_dir", type=str, default=default_config["cache_dir"],
                        help="Directory to store/load LaCore caches (default: ./lacore_cache)")
    parser.add_argument("--cache_file", type=str, default=default_config["cache_file"],
                        help="Specific cache file to load (overrides auto-generated cache)")
    parser.add_argument("--save_only", action="store_true",
                        help="Only compute and save LaCore assignments, then exit")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputation even if cache exists")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching (compute fresh each time)")
    parser.add_argument("--use_node_attr", action="store_true",
                        help="Include continuous node attributes if available (ENZYMES needs this)")
    parser.add_argument("--protocol", type=str, choices=['gpn24'], default=None,
                        help="Evaluation protocol to use. 'gpn24' runs GPN ICLR'24 protocol (20 seeds x 10 folds)")
    parser.add_argument("--log_dir", type=str, default="./gpn24_logs",
                        help="Directory to save GPN24 protocol logs (default: ./gpn24_logs)")

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

    config_dict['no_cache'] = args.no_cache

    run_lacore_pool_experiment(**config_dict)

if __name__ == "__main__":
    main()

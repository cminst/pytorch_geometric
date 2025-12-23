from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_dense_adj

try:
    from torch_scatter import scatter_mean
except Exception:
    scatter_mean = None

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


def split_train_val(ds, val_ratio: float, seed: int):
    """Split a dataset (with cached transforms) into train/val subsets deterministically.

    Args:
        ds: PyTorch Geometric dataset (assumed already pre-transformed and cached)
        val_ratio: fraction [0,1] of data to reserve for validation
        seed: random seed for reproducibility (must be >=0)

    Returns:
        Tuple[Subset, Subset]: train_subset, val_subset
    """
    n = len(ds)
    n_val = int(round(n * val_ratio))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)

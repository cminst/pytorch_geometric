"""Train/evaluate PointWavelet baselines on ShapeNet part segmentation.

This script mirrors umc_pointwavelet.py but targets ShapeNetPart segmentation
using ShapeNetPatched (updated download URL).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch_geometric.transforms import Compose, NormalizeScale

from pointwavelet import PointWaveletPartSeg, PointWaveletPartSegConfig
from utils.custom_datasets import ShapeNetPatched
from utils.transforms import PointJitter, PointMLPAffine


class SamplePointsWithReplacement:
    """Uniformly resample points with replacement to a fixed size.

    Keeps pos/x/y aligned for part segmentation.
    """

    def __init__(self, num_points: int) -> None:
        self.num_points = int(num_points)

    def __call__(self, data):
        pos = data.pos
        n_points = int(pos.size(0))
        if n_points <= 0:
            return data
        idx = torch.randint(0, n_points, (self.num_points,), device=pos.device)
        data.pos = pos[idx]
        if hasattr(data, "x") and data.x is not None and data.x.size(0) == n_points:
            data.x = data.x[idx]
        if hasattr(data, "y") and data.y is not None and data.y.numel() == n_points:
            data.y = data.y[idx]
        for key in ["edge_index", "edge_attr", "phi", "phi_evals", "deg", "batch"]:
            if hasattr(data, key):
                delattr(data, key)
        data.num_nodes = self.num_points
        return data


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _wandb_run_name(wf_learnable: bool, method: str) -> str:
    base = "PointWavelet-L" if wf_learnable else "PointWavelet"
    if method == "pointwavelet_umc":
        return f"{base}-UMC"
    return base


def _save_checkpoint(path: str, model: PointWaveletPartSeg, meta: dict) -> None:
    payload = {
        "model_state": model.state_dict(),
        "cfg": asdict(model.cfg),
        "meta": meta,
    }
    torch.save(payload, path)


def _init_wandb(enabled: bool, args: argparse.Namespace, method: str, seed: int):
    if not enabled:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb is enabled but not installed. Install it with `pip install wandb` "
            "or disable logging via `--no-wandb`."
        ) from exc

    run_name = _wandb_run_name(bool(args.wf_learnable), method)
    config = {**vars(args), "method": method, "seed": seed}
    return wandb.init(
        project="UMC",
        name=run_name,
        config=config,
        job_type=method,
        tags=[method],
        reinit=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _batch_pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)
    return (a * b).sum(dim=1) / denom


@torch.no_grad()
def _collect_umc_stats(
    model: PointWaveletPartSeg,
    loader: DataLoader,
    device: torch.device,
    num_categories: int,
    max_batches: int = 10,
) -> Dict[str, float]:
    layers = {
        "sa1": model.sa1.wf,
        "sa2": model.sa2.wf,
        "sa3": model.sa3.wf,
        "sa4": model.sa4.wf,
    }
    if not any(getattr(layer, "umc", None) is not None for layer in layers.values()):
        return {}

    acc = {
        name: {
            "w_var": 0.0,
            "w_std": 0.0,
            "corr_w_meandist": 0.0,
            "corr_w_invmeandist": 0.0,
            "w_min": float("inf"),
            "w_max": float("-inf"),
            "count": 0,
        }
        for name in layers
    }
    model.eval()
    for b_idx, (xyz, feats, _, cat) in enumerate(loader):
        if b_idx >= max_batches:
            break
        xyz = xyz.to(device)
        feats = feats.to(device) if feats is not None else None
        cat = cat.to(device)
        cat_onehot = _category_onehot(cat, num_categories=num_categories) if model.cfg.use_category_label else None
        _ = model(xyz, feats, cat_onehot)
        for name, wf in layers.items():
            umc = getattr(wf, "last_umc", None)
            if not umc:
                continue
            w = umc["w"]
            md = umc["mean_dist"]
            if w.numel() == 0:
                continue
            w_var = w.var(dim=1, unbiased=False)
            w_std = w.std(dim=1, unbiased=False)
            w_min = w.min(dim=1).values
            w_max = w.max(dim=1).values
            corr_md = _batch_pearson_corr(w, md)
            corr_inv = _batch_pearson_corr(w, 1.0 / (md + 1e-6))

            acc[name]["w_var"] += float(w_var.sum().item())
            acc[name]["w_std"] += float(w_std.sum().item())
            acc[name]["corr_w_meandist"] += float(corr_md.sum().item())
            acc[name]["corr_w_invmeandist"] += float(corr_inv.sum().item())
            acc[name]["w_min"] = float(min(acc[name]["w_min"], w_min.min().item()))
            acc[name]["w_max"] = float(max(acc[name]["w_max"], w_max.max().item()))
            acc[name]["count"] += int(w.shape[0])

    out: Dict[str, float] = {}
    total = 0
    total_w_var = 0.0
    total_w_std = 0.0
    total_corr_md = 0.0
    total_corr_inv = 0.0
    total_w_min = float("inf")
    total_w_max = float("-inf")
    for name, stats in acc.items():
        count = stats["count"]
        if count <= 0:
            continue
        out[f"{name}_w_var"] = stats["w_var"] / count
        out[f"{name}_w_std"] = stats["w_std"] / count
        out[f"{name}_corr_w_meandist"] = stats["corr_w_meandist"] / count
        out[f"{name}_corr_w_invmeandist"] = stats["corr_w_invmeandist"] / count
        out[f"{name}_w_min"] = stats["w_min"]
        out[f"{name}_w_max"] = stats["w_max"]
        total += count
        total_w_var += stats["w_var"]
        total_w_std += stats["w_std"]
        total_corr_md += stats["corr_w_meandist"]
        total_corr_inv += stats["corr_w_invmeandist"]
        total_w_min = min(total_w_min, stats["w_min"])
        total_w_max = max(total_w_max, stats["w_max"])

    if total > 0:
        out["umc_w_var"] = total_w_var / total
        out["umc_w_std"] = total_w_std / total
        out["umc_corr_w_meandist"] = total_corr_md / total
        out["umc_corr_w_invmeandist"] = total_corr_inv / total
        out["umc_w_min"] = total_w_min
        out["umc_w_max"] = total_w_max
    return out


def _umc_corr_reg_loss(model: PointWaveletPartSeg) -> torch.Tensor:
    layers = {
        "sa1": model.sa1.wf,
        "sa2": model.sa2.wf,
        "sa3": model.sa3.wf,
        "sa4": model.sa4.wf,
    }
    corr_terms: List[torch.Tensor] = []
    for wf in layers.values():
        umc = getattr(wf, "last_umc_raw", None)
        if not umc:
            continue
        w = umc.get("w")
        md = umc.get("mean_dist")
        if w is None or md is None or w.numel() == 0:
            continue
        corr = _batch_pearson_corr(w, md)
        corr_terms.append(corr.mean())
    if not corr_terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(corr_terms).mean()


def _umc_range_reg_loss(
    model: PointWaveletPartSeg,
    low: float = -3.0,
    high: float = 3.0,
) -> torch.Tensor:
    layers = {
        "sa1": model.sa1.wf,
        "sa2": model.sa2.wf,
        "sa3": model.sa3.wf,
        "sa4": model.sa4.wf,
    }
    terms: List[torch.Tensor] = []
    for wf in layers.values():
        umc = getattr(wf, "last_umc_raw", None)
        if not umc:
            continue
        w = umc.get("w")
        if w is None or w.numel() == 0:
            continue
        below = torch.relu(low - w)
        above = torch.relu(w - high)
        terms.append((below * below + above * above).mean())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).mean()


def _category_onehot(category: torch.Tensor, num_categories: int) -> torch.Tensor:
    return F.one_hot(category, num_classes=num_categories).float()


def collate_segmentation(items) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    xyz = torch.stack([it.pos for it in items], dim=0)
    feats = None
    if hasattr(items[0], "x") and items[0].x is not None:
        feats = torch.stack([it.x for it in items], dim=0)
    y = torch.stack([it.y.view(-1) for it in items], dim=0).long()
    cats = []
    for it in items:
        cat = getattr(it, "category", None)
        if torch.is_tensor(cat):
            cats.append(int(cat.view(-1)[0].item()))
        else:
            cats.append(int(cat))
    category = torch.tensor(cats, dtype=torch.long)
    return xyz, feats, y, category


def _shape_iou(pred: torch.Tensor, target: torch.Tensor, part_labels: List[int]) -> float:
    ious = []
    for part in part_labels:
        pred_mask = pred == part
        target_mask = target == part
        if not pred_mask.any() and not target_mask.any():
            ious.append(1.0)
            continue
        inter = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(float(inter) / float(union))
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def eval_metrics(
    model: PointWaveletPartSeg,
    loader: DataLoader,
    device: torch.device,
    categories: List[str],
    seg_classes: Dict[str, List[int]],
) -> dict:
    model.eval()
    correct = 0
    total = 0
    shape_ious: List[float] = []
    cat_ious: Dict[int, List[float]] = {i: [] for i in range(len(categories))}

    for xyz, feats, y, cat in loader:
        xyz = xyz.to(device)
        feats = feats.to(device) if feats is not None else None
        y = y.to(device)
        cat = cat.to(device)
        cat_onehot = _category_onehot(cat, num_categories=len(categories)) if model.cfg.use_category_label else None
        logits = model(xyz, feats, cat_onehot)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

        for i in range(pred.size(0)):
            cat_idx = int(cat[i].item())
            cat_name = categories[cat_idx]
            parts = seg_classes[cat_name]
            iou = _shape_iou(pred[i], y[i], parts)
            shape_ious.append(iou)
            cat_ious[cat_idx].append(iou)

    inst_miou = float(np.mean(shape_ious)) if shape_ious else 0.0
    class_miou = float(
        np.mean([np.mean(vals) for vals in cat_ious.values() if vals])
    ) if cat_ious else 0.0
    oa = float(correct) / float(max(total, 1))
    return {
        "oa": oa,
        "inst_miou": inst_miou,
        "class_miou": class_miou,
    }


def _resolve_dataset_root(data_root: str, dataset_name: str) -> str:
    base = os.path.basename(os.path.normpath(data_root))
    if base == dataset_name:
        return data_root
    return os.path.join(data_root, dataset_name)


def build_datasets(
    root: str,
    num_points: int,
    force_reload: bool,
    categories: Optional[List[str]],
    include_normals: bool,
    train_mode: str,
    normalize: bool,
) -> Tuple[Dataset, Dataset]:
    pre_transform = NormalizeScale() if normalize else None
    train_transforms: List[object] = [SamplePointsWithReplacement(num_points)]
    if train_mode == "aug":
        train_transforms.extend([
            PointMLPAffine(),
            PointJitter(sigma=0.01, clip=0.02),
        ])
    train_transform = Compose(train_transforms)
    test_transform = Compose([SamplePointsWithReplacement(num_points)])

    train_ds = ShapeNetPatched(
        root=root,
        categories=categories,
        include_normals=include_normals,
        split="trainval",
        transform=train_transform,
        pre_transform=pre_transform,
        force_reload=force_reload,
    )
    test_ds = ShapeNetPatched(
        root=root,
        categories=categories,
        include_normals=include_normals,
        split="test",
        transform=test_transform,
        pre_transform=pre_transform,
        force_reload=force_reload,
    )
    return train_ds, test_ds


def build_model(
    use_umc: bool,
    wf_learnable: bool,
    umc_hidden: Tuple[int, int],
    umc_knn: int,
    umc_min_weight: float,
    umc_use_inverse: bool,
    num_parts: int,
    input_channels: int,
    num_categories: int,
    use_category_label: bool,
    wf_J: int = 5,
    wf_beta: float = 0.05,
) -> PointWaveletPartSeg:
    cfg = PointWaveletPartSegConfig(
        num_parts=num_parts,
        input_channels=input_channels,
        category_embed_dim=max(num_categories, 1),
        use_category_label=use_category_label,
        wf_learnable=wf_learnable,
        wf_beta=wf_beta,
        wf_J=wf_J,
        wf_depth=2,
        wf_heads=4,
        wf_sigma_mode="mean",
        wf_chunk_size=None,
        wf_force_math_attn=False,
        wf_use_umc=use_umc,
        wf_umc_hidden=umc_hidden,
        wf_umc_knn=umc_knn,
        wf_umc_min_weight=umc_min_weight,
        wf_umc_use_inverse=umc_use_inverse,
    )
    return PointWaveletPartSeg(cfg)


def train_one(
    model: PointWaveletPartSeg,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lr_step: int,
    lr_gamma: float,
    optim_name: str,
    scheduler_name: str,
    lr_min: Optional[float],
    sgd_momentum: float,
    amp: bool,
    categories: List[str],
    seg_classes: Dict[str, List[int]],
    wandb_run: Optional[object] = None,
    umc_stats_batches: int = 10,
    umc_corr_reg: float = 0.0,
    umc_range_reg: float = 0.0,
) -> dict:
    model = model.to(device)

    if optim_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "cosine":
        eta_min = lr_min if lr_min is not None else lr / 100.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(lr_step, 1), gamma=lr_gamma)

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"inst_miou": 0.0, "class_miou": 0.0, "oa": 0.0, "epoch": 0}
    has_umc = bool(getattr(model.cfg, "wf_use_umc", False))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for xyz, feats, y, cat in train_loader:
            xyz = xyz.to(device)
            feats = feats.to(device) if feats is not None else None
            y = y.to(device)
            cat = cat.to(device)
            cat_onehot = _category_onehot(cat, num_categories=len(categories)) if model.cfg.use_category_label else None

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(xyz, feats, cat_onehot)
                loss_seg = F.cross_entropy(logits.view(-1, model.cfg.num_parts), y.view(-1))
                reg_loss = model.regularization_loss()
                corr_reg_loss = _umc_corr_reg_loss(model) if (has_umc and umc_corr_reg > 0) else 0.0
                range_reg_loss = _umc_range_reg_loss(model) if (has_umc and umc_range_reg > 0) else 0.0
                loss = loss_seg + reg_loss + (umc_corr_reg * corr_reg_loss) + (umc_range_reg * range_reg_loss)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(y.size(0))
            total_loss += float(loss.item()) * bs
            total_items += bs

        scheduler.step()

        metrics = eval_metrics(model, test_loader, device=device, categories=categories, seg_classes=seg_classes)
        if metrics["class_miou"] > best["class_miou"]:
            best = {**metrics, "epoch": epoch}

        umc_stats = {}
        if has_umc and umc_stats_batches > 0:
            umc_stats = _collect_umc_stats(
                model,
                test_loader,
                device=device,
                num_categories=len(categories),
                max_batches=umc_stats_batches,
            )

        avg_loss = total_loss / max(total_items, 1)
        lr_now = scheduler.get_last_lr()[0]
        msg = (
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | "
            f"test_OA={metrics['oa']*100:.2f} | "
            f"test_InsIoU={metrics['inst_miou']*100:.2f} | "
            f"test_ClsIoU={metrics['class_miou']*100:.2f}"
        )
        if "umc_w_var" in umc_stats:
            msg += (
                f" | umc_w_var={umc_stats['umc_w_var']:.6f}"
                f" | umc_w_std={umc_stats.get('umc_w_std', 0.0):.6f}"
                f" | umc_w_min={umc_stats.get('umc_w_min', 0.0):.6f}"
                f" | umc_w_max={umc_stats.get('umc_w_max', 0.0):.6f}"
            )
        print(msg, flush=True)
        if wandb_run is not None:
            payload = {
                "loss": avg_loss,
                "lr": lr_now,
                "test_OA": metrics["oa"] * 100,
                "test_InsIoU": metrics["inst_miou"] * 100,
                "test_ClsIoU": metrics["class_miou"] * 100,
            }
            if umc_stats:
                payload.update(umc_stats)
            wandb_run.log(payload, step=epoch)

    final = eval_metrics(model, test_loader, device=device, categories=categories, seg_classes=seg_classes)
    return {"final": final, "best": best}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for datasets (default: data)",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="ShapeNet",
        help="Dataset name (default: ShapeNet)",
    )
    p.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated ShapeNet categories (default: all)",
    )
    p.add_argument("--force_reload", action="store_true", help="Force reprocessing the dataset")
    p.add_argument("--num_points", type=int, default=2048, help="Number of points to sample (default: 2048)")
    p.add_argument(
        "--train_mode",
        type=str,
        default="clean",
        choices=["clean", "aug"],
        help="Training transform: clean or aug (affine + jitter)",
    )
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False, help="Normalize point clouds (default: False)")
    p.add_argument("--no_normals", action="store_true", help="Disable normals (data.x) input")
    p.add_argument("--no_category_label", action="store_true", help="Disable class conditioning on category label")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoaders (default: 16)")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoaders (default: 4)")
    p.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer (default: adam)")
    p.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine"], help="LR scheduler (default: step)")
    p.add_argument("--lr_step", type=int, default=20, help="Learning rate decay step (default: 20)")
    p.add_argument("--lr_gamma", type=float, default=0.7, help="Learning rate decay gamma (default: 0.7)")
    p.add_argument("--lr_min", type=float, default=None, help="Cosine min LR (default: lr/100)")
    p.add_argument("--sgd_momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    p.add_argument("--amp", action="store_true", help="Use CUDA AMP (fp16) if available")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. '0,1,2,3' (default: 0)")
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging. (default: True)",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["vanilla", "umc", "both"],
        help="Which method(s) to run: vanilla (PointWavelet), umc, or both. (default: both)",
    )

    # PointWavelet variant
    p.add_argument(
        "--wf_learnable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PointWavelet-L (learnable spectral basis). (default: True)",
    )
    # UMC settings
    p.add_argument("--umc_hidden", type=str, default="128,32", help="Hidden widths for UMC MLP (default: 128,32)")
    p.add_argument("--umc_knn", type=int, default=20, help="k for UMC k-NN graph (default: 20)")
    p.add_argument("--umc_min_weight", type=float, default=1e-4, help="Minimum weight for UMC (default: 1e-4)")
    p.add_argument("--umc_no_inverse", action="store_true", help="Disable the W^{-1} factor in reconstruction")
    p.add_argument(
        "--umc_corr_reg",
        type=float,
        default=0.0,
        help="Weight for corr(w, mean_dist) regularizer; positive encourages negative correlation. (default: 0.0)",
    )
    p.add_argument(
        "--umc_range_reg",
        type=float,
        default=1e-3,
        help="Weight for UMC range regularizer to keep weights in [-3, 3]. (default: 1e-3)",
    )
    p.add_argument("--save_ckpt", action="store_true", help="Save model checkpoint after training")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints (default: checkpoints)")

    # UMC diagnostics
    p.add_argument("--umc_stats_batches", type=int, default=10, help="Batches to estimate UMC stats each epoch (default: 10)")

    p.add_argument("--out_csv", type=str, default="pointwavelet_umc_partseg_results.csv", help="Output CSV file name")
    return p.parse_args()


def _parse_int_pair(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two ints like '32,32', got: {s}")
    return int(parts[0]), int(parts[1])


def _parse_seeds(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_categories(s: str) -> Optional[List[str]]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None


def _resolve_methods(choice: str) -> List[Tuple[str, bool, str]]:
    if choice == "vanilla":
        return [("pointwavelet", False, "PointWavelet")]
    if choice == "umc":
        return [("pointwavelet_umc", True, "PointWavelet + UMC")]
    return [
        ("pointwavelet", False, "PointWavelet"),
        ("pointwavelet_umc", True, "PointWavelet + UMC"),
    ]


def main() -> None:
    args = parse_args()

    device = _device()
    print(f"Device: {device}", flush=True)

    umc_hidden = _parse_int_pair(args.umc_hidden)
    seeds = _parse_seeds(args.seeds)
    methods = _resolve_methods(args.methods)
    categories = _parse_categories(args.categories)

    dataset_root = _resolve_dataset_root(args.data_root, args.dataset_name)
    train_ds, test_ds = build_datasets(
        dataset_root,
        args.num_points,
        force_reload=bool(args.force_reload),
        categories=categories,
        include_normals=not bool(args.no_normals),
        train_mode=str(args.train_mode),
        normalize=bool(args.normalize),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_segmentation,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_segmentation,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    num_parts = int(train_ds.num_classes)
    category_list = list(train_ds.categories)
    seg_classes = train_ds.seg_classes
    num_categories = len(category_list)
    input_channels = 0 if bool(args.no_normals) else 3

    results: List[dict] = []
    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===", flush=True)

        for method_key, use_umc, label in methods:
            print(f"\n--- Method: {label} ---", flush=True)
            run = _init_wandb(bool(args.wandb), args, method=method_key, seed=seed)
            model = build_model(
                use_umc=use_umc,
                wf_learnable=bool(args.wf_learnable),
                umc_hidden=umc_hidden,
                umc_knn=args.umc_knn,
                umc_min_weight=args.umc_min_weight,
                umc_use_inverse=not bool(args.umc_no_inverse),
                num_parts=num_parts,
                input_channels=input_channels,
                num_categories=num_categories,
                use_category_label=not bool(args.no_category_label),
            )
            out = train_one(
                model,
                train_loader,
                test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lr_step=args.lr_step,
                lr_gamma=args.lr_gamma,
                optim_name=str(args.optimizer),
                scheduler_name=str(args.scheduler),
                lr_min=args.lr_min,
                sgd_momentum=float(args.sgd_momentum),
                amp=bool(args.amp),
                categories=category_list,
                seg_classes=seg_classes,
                wandb_run=run,
                umc_stats_batches=max(int(args.umc_stats_batches), 0),
                umc_corr_reg=float(args.umc_corr_reg),
                umc_range_reg=float(args.umc_range_reg),
            )
            if run is not None:
                run.finish()
            results.append(
                {
                    "seed": seed,
                    "method": method_key,
                    **out["final"],
                    "best_class_miou": out["best"]["class_miou"],
                    "best_epoch": out["best"]["epoch"],
                }
            )
            if args.save_ckpt:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                ckpt_name = f"{method_key}_ShapeNetPart_n{args.num_points}_seed{seed}.pt"
                ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
                meta = {
                    "seed": seed,
                    "method": method_key,
                    "num_points": int(args.num_points),
                    "num_parts": int(num_parts),
                    "num_categories": int(num_categories),
                    "wf_learnable": bool(args.wf_learnable),
                    "umc_hidden": list(umc_hidden),
                    "umc_knn": int(args.umc_knn),
                    "umc_min_weight": float(args.umc_min_weight),
                    "umc_use_inverse": not bool(args.umc_no_inverse),
                    "umc_range_reg": float(args.umc_range_reg),
                    "epochs": int(args.epochs),
                    "final": out["final"],
                    "best": out["best"],
                }
                _save_checkpoint(ckpt_path, model, meta)
                print(f"Saved checkpoint to {ckpt_path}", flush=True)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Write results
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["seed", "method", "oa", "inst_miou", "class_miou", "best_class_miou", "best_epoch"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nSaved results to {args.out_csv}", flush=True)
    # Also print a short summary
    for m, _, _ in methods:
        vals = [r["class_miou"] for r in results if r["method"] == m]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"{m}: class_mIoU mean={mean*100:.2f} std={std*100:.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()

import argparse
import copy
import os
import time
from typing import Callable, Optional

import pandas as pd
import torch
from torch_geometric.data import Batch
from torch.utils.data import Subset
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    Compose,
    KNNGraph,
    NormalizeScale,
    SamplePoints,
)
from umc_pointcloud_utils import (
    ComputePhiRWFromSym,
    CopyCategoryToY,
    ExtraCapacityControl,
    FixedDegreeClassifier,
    InvDegreeHeuristicClassifier,
    IrregularResample,
    MakeUndirected,
    MeanDistHeuristicClassifier,
    NoWeightClassifier,
    RandomIrregularResample,
    TrainConfig,
    UMCClassifier,
    eval_accuracy,
    eval_feature_stability,
    eval_weight_correlations,
    seed_everything,
    train_model,
)

from datasets import ScanObjectNN

# ----------------------------
# Dataset Registry
# ----------------------------

DATASET_INFO = {
    "ModelNet10": {
        "num_classes": 10
    },
    "ModelNet40": {
        "num_classes": 40
    },
    "ScanObjectNN": {
        "num_classes": 15
    },
}


def load_dataset(
    name: str,
    root: str,
    train: bool,
    pre_transform: Optional[Callable],
    transform: Optional[Callable],
    force_reload: bool = False,
):
    """Load a dataset by name.

    Args:
        name: One of 'ModelNet10', 'ModelNet40', 'ScanObjectNN'
        root: Root directory for data
        train: If True, load training split; else test split
        pre_transform: Pre-transform to apply (cached)
        transform: Transform to apply on access
        force_reload: Whether to reprocess the dataset

    Returns:
        Dataset instance
    """
    if name == "ModelNet10":
        return ModelNet(
            root=root,
            name="10",
            train=train,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    elif name == "ModelNet40":
        return ModelNet(
            root=root,
            name="40",
            train=train,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    elif name == "ScanObjectNN":
        # This loads the PB_T50_RS variant!!
        return ScanObjectNN(
            root=root,
            train=train,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(DATASET_INFO.keys())}")


def build_pre_transform(dataset_name: str, dense_points: int) -> Compose:
    """Build pre_transform pipeline (applied once and cached)."""
    transforms = []

    # ScanObjectNN already comes with 2048-point clouds; skip sampling there.
    if dataset_name != "ScanObjectNN":
        transforms.append(SamplePoints(dense_points))

    transforms.append(NormalizeScale())

    return Compose(transforms)


def build_train_transform_clean(
    num_points: int,
    knn_k: int,
    K: int
) -> Compose:
    """Build clean (uniform) training transform."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    return Compose(transforms)


def build_train_transform_aug(
    num_points: int,
    knn_k: int,
    K: int,
    max_bias: float
) -> Compose:
    """Build augmented (random bias) training transform."""
    transforms = [
        RandomIrregularResample(num_points=num_points, max_bias=max_bias),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    return Compose(transforms)


def build_test_transform_clean(
    num_points: int,
    knn_k: int,
    K: int
) -> Compose:
    """Build clean (uniform) test transform."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    return Compose(transforms)


def build_stress_transform(
    num_points: int,
    knn_k: int,
    K: int,
    bias_strength: float
) -> Compose:
    """Build stress test transform with specific bias level."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=bias_strength),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    return Compose(transforms)


# ----------------------------
# Helper Functions
# ----------------------------

def split_train_val(ds, val_ratio: float, seed: int):
    n = len(ds)
    n_val = int(round(n * val_ratio))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(ds, train_idx), Subset(ds, val_idx)


class CachedDataset:
    """Simple dataset wrapper that returns deep copies of preprocessed items."""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return copy.deepcopy(self.data_list[idx])


def cache_dataset(dataset, num_workers: int = 0):
    """Materialize a dataset once so we can reuse expensive transforms."""
    if len(dataset) == 0:
        return CachedDataset([])

    persistent_workers = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    cached = []
    for batch in loader:
        if isinstance(batch, Batch):
            cached.extend(batch.to_data_list())
        else:
            cached.append(batch)
    return CachedDataset(cached)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, seed: int, drop_last_train: bool = True, num_workers: int = 0):
    persistent_workers = num_workers > 0
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        generator=g,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def eval_stress_table(
    model,
    dataset_name: str,
    root: str,
    pre_transform,
    bias_levels,
    num_points: int,
    knn_k: int,
    K: int,
    device,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
):
    """Evaluate accuracy across bias levels with deterministic corruption per (seed, bias)."""
    out = {}
    persistent_workers = num_workers > 0
    for bias in bias_levels:
        # deterministic per bias so all models see the "same" stress corruption
        seed_everything(seed + int(1000 * bias))

        stress_transform = build_stress_transform(
            num_points=num_points,
            knn_k=knn_k,
            K=K,
            bias_strength=float(bias)
        )

        ds = load_dataset(
            name=dataset_name,
            root=root,
            train=False,
            pre_transform=pre_transform,
            transform=stress_transform,
            force_reload=False,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        acc = eval_accuracy(model, loader, device)
        out[f"stress_bias_{bias:.1f}"] = float(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="ModelNet10",
                    choices=list(DATASET_INFO.keys()),
                    help="Dataset: ModelNet10, ModelNet40, or ScanObjectNN")
    ap.add_argument("--num_points", type=int, default=512)
    ap.add_argument("--dense_points", type=int, default=2048)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--max_bias_train", type=float, default=3.0, help="max bias for training augmentation mode")
    ap.add_argument("--train_mode", type=str, default="both", choices=["clean", "aug", "both"])

    ap.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--bias_levels", type=str, default="0,1,2,3,4", help="comma-separated bias levels for stress test")

    ap.add_argument("--lambda_ortho_grid", type=str, default="0,0.001,0.01,0.1,1,10")
    ap.add_argument("--lambda_w_reg", type=float, default=0.0, help="(w-1)^2 reg for learned-weight models")

    ap.add_argument("--results_csv", type=str, default="umc_sweep_results.csv")
    ap.add_argument("--stability_bias", type=float, default=3.0)
    ap.add_argument("--stability_items", type=int, default=200)

    ap.add_argument("--methods", type=str, default="naive,deg,invdeg,meandist,cap,umc", help="comma-separated methods to run")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for train/eval (0 keeps everything on the main process).")
    ap.add_argument("--torch_threads", type=int, default=None, help="Optional cap on torch threads to avoid overusing CPU cores during preprocessing.")
    ap.add_argument("--no_cache_eval", dest="cache_eval", action="store_false", help="Disable caching val/test sets after the first preprocessing pass.")
    ap.set_defaults(cache_eval=True)

    args = ap.parse_args()

    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, args.torch_threads // 2))
        os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.dataset)

    # Get dataset info
    dataset_info = DATASET_INFO[args.dataset]
    num_classes = dataset_info["num_classes"]

    print(f"  num_classes: {num_classes}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    bias_levels = [float(b) for b in args.bias_levels.split(",") if b.strip() != ""]
    lambda_ortho_grid = [float(x) for x in args.lambda_ortho_grid.split(",") if x.strip() != ""]

    methods_to_run = {s.strip() for s in args.methods.split(",") if s.strip()}

    # ------------------------------------------------------------
    # Pre-transform: cache a dense (aligned) point set once.
    # ------------------------------------------------------------
    pre_transform_dense = build_pre_transform(
        dataset_name=args.dataset,
        dense_points=args.dense_points
    )

    # ------------------------------------------------------------
    # Build transform pipelines
    # ------------------------------------------------------------
    # TRAIN: clean (uniform) or augmented (random bias)
    train_transform_clean = build_train_transform_clean(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K
    )

    train_transform_aug = build_train_transform_aug(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        max_bias=args.max_bias_train
    )

    # TEST: clean (uniform)
    test_transform_clean = build_test_transform_clean(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K
    )

    # Bias transforms for stability test
    transform_bias0 = build_stress_transform(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        bias_strength=0.0
    )
    transform_biasS = build_stress_transform(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        bias_strength=float(args.stability_bias)
    )

    print("Loading datasets (may process once on first run) - ", end="", flush=True)
    if not args.root:
        args.root = f'../data/{args.dataset}'
        print(f"No root directory specified, using {args.root}", flush=True)
    else:
        print(args.root, flush=True)

    # Also keep a "dense raw" test set for stability analysis (no transform here)
    test_dense_ds = load_dataset(
        name=args.dataset,
        root=args.root,
        train=False,
        pre_transform=pre_transform_dense,
        transform=None,
        force_reload=False,
    )

    # ------------------------------------------------------------
    # Experiment specs
    # ------------------------------------------------------------
    def make_model(tag: str, lam_ortho: float):
        # baselines
        if tag == "naive":
            return NoWeightClassifier(K=args.K, num_classes=num_classes).to(device)
        if tag == "deg":
            return FixedDegreeClassifier(K=args.K, num_classes=num_classes).to(device)
        if tag == "invdeg":
            return InvDegreeHeuristicClassifier(K=args.K, num_classes=num_classes).to(device)
        if tag == "meandist":
            return MeanDistHeuristicClassifier(K=args.K, num_classes=num_classes).to(device)
        # learned
        if tag == "umc":
            return UMCClassifier(K=args.K, num_classes=num_classes, use_pos=True, use_density=True).to(device)
        if tag == "cap":
            return ExtraCapacityControl(K=args.K, num_classes=num_classes, use_pos=True, use_density=True).to(device)
        raise ValueError(tag)

    # We'll run:
    # - fixed baselines once
    # - extra capacity once
    # - UMC for each lambda_ortho in grid
    base_variants = [
        ("naive", None),
        ("deg", None),
        ("invdeg", None),
        ("meandist", None),
        ("cap", 0.0),
    ]
    umc_variants = [("umc", lam) for lam in lambda_ortho_grid]
    base_variants = [v for v in base_variants if v[0] in methods_to_run]
    umc_variants = [v for v in umc_variants if v[0] in methods_to_run]

    # ------------------------------------------------------------
    # Protocols: clean vs aug training
    # ------------------------------------------------------------
    train_modes = []
    if args.train_mode in ["clean", "both"]:
        train_modes.append(("train_clean", train_transform_clean))
    if args.train_mode in ["aug", "both"]:
        train_modes.append(("train_aug", train_transform_aug))

    rows = []

    for (mode_name, train_transform) in train_modes:
        print("\n=== TRAIN MODE:", mode_name, "===\n")

        # Build the dataset objects for this mode
        train_ds_full = load_dataset(
            name=args.dataset,
            root=args.root,
            train=True,
            pre_transform=pre_transform_dense,
            transform=train_transform,
            force_reload=False,
        )
        test_ds = load_dataset(
            name=args.dataset,
            root=args.root,
            train=False,
            pre_transform=pre_transform_dense,
            transform=test_transform_clean,
            force_reload=False,
        )

        for seed in seeds:
            seed_everything(seed)
            print(f"\n--- Seed {seed} ---")

            train_ds, val_ds = split_train_val(train_ds_full, val_ratio=args.val_ratio, seed=seed)

            if args.cache_eval:
                print("  Caching val/test once to reuse graph preprocessing...", flush=True)
                val_ds_eff = cache_dataset(val_ds, num_workers=args.num_workers)
                test_ds_eff = cache_dataset(test_ds, num_workers=args.num_workers)
            else:
                val_ds_eff = val_ds
                test_ds_eff = test_ds

            train_loader, val_loader, test_loader = make_loaders(
                train_ds,
                val_ds_eff,
                test_ds_eff,
                batch_size=args.batch_size,
                seed=seed,
                drop_last_train=True,
                num_workers=args.num_workers,
            )

            # Run baselines + capacity control
            for tag, lam in base_variants:
                model = make_model(tag, lam_ortho=0.0)

                cfg = TrainConfig(
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lambda_ortho=0.0,
                    lambda_w_reg=(args.lambda_w_reg if tag in ["umc", "cap"] else 0.0),
                )

                t0 = time.time()
                metrics = train_model(model, train_loader, val_loader, test_loader, device, K=args.K, cfg=cfg)
                dt = time.time() - t0

                # stress
                stress = eval_stress_table(
                    model=model,
                    dataset_name=args.dataset,
                    root=args.root,
                    pre_transform=pre_transform_dense,
                    bias_levels=bias_levels,
                    num_points=args.num_points,
                    knn_k=args.knn_k,
                    K=args.K,
                    device=device,
                    batch_size=args.batch_size,
                    seed=seed,
                    num_workers=args.num_workers,
                )

                # correlations (for all; meaningful mainly for learned models)
                corr_loader = DataLoader(
                    test_ds_eff,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers,
                    persistent_workers=args.num_workers > 0,
                )
                corrs = eval_weight_correlations(model, corr_loader, device, max_batches=20)

                # feature stability (bias 0 vs bias S)
                dense_items = [test_dense_ds[i] for i in range(min(len(test_dense_ds), args.stability_items))]
                stab = eval_feature_stability(
                    model=model,
                    data_list_dense=dense_items,
                    transform_bias0=transform_bias0,
                    transform_bias1=transform_biasS,
                    device=device,
                    max_items=args.stability_items,
                )

                row = {
                    "dataset": args.dataset,
                    "train_mode": mode_name,
                    "seed": seed,
                    "variant": tag,
                    "lambda_ortho": 0.0,
                    "time_sec": dt,
                    **metrics,
                    **stress,
                    **corrs,
                    "feature_stability_cos": stab,
                }
                rows.append(row)
                print(f"[{mode_name} | seed={seed} | {tag}] test_acc={metrics['test_acc']*100:.2f}%  "
                      f"stress@{bias_levels[-1]}={stress[f'stress_bias_{bias_levels[-1]:.1f}']*100:.2f}%")

            # UMC lambda sweep
            for tag, lam in umc_variants:
                model = make_model(tag, lam_ortho=lam)

                cfg = TrainConfig(
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    lambda_ortho=lam,
                    lambda_w_reg=args.lambda_w_reg,
                )

                t0 = time.time()
                metrics = train_model(model, train_loader, val_loader, test_loader, device, K=args.K, cfg=cfg)
                dt = time.time() - t0

                stress = eval_stress_table(
                    model=model,
                    dataset_name=args.dataset,
                    root=args.root,
                    pre_transform=pre_transform_dense,
                    bias_levels=bias_levels,
                    num_points=args.num_points,
                    knn_k=args.knn_k,
                    K=args.K,
                    device=device,
                    batch_size=args.batch_size,
                    seed=seed,
                    num_workers=args.num_workers,
                )

                corr_loader = DataLoader(
                    test_ds_eff,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers,
                    persistent_workers=args.num_workers > 0,
                )
                corrs = eval_weight_correlations(model, corr_loader, device, max_batches=20)

                dense_items = [test_dense_ds[i] for i in range(min(len(test_dense_ds), args.stability_items))]
                stab = eval_feature_stability(
                    model=model,
                    data_list_dense=dense_items,
                    transform_bias0=transform_bias0,
                    transform_bias1=transform_biasS,
                    device=device,
                    max_items=args.stability_items,
                )

                row = {
                    "dataset": args.dataset,
                    "train_mode": mode_name,
                    "seed": seed,
                    "variant": f"umc",
                    "lambda_ortho": lam,
                    "time_sec": dt,
                    **metrics,
                    **stress,
                    **corrs,
                    "feature_stability_cos": stab,
                }
                rows.append(row)
                print(f"[{mode_name} | seed={seed} | UMC lam={lam}] test_acc={metrics['test_acc']*100:.2f}%  "
                      f"stress@{bias_levels[-1]}={stress[f'stress_bias_{bias_levels[-1]:.1f}']*100:.2f}%")

    # ------------------------------------------------------------
    # Save + summarize
    # ------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(args.results_csv, index=False)
    print("\nSaved:", args.results_csv)

    # Print a quick mean±std summary over seeds per (dataset, train_mode, variant, lambda_ortho)
    group_cols = ["dataset", "train_mode", "variant", "lambda_ortho"]
    summary = df.groupby(group_cols).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        stress_max_mean=(f"stress_bias_{bias_levels[-1]:.1f}", "mean"),
        stress_max_std=(f"stress_bias_{bias_levels[-1]:.1f}", "std"),
        stab_mean=("feature_stability_cos", "mean"),
        stab_std=("feature_stability_cos", "std"),
    ).reset_index()

    # nicer %
    summary["test_acc_mean"] *= 100
    summary["test_acc_std"] *= 100
    summary["stress_max_mean"] *= 100
    summary["stress_max_std"] *= 100

    print("\n=== Summary (mean±std over seeds) ===")
    print(summary.sort_values(["dataset", "train_mode", "variant", "lambda_ortho"]).to_string(index=False))


if __name__ == "__main__":
    main()

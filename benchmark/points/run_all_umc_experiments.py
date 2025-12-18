import os
import time
import copy
import argparse
import pandas as pd
from typing import Optional, Callable

import torch
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, KNNGraph
from torch.utils.data import Subset

from umc_modelnet_utils import (
    seed_everything,
    MakeUndirected,
    CopyCategoryToY,
    IrregularResample,
    RandomIrregularResample,
    ComputePhiRWFromSym,
    TrainConfig,
    train_model,
    eval_accuracy,
    eval_weight_correlations,
    eval_feature_stability,
    NoWeightClassifier,
    FixedDegreeClassifier,
    InvDegreeHeuristicClassifier,
    MeanDistHeuristicClassifier,
    UMCClassifier,
    ExtraCapacityControl,
)


# ----------------------------
# Dataset Registry
# ----------------------------

DATASET_INFO = {
    "ModelNet10": {
        "num_classes": 10,
        "needs_category_to_y": False,
    },
    "ModelNet40": {
        "num_classes": 40,
        "needs_category_to_y": False,
    },
    "ShapeNet": {
        "num_classes": 16,  # 16 shape categories
        "needs_category_to_y": True,
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
    """
    Load a dataset by name.

    Args:
        name: One of 'ModelNet10', 'ModelNet40', 'ShapeNet'
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
    elif name == "ShapeNet":
        # ShapeNet uses split='trainval'/'test' instead of train=True/False
        split = "trainval" if train else "test"

        # Download ShapeNet dataset from HF since link is broken
        if not os.path.exists(root):
            os.makedirs(root)

        zip_path = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip")
        if not os.path.exists(zip_path) or not os.listdir(os.path.dirname(zip_path)):
            import urllib.request
            url = "https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
            print(f"Downloading ShapeNet dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")

        return ShapeNet(
            root=root,
            categories=None,  # Load all categories
            include_normals=False,  # We only use positions
            split=split,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(DATASET_INFO.keys())}")


def build_pre_transform(dense_points: int, needs_category_to_y: bool) -> Compose:
    """Build pre_transform pipeline (applied once and cached)."""
    transforms = [
        SamplePoints(dense_points),
        NormalizeScale(),
    ]
    if needs_category_to_y:
        transforms.append(CopyCategoryToY())
    return Compose(transforms)


def build_train_transform_clean(
    num_points: int,
    knn_k: int,
    K: int,
    needs_category_to_y: bool,
) -> Compose:
    """Build clean (uniform) training transform."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    if needs_category_to_y:
        transforms.append(CopyCategoryToY())
    return Compose(transforms)


def build_train_transform_aug(
    num_points: int,
    knn_k: int,
    K: int,
    max_bias: float,
    needs_category_to_y: bool,
) -> Compose:
    """Build augmented (random bias) training transform."""
    transforms = [
        RandomIrregularResample(num_points=num_points, max_bias=max_bias),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    if needs_category_to_y:
        transforms.append(CopyCategoryToY())
    return Compose(transforms)


def build_test_transform_clean(
    num_points: int,
    knn_k: int,
    K: int,
    needs_category_to_y: bool,
) -> Compose:
    """Build clean (uniform) test transform."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    if needs_category_to_y:
        transforms.append(CopyCategoryToY())
    return Compose(transforms)


def build_stress_transform(
    num_points: int,
    knn_k: int,
    K: int,
    bias_strength: float,
    needs_category_to_y: bool,
) -> Compose:
    """Build stress test transform with specific bias level."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=bias_strength),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=K, store_aux=True),
    ]
    if needs_category_to_y:
        transforms.append(CopyCategoryToY())
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


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, seed: int, drop_last_train: bool = True):
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last_train, generator=g, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
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
    needs_category_to_y: bool,
):
    """
    Evaluate accuracy across bias levels with deterministic corruption per (seed, bias).
    """
    out = {}
    for bias in bias_levels:
        # deterministic per bias so all models see the "same" stress corruption
        seed_everything(seed + int(1000 * bias))

        stress_transform = build_stress_transform(
            num_points=num_points,
            knn_k=knn_k,
            K=K,
            bias_strength=float(bias),
            needs_category_to_y=needs_category_to_y,
        )

        ds = load_dataset(
            name=dataset_name,
            root=root,
            train=False,
            pre_transform=pre_transform,
            transform=stress_transform,
            force_reload=False,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        acc = eval_accuracy(model, loader, device)
        out[f"stress_bias_{bias:.1f}"] = float(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--dataset", type=str, default="ModelNet10",
                    choices=list(DATASET_INFO.keys()),
                    help="Dataset: ModelNet10, ModelNet40, or ShapeNet")
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

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.dataset)

    # Get dataset info
    dataset_info = DATASET_INFO[args.dataset]
    num_classes = dataset_info["num_classes"]
    needs_category_to_y = dataset_info["needs_category_to_y"]

    print(f"  num_classes: {num_classes}")
    print(f"  needs_category_to_y: {needs_category_to_y}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    bias_levels = [float(b) for b in args.bias_levels.split(",") if b.strip() != ""]
    lambda_ortho_grid = [float(x) for x in args.lambda_ortho_grid.split(",") if x.strip() != ""]

    methods_to_run = {s.strip() for s in args.methods.split(",") if s.strip()}

    # ------------------------------------------------------------
    # Pre-transform: cache a dense (aligned) point set once.
    # ------------------------------------------------------------
    pre_transform_dense = build_pre_transform(
        dense_points=args.dense_points,
        needs_category_to_y=needs_category_to_y,
    )

    # ------------------------------------------------------------
    # Build transform pipelines
    # ------------------------------------------------------------
    # TRAIN: clean (uniform) or augmented (random bias)
    train_transform_clean = build_train_transform_clean(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        needs_category_to_y=needs_category_to_y,
    )

    train_transform_aug = build_train_transform_aug(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        max_bias=args.max_bias_train,
        needs_category_to_y=needs_category_to_y,
    )

    # TEST: clean (uniform)
    test_transform_clean = build_test_transform_clean(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        needs_category_to_y=needs_category_to_y,
    )

    # Bias transforms for stability test
    transform_bias0 = build_stress_transform(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        bias_strength=0.0,
        needs_category_to_y=needs_category_to_y,
    )
    transform_biasS = build_stress_transform(
        num_points=args.num_points,
        knn_k=args.knn_k,
        K=args.K,
        bias_strength=float(args.stability_bias),
        needs_category_to_y=needs_category_to_y,
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
            train_loader, val_loader, test_loader = make_loaders(
                train_ds, val_ds, test_ds, batch_size=args.batch_size, seed=seed, drop_last_train=True
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
                    needs_category_to_y=needs_category_to_y,
                )

                # correlations (for all; meaningful mainly for learned models)
                corr_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
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
                    needs_category_to_y=needs_category_to_y,
                )

                corr_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
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

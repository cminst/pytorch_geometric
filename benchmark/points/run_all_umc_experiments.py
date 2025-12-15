import os
import time
import copy
import argparse
import pandas as pd

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, SamplePoints, NormalizeScale, KNNGraph
from torch.utils.data import Subset

from umc_modelnet_utils import (
    seed_everything,
    MakeUndirected,
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
    root: str,
    name: str,
    pre_transform_dense,
    bias_levels,
    num_points: int,
    knn_k: int,
    K: int,
    device,
    batch_size: int,
    seed: int,
):
    """
    Evaluate accuracy across bias levels with deterministic corruption per (seed, bias).
    """
    out = {}
    for bias in bias_levels:
        # deterministic per bias so all models see the "same" stress corruption
        seed_everything(seed + int(1000 * bias))

        stress_transform = Compose([
            IrregularResample(num_points=num_points, bias_strength=float(bias)),
            NormalizeScale(),
            KNNGraph(k=knn_k),
            MakeUndirected(),
            ComputePhiRWFromSym(K=K, store_aux=True),
        ])

        ds = ModelNet(
            root=root,
            name=str(name),
            train=False,
            pre_transform=pre_transform_dense,
            transform=stress_transform,
            force_reload=False,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        acc = eval_accuracy(model, loader, device)
        out[f"stress_bias_{bias:.1f}"] = float(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="../data/ModelNet_UMC")
    ap.add_argument("--name", type=str, default="10", help="ModelNet class count: '10' or '40'")
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

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    bias_levels = [float(b) for b in args.bias_levels.split(",") if b.strip() != ""]
    lambda_ortho_grid = [float(x) for x in args.lambda_ortho_grid.split(",") if x.strip() != ""]

    num_classes = int(args.name)

    # ------------------------------------------------------------
    # Pre-transform: cache a dense (aligned) point set once.
    # ------------------------------------------------------------
    pre_transform_dense = Compose([
        SamplePoints(args.dense_points),
        NormalizeScale(),
    ])

    # ------------------------------------------------------------
    # Build transform pipelines
    # ------------------------------------------------------------
    # TRAIN: clean (uniform) or augmented (random bias)
    train_transform_clean = Compose([
        IrregularResample(num_points=args.num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=args.knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=args.K, store_aux=True),
    ])

    train_transform_aug = Compose([
        RandomIrregularResample(num_points=args.num_points, max_bias=args.max_bias_train),
        NormalizeScale(),
        KNNGraph(k=args.knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=args.K, store_aux=True),
    ])

    # TEST: clean (uniform)
    test_transform_clean = Compose([
        IrregularResample(num_points=args.num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=args.knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=args.K, store_aux=True),
    ])

    # Also keep a "dense raw" test set for stability analysis (no transform here)
    print("Loading datasets (may process once on first run)...")
    test_dense_ds = ModelNet(
        root=args.root,
        name=str(args.name),
        train=False,
        pre_transform=pre_transform_dense,
        transform=None,
        force_reload=False,
    )

    # Bias transforms for stability test applied manually on dense items
    transform_bias0 = Compose([
        IrregularResample(num_points=args.num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=args.knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=args.K, store_aux=True),
    ])
    transform_biasS = Compose([
        IrregularResample(num_points=args.num_points, bias_strength=float(args.stability_bias)),
        NormalizeScale(),
        KNNGraph(k=args.knn_k),
        MakeUndirected(),
        ComputePhiRWFromSym(K=args.K, store_aux=True),
    ])

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
        train_ds_full = ModelNet(
            root=args.root,
            name=str(args.name),
            train=True,
            pre_transform=pre_transform_dense,
            transform=train_transform,
            force_reload=False,
        )
        test_ds = ModelNet(
            root=args.root,
            name=str(args.name),
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
                    root=args.root,
                    name=args.name,
                    pre_transform_dense=pre_transform_dense,
                    bias_levels=bias_levels,
                    num_points=args.num_points,
                    knn_k=args.knn_k,
                    K=args.K,
                    device=device,
                    batch_size=args.batch_size,
                    seed=seed,
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
                    root=args.root,
                    name=args.name,
                    pre_transform_dense=pre_transform_dense,
                    bias_levels=bias_levels,
                    num_points=args.num_points,
                    knn_k=args.knn_k,
                    K=args.K,
                    device=device,
                    batch_size=args.batch_size,
                    seed=seed,
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

    # Print a quick mean±std summary over seeds per (train_mode, variant, lambda_ortho)
    group_cols = ["train_mode", "variant", "lambda_ortho"]
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
    print(summary.sort_values(["train_mode", "variant", "lambda_ortho"]).to_string(index=False))


if __name__ == "__main__":
    main()

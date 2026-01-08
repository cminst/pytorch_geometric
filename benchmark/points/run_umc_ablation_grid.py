import argparse
import os
import time
from typing import Optional

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from utils.datasets import (
    DATASET_INFO,
    build_pre_transform,
    build_stress_transform,
    build_test_transform_clean,
    build_train_transform_aug,
    build_train_transform_clean,
    cache_dataset,
    load_dataset,
    make_loaders,
)
from utils.models import UMCClassifier, UMCPointEncoderClassifier, format_duration, seed_everything
from utils.training import (
    TrainConfig,
    eval_accuracy,
    eval_feature_stability,
    eval_weight_correlations,
    split_train_val,
    train_model,
)

# ----------------------------
# UMC ablation configs
# ----------------------------

UMC_ABLATIONS = {
    "full": dict(use_pos=True, use_md=True, use_log_md=True, use_log_deg=True, use_deg=False),
    "full_point_encoder": dict(use_pos=True, use_md=True, use_log_md=True, use_log_deg=True, use_deg=False),
    "no_coords": dict(use_pos=False, use_md=True, use_log_md=True, use_log_deg=True, use_deg=False),
    "md_only": dict(use_pos=False, use_md=True, use_log_md=True, use_log_deg=False, use_deg=False),
    "deg_only": dict(use_pos=False, use_md=False, use_log_md=False, use_log_deg=True, use_deg=False),
}


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
    phi_device: Optional[str] = None,
    include_phi: bool = True,
):
    """Evaluate accuracy across bias levels with deterministic corruption per (seed, bias)."""
    out = {}
    persistent_workers = num_workers > 0
    for bias in bias_levels:
        seed_everything(seed + int(1000 * bias))

        stress_transform = build_stress_transform(
            num_points=num_points,
            knn_k=knn_k,
            K=K,
            bias_strength=float(bias),
            include_phi=include_phi,
            phi_device=phi_device,
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


def _parse_list(arg: str) -> list[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument(
        "--datasets",
        type=str,
        default="ModelNet10,ModelNet40,ScanObjectNN",
        help="comma-separated datasets to run",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASET_INFO.keys()),
        help="Optional single dataset (overrides --datasets)",
    )
    ap.add_argument(
        "--umc_configs",
        type=str,
        default="full,no_coords,md_only,deg_only",
        help="comma-separated UMC ablation configs to run",
    )
    ap.add_argument(
        "--degree_features",
        type=str,
        default="log",
        choices=["log", "both"],
        help="For deg_only: 'log' -> [log_deg], 'both' -> [deg, log_deg]",
    )
    ap.add_argument("--num_points", type=int, default=512)
    ap.add_argument("--dense_points", type=int, default=2048)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument(
        "--phi_device",
        type=str,
        default=None,
        help="Device for torch.linalg.eigh when computing phi (e.g., 'cuda'). Defaults to the data tensor device.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--max_bias_train", type=float, default=3.0, help="max bias for training augmentation mode")
    ap.add_argument("--train_mode", type=str, default="both", choices=["clean", "aug", "both"])
    ap.add_argument(
        "--augment_affine",
        action="store_true",
        help="Insert PointMLPAffine between resample and NormalizeScale for training transforms.",
    )

    ap.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--bias_levels", type=str, default="0,1,2,3,4", help="comma-separated bias levels for stress test")

    ap.add_argument("--lambda_ortho_grid", type=str, default="0,0.001,0.01,0.1,1,10")
    ap.add_argument("--lambda_w_reg", type=float, default=0.0, help="(w-1)^2 reg for learned-weight models")

    ap.add_argument("--results_csv", type=str, default="umc_ablation_results.csv")
    ap.add_argument("--stability_bias", type=float, default=3.0)
    ap.add_argument("--stability_items", type=int, default=200)

    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for train/eval (0 keeps everything on the main process).")
    ap.add_argument("--torch_threads", type=int, default=None, help="Optional cap on torch threads to avoid overusing CPU cores during preprocessing.")
    ap.add_argument("--no_cache_eval", dest="cache_eval", action="store_false", help="Disable caching val/test sets after the first preprocessing pass.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print per-run timing information.")
    ap.set_defaults(cache_eval=True)

    args = ap.parse_args()

    if args.torch_threads is not None and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, args.torch_threads // 2))
        os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seeds = [int(s) for s in _parse_list(args.seeds)]
    bias_levels = [float(b) for b in _parse_list(args.bias_levels)]
    lambda_ortho_grid = [float(x) for x in _parse_list(args.lambda_ortho_grid)]

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = _parse_list(args.datasets)

    valid_datasets = set(DATASET_INFO.keys())
    invalid = [d for d in datasets if d not in valid_datasets]
    if invalid:
        raise ValueError(f"Unknown dataset(s): {invalid}. Choices: {sorted(valid_datasets)}")

    configs = _parse_list(args.umc_configs)
    invalid_cfg = [c for c in configs if c not in UMC_ABLATIONS]
    if invalid_cfg:
        raise ValueError(f"Unknown umc_configs: {invalid_cfg}. Choices: {sorted(UMC_ABLATIONS.keys())}")

    degree_features = args.degree_features
    rows = []

    for dataset_name in datasets:
        print("\n==============================")
        print("Dataset:", dataset_name)
        print("==============================")

        dataset_info = DATASET_INFO[dataset_name]
        num_classes = dataset_info["num_classes"]
        print(f"  num_classes: {num_classes}")

        # ------------------------------------------------------------
        # Pre-transform: cache a dense (aligned) point set once.
        # ------------------------------------------------------------
        pre_transform_dense = build_pre_transform(
            dataset_name=dataset_name,
            dense_points=args.dense_points,
        )

        # ------------------------------------------------------------
        # Build transform pipelines (phi always needed for UMC)
        # ------------------------------------------------------------
        train_transform_clean = build_train_transform_clean(
            num_points=args.num_points,
            knn_k=args.knn_k,
            K=args.K,
            include_phi=True,
            augment_affine=args.augment_affine,
            phi_device=args.phi_device,
        )

        train_transform_aug = build_train_transform_aug(
            num_points=args.num_points,
            knn_k=args.knn_k,
            K=args.K,
            max_bias=args.max_bias_train,
            include_phi=True,
            augment_affine=args.augment_affine,
            phi_device=args.phi_device,
        )

        test_transform_clean = build_test_transform_clean(
            num_points=args.num_points,
            knn_k=args.knn_k,
            K=args.K,
            include_phi=True,
            phi_device=args.phi_device,
        )

        transform_bias0 = build_stress_transform(
            num_points=args.num_points,
            knn_k=args.knn_k,
            K=args.K,
            bias_strength=0.0,
            include_phi=True,
            phi_device=args.phi_device,
        )
        transform_biasS = build_stress_transform(
            num_points=args.num_points,
            knn_k=args.knn_k,
            K=args.K,
            bias_strength=float(args.stability_bias),
            include_phi=True,
            phi_device=args.phi_device,
        )

        print("Loading datasets (may process once on first run) - ", end="", flush=True)
        if not args.root:
            root = f"../data/{dataset_name}"
            print(f"No root directory specified, using {root}", flush=True)
        else:
            root = args.root
            print(root, flush=True)

        test_dense_ds = load_dataset(
            name=dataset_name,
            root=root,
            train=False,
            pre_transform=pre_transform_dense,
            transform=None,
            force_reload=False,
        )

        train_modes = []
        if args.train_mode in ["clean", "both"]:
            train_modes.append(("train_clean", train_transform_clean))
        if args.train_mode in ["aug", "both"]:
            train_modes.append(("train_aug", train_transform_aug))

        for (mode_name, train_transform) in train_modes:
            print("\n=== TRAIN MODE:", mode_name, "===\n")

            train_ds_full = load_dataset(
                name=dataset_name,
                root=root,
                train=True,
                pre_transform=pre_transform_dense,
                transform=train_transform,
                force_reload=False,
            )
            test_ds = load_dataset(
                name=dataset_name,
                root=root,
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
                    print("Caching val/test once to reuse graph preprocessing...", flush=True)
                    val_ds_eff = cache_dataset(val_ds, num_workers=args.num_workers)
                    test_ds_eff = cache_dataset(test_ds, num_workers=args.num_workers)
                    print("Caching val/test done!", flush=True)
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

                for cfg_name in configs:
                    base_cfg = dict(UMC_ABLATIONS[cfg_name])
                    degree_label = None
                    if cfg_name == "deg_only":
                        if degree_features == "both":
                            base_cfg["use_deg"] = True
                            base_cfg["use_log_deg"] = True
                            degree_label = "deg+log_deg"
                        else:
                            base_cfg["use_deg"] = False
                            base_cfg["use_log_deg"] = True
                            degree_label = "log_deg"

                    for lam in lambda_ortho_grid:
                        if cfg_name == "full_point_encoder":
                            model = UMCPointEncoderClassifier(
                                K=args.K,
                                num_classes=num_classes,
                                use_pos=base_cfg["use_pos"],
                                use_md=base_cfg["use_md"],
                                use_log_md=base_cfg["use_log_md"],
                                use_log_deg=base_cfg["use_log_deg"],
                                use_deg=base_cfg["use_deg"],
                            ).to(device)
                        else:
                            model = UMCClassifier(
                                K=args.K,
                                num_classes=num_classes,
                                use_pos=base_cfg["use_pos"],
                                use_md=base_cfg["use_md"],
                                use_log_md=base_cfg["use_log_md"],
                                use_log_deg=base_cfg["use_log_deg"],
                                use_deg=base_cfg["use_deg"],
                            ).to(device)

                        cfg = TrainConfig(
                            epochs=args.epochs,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            lambda_ortho=lam,
                            lambda_w_reg=args.lambda_w_reg,
                        )

                        t0 = time.time()
                        metrics = train_model(
                            model,
                            train_loader,
                            val_loader,
                            test_loader,
                            device,
                            K=args.K,
                            cfg=cfg,
                            num_classes=num_classes,
                            verbose=args.verbose,
                        )
                        dt = time.time() - t0
                        timing_suffix = f" (took {format_duration(dt)})" if args.verbose else ""

                        stress = eval_stress_table(
                            model=model,
                            dataset_name=dataset_name,
                            root=root,
                            pre_transform=pre_transform_dense,
                            bias_levels=bias_levels,
                            num_points=args.num_points,
                            knn_k=args.knn_k,
                            K=args.K,
                            device=device,
                            batch_size=args.batch_size,
                            seed=seed,
                            num_workers=args.num_workers,
                            include_phi=True,
                            phi_device=args.phi_device,
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
                            "dataset": dataset_name,
                            "train_mode": mode_name,
                            "seed": seed,
                            "variant": cfg_name,
                            "lambda_ortho": lam,
                            "degree_features": degree_label,
                            "time_sec": dt,
                            **metrics,
                            **stress,
                            **corrs,
                            "feature_stability_cos": stab,
                        }
                        rows.append(row)
                        print(
                            f"[{mode_name} | seed={seed} | {cfg_name} | lam={lam}] "
                            f"test_acc={metrics['test_acc']*100:.2f}% test_macc={metrics['test_macc']*100:.2f}%  "
                            f"stress@{bias_levels[-1]}={stress[f'stress_bias_{bias_levels[-1]:.1f}']*100:.2f}%"
                            f"{timing_suffix}"
                        )

    df = pd.DataFrame(rows)
    df.to_csv(args.results_csv, index=False)
    print("\nSaved:", args.results_csv)

    group_cols = ["dataset", "train_mode", "variant", "lambda_ortho", "degree_features"]
    summary = df.groupby(group_cols).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        test_macc_mean=("test_macc", "mean"),
        test_macc_std=("test_macc", "std"),
        stress_max_mean=(f"stress_bias_{bias_levels[-1]:.1f}", "mean"),
        stress_max_std=(f"stress_bias_{bias_levels[-1]:.1f}", "std"),
        stab_mean=("feature_stability_cos", "mean"),
        stab_std=("feature_stability_cos", "std"),
    ).reset_index()

    summary["test_acc_mean"] *= 100
    summary["test_acc_std"] *= 100
    summary["test_macc_mean"] *= 100
    summary["test_macc_std"] *= 100
    summary["stress_max_mean"] *= 100
    summary["stress_max_std"] *= 100

    print("\n=== Summary (mean±std over seeds) ===")
    print(summary.sort_values(["dataset", "train_mode", "variant", "lambda_ortho"]).to_string(index=False))


if __name__ == "__main__":
    main()

"""Train/evaluate PointWavelet baselines on ModelNet10/40.

This script intentionally runs only two methods:
  1) PointWavelet (default: PointWavelet-L, i.e. learnable spectral basis)
  2) PointWavelet + UMC (learned quadrature weights inside WaveletFormer)

Paper-faithful parts implemented here:
  - 1024 points sampled from ModelNet meshes
  - SA config: (512/128/32/1) centroids with k=32 neighbors
  - WaveletFormer: 2 transformer encoders, 4 heads per WF block
  - Mexican hat wavelet: h(λ)=exp(-λ^4), g(λ)=λ exp(-λ)
  - J=5 scales (dyadic by default)
  - PointWavelet-L regularization: beta * ||q_eps||_1 with beta=0.05

The PointWavelet paper does not spell out all optimization and augmentation
hyperparameters in the PDF; this script exposes common knobs via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints

from pointwavelet import PointWaveletClassifier, PointWaveletClsConfig
from utils.transforms import PointJitter, PointMLPAffine


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
    # Determinism (can be slower; toggle if you want max speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def eval_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    correct = 0
    total = 0
    correct_per_class = torch.zeros(num_classes, device=device)
    count_per_class = torch.zeros(num_classes, device=device)

    for xyz, y in loader:
        xyz = xyz.to(device)
        y = y.to(device)
        logits = model(xyz)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

        count_per_class += torch.bincount(y, minlength=num_classes).to(count_per_class.dtype)
        correct_per_class += torch.bincount(y[pred.eq(y)], minlength=num_classes).to(correct_per_class.dtype)

    oa = correct / max(total, 1)
    valid = count_per_class > 0
    macc = float((correct_per_class[valid] / count_per_class[valid]).mean().item()) if bool(valid.any()) else 0.0
    return {"oa": float(oa), "macc": float(macc)}


def collate_points(items) -> Tuple[torch.Tensor, torch.Tensor]:
    # items are torch_geometric.data.Data
    xyz = torch.stack([it.pos for it in items], dim=0)  # (B,N,3)
    y = torch.tensor([int(it.y.item()) for it in items], dtype=torch.long)
    return xyz, y


def build_datasets(root: str, num_points: int, force_reload: bool, modelnet: str) -> Tuple[ModelNet, ModelNet]:
    train_transform = Compose(
        [
            SamplePoints(num_points),
            NormalizeScale(),
            PointMLPAffine(),
            PointJitter(),
        ]
    )
    test_transform = Compose(
        [
            SamplePoints(num_points),
            NormalizeScale(),
        ]
    )
    train_ds = ModelNet(
        root=root,
        name=modelnet,
        train=True,
        pre_transform=None,
        transform=train_transform,
        force_reload=force_reload,
    )
    test_ds = ModelNet(
        root=root,
        name=modelnet,
        train=False,
        pre_transform=None,
        transform=test_transform,
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
    num_classes: int,
    wf_J: int = 5,
    wf_beta: float = 0.05,
) -> PointWaveletClassifier:
    cfg = PointWaveletClsConfig(
        num_classes=num_classes,
        input_channels=0,
        wf_learnable=wf_learnable,
        wf_beta=wf_beta,
        wf_J=wf_J,
        wf_depth=2,
        wf_heads=4,
        wf_sigma_mode="mean",
        wf_use_umc=use_umc,
        wf_umc_hidden=umc_hidden,
        wf_umc_knn=umc_knn,
        wf_umc_min_weight=umc_min_weight,
        wf_umc_use_inverse=umc_use_inverse,
    )
    return PointWaveletClassifier(cfg)


def train_one(
    model: PointWaveletClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    lr_step: int,
    lr_gamma: float,
    amp: bool,
    num_classes: int,
    wandb_run: Optional[object] = None,
) -> dict:
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(lr_step, 1), gamma=lr_gamma)

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"oa": 0.0, "macc": 0.0, "epoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for xyz, y in train_loader:
            xyz = xyz.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(xyz)
                cls_loss = F.cross_entropy(logits, y)
                reg_loss = model.regularization_loss()
                loss = cls_loss + reg_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = int(y.numel())
            total_loss += float(loss.item()) * bs
            total_items += bs

        scheduler.step()

        metrics = eval_metrics(model, test_loader, device=device, num_classes=num_classes)
        if metrics["oa"] > best["oa"]:
            best = {**metrics, "epoch": epoch}

        avg_loss = total_loss / max(total_items, 1)
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={lr_now:.2e} | test_OA={metrics['oa']*100:.2f} | test_mAcc={metrics['macc']*100:.2f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": avg_loss,
                    "lr": lr_now,
                    "test_OA": metrics["oa"] * 100,
                    "test_mAcc": metrics["macc"] * 100,
                },
                step=epoch,
            )

    final = eval_metrics(model, test_loader, device=device, num_classes=num_classes)
    return {"final": final, "best": best}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data", help="Root directory for torch_geometric ModelNet")
    p.add_argument(
        "--modelnet",
        type=str,
        default="40",
        choices=["10", "40"],
        help="Which dataset split to use: ModelNet10 or ModelNet40.",
    )
    p.add_argument("--force_reload", action="store_true", help="Force reprocessing the dataset")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_step", type=int, default=20)
    p.add_argument("--lr_gamma", type=float, default=0.7)
    p.add_argument("--amp", action="store_true", help="Use CUDA AMP (fp16) if available")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. '0,1,2,3'")
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["vanilla", "umc", "both"],
        help="Which method(s) to run: vanilla (PointWavelet), umc, or both.",
    )

    # PointWavelet variant
    # Default to PointWavelet-L (learnable basis). Use --no-wf_learnable to
    # switch to per-patch eigendecomposition (slow).
    p.add_argument(
        "--wf_learnable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PointWavelet-L (learnable spectral basis).",
    )

    # UMC settings
    p.add_argument("--umc_hidden", type=str, default="32,32", help="Hidden widths for UMC MLP, e.g. '32,32'")
    p.add_argument("--umc_knn", type=int, default=8)
    p.add_argument("--umc_min_weight", type=float, default=1e-3)
    p.add_argument("--umc_no_inverse", action="store_true", help="Disable the W^{-1} factor in reconstruction")

    p.add_argument("--out_csv", type=str, default="pointwavelet_umc_results.csv")
    return p.parse_args()


def _parse_int_pair(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two ints like '32,32', got: {s}")
    return int(parts[0]), int(parts[1])


def _parse_seeds(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


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

    train_ds, test_ds = build_datasets(
        args.data_root,
        args.num_points,
        force_reload=bool(args.force_reload),
        modelnet=str(args.modelnet),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_points,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_points,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = 10 if str(args.modelnet) == "10" else 40
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
                num_classes=num_classes,
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
                amp=bool(args.amp),
                num_classes=num_classes,
                wandb_run=run,
            )
            if run is not None:
                run.finish()
            results.append(
                {
                    "seed": seed,
                    "method": method_key,
                    **out["final"],
                    "best_oa": out["best"]["oa"],
                    "best_epoch": out["best"]["epoch"],
                }
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Write results
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["seed", "method", "oa", "macc", "best_oa", "best_epoch"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})

    print(f"\nSaved results to {args.out_csv}", flush=True)
    # Also print a short summary
    for m, _, _ in methods:
        vals = [r["oa"] for r in results if r["method"] == m]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"{m}: OA mean={mean*100:.2f} std={std*100:.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()

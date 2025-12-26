"""Evaluate PointWavelet checkpoints under IrregularResample stress for multiple betas."""

from __future__ import annotations

import argparse
import copy
import os
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, NormalizeScale, SamplePoints

from pointwavelet import PointWaveletClassifier, PointWaveletClsConfig
from umc_pointwavelet import _device, _eval_stress_accuracy, _preserve_rng_state, _seed_rng, build_stress_loader
from utils.transforms import IrregularResample


def _parse_betas(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint saved by umc_pointwavelet.py")
    p.add_argument("--data_root", type=str, default="data", help="Root directory for torch_geometric ModelNet (default: data)")
    p.add_argument("--betas", type=str, default="2.0", help="Comma-separated bias strengths (default: 2.0)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for stress DataLoader (default: 16)")
    p.add_argument("--stress_dense_points", type=int, default=2048, help="Dense points before stress resample (default: 2048)")
    p.add_argument("--modelnet", type=str, default=None, help="Override ModelNet split (10 or 40). Defaults to ckpt meta.")
    p.add_argument("--num_points", type=int, default=None, help="Override num_points. Defaults to ckpt meta.")
    p.add_argument("--seed", type=int, default=None, help="Override stress seed base. Defaults to ckpt meta.")
    p.add_argument("--force_reload", action="store_true", help="Force reprocessing the dataset")
    p.add_argument("--debug", action="store_true", help="Save 3D scatter images for the first test sample per beta")
    p.add_argument("--debug_dir", type=str, default="stress_debug", help="Directory for debug images (default: stress_debug)")
    return p.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> tuple[PointWaveletClassifier, dict]:
    ckpt = torch.load(path, map_location=device)
    if "cfg" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint missing required keys: 'cfg' and 'model_state'")
    cfg = PointWaveletClsConfig(**ckpt["cfg"])
    model = PointWaveletClassifier(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, ckpt.get("meta", {})


def _beta_tag(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def _set_axes_equal(ax, xyz: np.ndarray) -> None:
    ranges = np.ptp(xyz, axis=0)
    max_range = float(ranges.max() if ranges.size else 1.0)
    centers = xyz.mean(axis=0)
    half = max_range / 2.0
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def _get_debug_base_sample(
    root: str,
    modelnet: str,
    dense_points: int,
    force_reload: bool,
    seed: int,
) -> object:
    def _run():
        _seed_rng(seed)
        base_transform = Compose([SamplePoints(dense_points)])
        ds = ModelNet(
            root=root,
            name=modelnet,
            train=False,
            pre_transform=None,
            transform=base_transform,
            force_reload=force_reload,
        )
        return ds[0]

    return _preserve_rng_state(_run)


def _apply_debug_stress(
    data: object,
    num_points: int,
    beta: float,
    seed: int,
) -> object:
    def _run():
        _seed_rng(seed)
        transform = Compose(
            [
                IrregularResample(num_points=num_points, bias_strength=beta),
                NormalizeScale(),
            ]
        )
        return transform(copy.deepcopy(data))

    return _preserve_rng_state(_run)


def _save_debug_plot(data: object, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    pos = data.pos.detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=4, c=pos[:, 2], cmap="viridis", alpha=0.9, linewidths=0)
    ax.view_init(elev=20, azim=45)
    _set_axes_equal(ax, pos)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = _device()

    model, meta = _load_checkpoint(args.ckpt, device=device)
    model = model.to(device)

    modelnet = args.modelnet if args.modelnet is not None else str(meta.get("modelnet", "40"))
    num_points = args.num_points if args.num_points is not None else int(meta.get("num_points", 1024))
    seed = args.seed if args.seed is not None else int(meta.get("seed", 0))

    betas = _parse_betas(args.betas)
    print(f"Loaded {args.ckpt}", flush=True)
    print(f"ModelNet{modelnet} | num_points={num_points} | base_seed={seed}", flush=True)

    debug_sample = None
    if args.debug:
        os.makedirs(args.debug_dir, exist_ok=True)
        debug_sample = _get_debug_base_sample(
            root=args.data_root,
            modelnet=str(modelnet),
            dense_points=args.stress_dense_points,
            force_reload=bool(args.force_reload),
            seed=seed,
        )

    for beta in betas:
        stress_loader = build_stress_loader(
            root=args.data_root,
            num_points=num_points,
            dense_points=args.stress_dense_points,
            modelnet=str(modelnet),
            bias_strength=float(beta),
            batch_size=args.batch_size,
            force_reload=bool(args.force_reload),
            device=device,
        )
        eval_seed = seed + int(round(1000 * float(beta)))
        acc = _eval_stress_accuracy(model, stress_loader, device=device, seed=eval_seed)
        print(f"beta={beta:.2f} | stress_OA={acc*100:.2f}", flush=True)
        if args.debug and debug_sample is not None:
            debug_seed = seed + int(round(1000 * float(beta)))
            stressed = _apply_debug_stress(debug_sample, num_points=num_points, beta=float(beta), seed=debug_seed)
            out_path = os.path.join(args.debug_dir, f"stress_beta_{_beta_tag(beta)}.png")
            _save_debug_plot(stressed, out_path, title=f"beta={beta:.2f}")
            print(f"debug saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

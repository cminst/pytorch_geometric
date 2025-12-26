"""Evaluate PointWavelet checkpoints under IrregularResample stress for multiple betas."""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch

from pointwavelet import PointWaveletClassifier, PointWaveletClsConfig
from umc_pointwavelet import _device, _eval_stress_accuracy, build_stress_loader


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
    return p.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> tuple[PointWaveletClassifier, dict]:
    ckpt = torch.load(path, map_location=device)
    if "cfg" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint missing required keys: 'cfg' and 'model_state'")
    cfg = PointWaveletClsConfig(**ckpt["cfg"])
    model = PointWaveletClassifier(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, ckpt.get("meta", {})


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


if __name__ == "__main__":
    main()

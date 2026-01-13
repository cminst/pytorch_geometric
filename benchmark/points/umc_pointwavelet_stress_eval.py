"""Evaluate PointWavelet checkpoints under IrregularResample stress for multiple betas.

Supports classification (ModelNet10/40, ScanObjectNN) and ShapeNet part
segmentation checkpoints.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose, NormalizeScale

from pointwavelet import (
    PointWaveletClassifier,
    PointWaveletClsConfig,
    PointWaveletPartSeg,
    PointWaveletPartSegConfig,
)
from umc_pointwavelet import (
    _device,
    _eval_stress_accuracy,
    _preserve_rng_state,
    _resolve_dataset_root,
    _seed_rng,
    build_stress_loader,
)
from umc_pointwavelet_segmentation import collate_segmentation, eval_metrics as eval_segmentation_metrics
from utils.custom_datasets import ShapeNetPatched


def _parse_betas(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def _parse_seeds(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def _parse_categories(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else None


def _is_segmentation_cfg(cfg: dict) -> bool:
    return "num_parts" in cfg


def _canonical_seg_dataset_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    low = name.strip().lower()
    if low in {"shapenet", "shapenetpart", "shapenetpartseg"}:
        return "ShapeNet"
    return None


class SegIrregularResample:
    """Irregular resample that keeps x/y aligned for segmentation."""

    def __init__(self, num_points: int = 2048, bias_strength: float = 0.0) -> None:
        self.num_points = int(num_points)
        self.bias = float(bias_strength)

    def __call__(self, data):
        pos = data.pos
        device = pos.device
        n_curr = int(pos.size(0))
        if n_curr <= 0:
            return data

        if self.bias > 1e-6:
            focus = torch.randn(1, 3, device=device)
            focus = focus / (focus.norm() + 1e-12)
            proj = (pos @ focus.T).squeeze()
            proj = proj - proj.max()
            weights = torch.exp(self.bias * proj)
            weights = weights / (weights.sum() + 1e-12)
            idx = torch.multinomial(weights, self.num_points, replacement=True)
        else:
            if n_curr >= self.num_points:
                idx = torch.randperm(n_curr, device=device)[:self.num_points]
            else:
                idx = torch.randint(0, n_curr, (self.num_points,), device=device)

        data.pos = pos[idx]
        if hasattr(data, "x") and data.x is not None and data.x.size(0) == n_curr:
            data.x = data.x[idx]
        if hasattr(data, "y") and data.y is not None and data.y.numel() == n_curr:
            data.y = data.y[idx]
        for key in ["edge_index", "edge_attr", "phi", "phi_evals", "deg", "batch"]:
            if hasattr(data, key):
                delattr(data, key)
        data.num_nodes = self.num_points
        return data


def build_segmentation_stress_loader(
    root: str,
    num_points: int,
    bias_strength: float,
    batch_size: int,
    force_reload: bool,
    device: torch.device,
    categories: Optional[List[str]],
    include_normals: bool,
    normalize: bool,
) -> tuple[DataLoader, List[str], Dict[str, List[int]]]:
    pre_transform = NormalizeScale() if normalize else None
    stress_transform = Compose([SegIrregularResample(num_points=num_points, bias_strength=bias_strength)])
    stress_ds = ShapeNetPatched(
        root=root,
        categories=categories,
        include_normals=include_normals,
        split="test",
        transform=stress_transform,
        pre_transform=pre_transform,
        force_reload=force_reload,
    )
    loader = DataLoader(
        stress_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segmentation,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    return loader, list(stress_ds.categories), stress_ds.seg_classes


@torch.no_grad()
def _eval_stress_segmentation_metrics(
    model: PointWaveletPartSeg,
    loader: DataLoader,
    device: torch.device,
    categories: List[str],
    seg_classes: Dict[str, List[int]],
    seed: int,
) -> dict:
    def _run():
        _seed_rng(seed)
        return eval_segmentation_metrics(
            model,
            loader,
            device=device,
            categories=categories,
            seg_classes=seg_classes,
        )

    return _preserve_rng_state(_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint saved by umc_pointwavelet.py")
    p.add_argument("--data_root", type=str, default="data", help="Root directory for torch_geometric ModelNet (default: data)")
    p.add_argument("--betas", type=str, default="2.0", help="Comma-separated bias strengths (default: 2.0)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for stress DataLoader (default: 16)")
    p.add_argument("--stress_dense_points", type=int, default=2048, help="Dense points before stress resample (default: 2048)")
    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        choices=["ModelNet10", "ModelNet40", "ScanObjectNN", "ShapeNet", "ShapeNetPart", "ShapeNetPartSeg"],
        help="Override dataset name. Defaults to ckpt meta if available.",
    )
    p.add_argument("--modelnet", type=str, default=None, help="Override ModelNet split (10 or 40). Defaults to ckpt meta.")
    p.add_argument("--num_points", type=int, default=None, help="Override num_points. Defaults to ckpt meta.")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated stress seeds. Defaults to ckpt meta seed.",
    )
    p.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated ShapeNet categories for segmentation (default: all).",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize ShapeNet point clouds (default: False).",
    )
    p.add_argument("--force_reload", action="store_true", help="Force reprocessing the dataset")
    return p.parse_args()


def _load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if "cfg" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint missing required keys: 'cfg' and 'model_state'")
    return ckpt


def main() -> None:
    args = parse_args()
    device = _device()

    ckpt = _load_checkpoint(args.ckpt, device=device)
    cfg_dict = ckpt["cfg"]
    meta = ckpt.get("meta", {})
    is_segmentation = _is_segmentation_cfg(cfg_dict)

    seg_dataset_name = _canonical_seg_dataset_name(args.dataset_name)
    if is_segmentation:
        if args.dataset_name is not None and seg_dataset_name is None:
            raise ValueError(
                f"Segmentation checkpoint expects ShapeNet dataset; got dataset_name={args.dataset_name}."
            )
        dataset_name = seg_dataset_name or "ShapeNet"
    else:
        if seg_dataset_name is not None:
            raise ValueError(
                "Classification checkpoint does not support ShapeNet. "
                "Provide a segmentation checkpoint instead."
            )
        if args.dataset_name is not None:
            dataset_name = args.dataset_name
        else:
            dataset_name = str(meta.get("dataset_name", "") or "")
            if not dataset_name:
                modelnet = args.modelnet if args.modelnet is not None else str(meta.get("modelnet", "40"))
                dataset_name = "ModelNet10" if str(modelnet) == "10" else "ModelNet40"

    default_num_points = 2048 if is_segmentation else 1024
    num_points = args.num_points if args.num_points is not None else int(meta.get("num_points", default_num_points))
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [int(meta.get("seed", 0))]

    betas = _parse_betas(args.betas)
    task = "segmentation" if is_segmentation else "classification"
    print(f"Loaded {args.ckpt} ({task})", flush=True)

    dataset_root = _resolve_dataset_root(args.data_root, dataset_name)

    if is_segmentation:
        cfg = PointWaveletPartSegConfig(**cfg_dict)
        model = PointWaveletPartSeg(cfg)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device)

        categories = _parse_categories(args.categories)
        include_normals = bool(cfg.input_channels > 0)
        if cfg.use_category_label:
            expected = int(cfg.category_embed_dim)
            if categories is not None and len(categories) != expected:
                raise ValueError(
                    f"categories length ({len(categories)}) must match category_embed_dim ({expected})"
                )

        print(
            f"{dataset_name} | num_points={num_points} | normals={include_normals} | "
            f"normalize={bool(args.normalize)} | seeds={','.join(str(s) for s in seeds)}",
            flush=True,
        )

        for beta in betas:
            stress_loader, category_list, seg_classes = build_segmentation_stress_loader(
                root=dataset_root,
                num_points=num_points,
                bias_strength=float(beta),
                batch_size=args.batch_size,
                force_reload=bool(args.force_reload),
                device=device,
                categories=categories,
                include_normals=include_normals,
                normalize=bool(args.normalize),
            )
            if cfg.use_category_label and len(category_list) != int(cfg.category_embed_dim):
                raise ValueError(
                    f"dataset categories ({len(category_list)}) must match category_embed_dim ({cfg.category_embed_dim})"
                )
            metrics_list = []
            for seed in seeds:
                eval_seed = seed + int(round(1000 * float(beta)))
                metrics = _eval_stress_segmentation_metrics(
                    model,
                    stress_loader,
                    device=device,
                    categories=category_list,
                    seg_classes=seg_classes,
                    seed=eval_seed,
                )
                metrics_list.append(metrics)
            oas = [m["oa"] for m in metrics_list]
            inst_ious = [m["inst_miou"] for m in metrics_list]
            class_ious = [m["class_miou"] for m in metrics_list]
            mean_oa = float(np.mean(oas)) if oas else 0.0
            std_oa = float(np.std(oas)) if oas else 0.0
            mean_inst = float(np.mean(inst_ious)) if inst_ious else 0.0
            std_inst = float(np.std(inst_ious)) if inst_ious else 0.0
            mean_cls = float(np.mean(class_ious)) if class_ious else 0.0
            std_cls = float(np.std(class_ious)) if class_ious else 0.0
            print(
                "beta={beta:.2f} | stress_OA={oa:.2f} ± {oa_std:.2f} | "
                "stress_InsIoU={ins:.2f} ± {ins_std:.2f} | "
                "stress_ClsIoU={cls:.2f} ± {cls_std:.2f} (n={n})".format(
                    beta=beta,
                    oa=mean_oa * 100,
                    oa_std=std_oa * 100,
                    ins=mean_inst * 100,
                    ins_std=std_inst * 100,
                    cls=mean_cls * 100,
                    cls_std=std_cls * 100,
                    n=len(metrics_list),
                ),
                flush=True,
            )
    else:
        cfg = PointWaveletClsConfig(**cfg_dict)
        model = PointWaveletClassifier(cfg)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device)
        print(
            f"{dataset_name} | num_points={num_points} | seeds={','.join(str(s) for s in seeds)}",
            flush=True,
        )

        for beta in betas:
            stress_loader = build_stress_loader(
                root=dataset_root,
                num_points=num_points,
                dense_points=args.stress_dense_points,
                dataset_name=dataset_name,
                bias_strength=float(beta),
                batch_size=args.batch_size,
                force_reload=bool(args.force_reload),
                device=device,
            )
            accs = []
            for seed in seeds:
                eval_seed = seed + int(round(1000 * float(beta)))
                acc = _eval_stress_accuracy(model, stress_loader, device=device, seed=eval_seed)
                accs.append(acc)
            mean = float(np.mean(accs)) if accs else 0.0
            std = float(np.std(accs)) if accs else 0.0
            print(
                f"beta={beta:.2f} | stress_OA={mean*100:.2f} ± {std*100:.2f} (n={len(accs)})",
                flush=True,
            )


if __name__ == "__main__":
    main()

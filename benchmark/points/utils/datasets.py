from __future__ import annotations

import copy
import os.path as osp
from typing import Callable, Optional

import torch_geometric.transforms as T
import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (
    Compose,
    KNNGraph,
    NormalizeScale,
    SamplePoints,
)
from utils.custom_datasets import ScanObjectNN
from utils.transforms import (
    ComputePhiRWFromSym,
    IrregularResample,
    MakeUndirected,
    PointMLPAffine,
    RandomIrregularResample,
)

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
    if name == "ModelNet40":
        return ModelNet(
            root=root,
            name="40",
            train=train,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
        )
    if name == "ScanObjectNN":
        return ScanObjectNN(
            root=root,
            train=train,
            pre_transform=pre_transform,
            transform=transform,
            force_reload=force_reload,
            variant="OBJ_ONLY",
        )
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
    K: int,
    include_phi: bool = True,
    augment_affine: bool = False,
    phi_device: Optional[str] = None,
) -> Compose:
    """Build clean (uniform) training transform."""
    transforms = [IrregularResample(num_points=num_points, bias_strength=0.0)]
    if augment_affine:
        transforms.append(PointMLPAffine())
    transforms.extend([
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
    ])
    if include_phi:
        transforms.append(ComputePhiRWFromSym(K=K, store_aux=True, eig_device=phi_device))
    return Compose(transforms)


def build_train_transform_aug(
    num_points: int,
    knn_k: int,
    K: int,
    max_bias: float,
    include_phi: bool = True,
    augment_affine: bool = False,
    phi_device: Optional[str] = None,
) -> Compose:
    """Build augmented (random bias) training transform."""
    transforms = [RandomIrregularResample(num_points=num_points, max_bias=max_bias)]
    if augment_affine:
        transforms.append(PointMLPAffine())
    transforms.extend([
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
    ])
    if include_phi:
        transforms.append(ComputePhiRWFromSym(K=K, store_aux=True, eig_device=phi_device))
    return Compose(transforms)


def build_test_transform_clean(
    num_points: int,
    knn_k: int,
    K: int,
    include_phi: bool = True,
    phi_device: Optional[str] = None,
) -> Compose:
    """Build clean (uniform) test transform."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=0.0),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
    ]
    if include_phi:
        transforms.append(ComputePhiRWFromSym(K=K, store_aux=True, eig_device=phi_device))
    return Compose(transforms)


def build_stress_transform(
    num_points: int,
    knn_k: int,
    K: int,
    bias_strength: float,
    include_phi: bool = True,
    phi_device: Optional[str] = None,
) -> Compose:
    """Build stress test transform with specific bias level."""
    transforms = [
        IrregularResample(num_points=num_points, bias_strength=bias_strength),
        NormalizeScale(),
        KNNGraph(k=knn_k),
        MakeUndirected(),
    ]
    if include_phi:
        transforms.append(ComputePhiRWFromSym(K=K, store_aux=True, eig_device=phi_device))
    return Compose(transforms)


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
    """Create train/val/test loaders with deterministic train shuffling."""
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
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=persistent_workers)
    return train_loader, val_loader, test_loader


def get_modelnet_dataset(num_points):
    name = 'ModelNet10'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(path, name='10', train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name='10', train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset

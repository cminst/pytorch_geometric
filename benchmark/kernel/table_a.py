import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU, Sequential

from torch_geometric import seed_everything
from torch_geometric.data import Batch
from torch_geometric.nn import (
    ASAPooling,
    EdgePooling,
    GCNConv,
    LaCorePooling,
    SAGPooling,
    Set2Set,
    TopKPooling,
    graclus,
    global_max_pool,
    global_mean_pool,
    max_pool,
)

from datasets import get_dataset
from diff_pool import DiffPool
from lacore_pool import LaCoreAssignment, LaCore
from train_eval import cross_validation_with_val_set, single_split_train_eval


POOLING_METHODS = [
    "gcn",
    "set2set",
    "graclus",
    "diffpool",
    "topk",
    "sag",
    "asap",
    "edge",
    "lacore",
]


def parse_csv_list(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [item.strip() for item in arg.split(",") if item.strip()]


def parse_seeds(arg: Optional[str], default: Iterable[int]) -> List[int]:
    if not arg:
        return list(default)
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def _unpack_pool_output(out: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(out) == 6:
        x, edge_index, _, batch, _, _ = out
    elif len(out) == 5:
        x, edge_index, _, batch, _ = out
    elif len(out) == 4:
        x, edge_index, batch, _ = out
    else:
        raise ValueError(f"Unexpected pool output length: {len(out)}")
    return x, edge_index, batch


class FixedBackbonePoolNet(torch.nn.Module):
    def __init__(
        self,
        dataset,
        hidden: int,
        dropout: float,
        pool_name: str,
        pool_ratio: float,
        readout: str = "global",
    ) -> None:
        super().__init__()
        self.pool_name = pool_name
        self.readout = readout
        self.dropout = dropout

        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)

        if pool_name == "topk":
            self.pool = TopKPooling(hidden, ratio=pool_ratio)
        elif pool_name == "sag":
            self.pool = SAGPooling(hidden, ratio=pool_ratio)
        elif pool_name == "asap":
            self.pool = ASAPooling(hidden, ratio=pool_ratio)
        elif pool_name == "edge":
            self.pool = EdgePooling(hidden)
        elif pool_name == "lacore":
            self.pool = LaCorePooling(aggregate="mean")
        elif pool_name in {"none", "graclus"}:
            self.pool = None
        else:
            raise ValueError(f"Unknown pool_name: {pool_name}")

        if readout == "set2set":
            self.set2set = Set2Set(hidden, processing_steps=4)
            self.lin = Sequential(
                Linear(2 * hidden, hidden),
                ReLU(),
                Dropout(dropout),
                Linear(hidden, dataset.num_classes),
            )
        else:
            self.set2set = None
            self.lin = Sequential(
                Linear(4 * hidden, 2 * hidden),
                ReLU(),
                Dropout(dropout),
                Linear(2 * hidden, dataset.num_classes),
            )

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.pool is not None and hasattr(self.pool, "reset_parameters"):
            self.pool.reset_parameters()
        if self.set2set is not None:
            self.set2set.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _apply_pool(self, x, edge_index, batch, data):
        if self.pool_name in {"none"}:
            return x, edge_index, batch
        if self.pool_name == "graclus":
            cluster = graclus(edge_index, num_nodes=x.size(0))
            pooled = max_pool(cluster, Batch(x=x, edge_index=edge_index, batch=batch))
            return pooled.x, pooled.edge_index, pooled.batch
        if self.pool_name == "lacore":
            if not hasattr(data, "cluster") or not hasattr(data, "num_clusters"):
                raise ValueError("LaCore pooling requires data.cluster and data.num_clusters.")
            out = self.pool(x, edge_index, batch, data.cluster, data.num_clusters)
            return _unpack_pool_output(out)

        out = self.pool(x, edge_index, batch=batch)
        return _unpack_pool_output(out)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        g0_mean = global_mean_pool(x, batch)
        g0_max = global_max_pool(x, batch)

        x, edge_index, batch = self._apply_pool(x, edge_index, batch, data)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.readout == "set2set":
            g = self.set2set(x, batch)
            out = self.lin(g)
        else:
            g1_mean = global_mean_pool(x, batch)
            g1_max = global_max_pool(x, batch)
            g = torch.cat([g0_mean, g0_max, g1_mean, g1_max], dim=-1)
            out = self.lin(g)
        return F.log_softmax(out, dim=-1)


def build_dataset_config(args) -> Dict[str, dict]:
    cfg = {
        "ModelNet40": {
            "num_points": args.num_points,
            "knn_k": args.knn_k,
            "canonical_split": True,
            "val_ratio": args.val_ratio,
            "precompute": args.precompute,
        },
        "ScanObjectNN": {
            "num_points": args.num_points,
            "knn_k": args.knn_k,
            "canonical_split": True,
            "val_ratio": args.val_ratio,
            "variant": args.scan_variant,
            "split_dir": args.scan_split_dir,
            "resample": args.scan_resample,
            "bias_strength": args.scan_bias,
            "precompute": args.precompute,
        },
    }
    return cfg


def build_lacore_assignment(dataset_name: str, args) -> LaCoreAssignment:
    hp = LaCore.default_hparams(dataset_name)
    epsilon = args.lacore_epsilon if args.lacore_epsilon is not None else hp.epsilon
    target_ratio = args.lacore_target_ratio if args.lacore_target_ratio is not None else hp.target_ratio
    min_size = args.lacore_min_size if args.lacore_min_size is not None else hp.min_size
    max_clusters = args.lacore_max_clusters if args.lacore_max_clusters is not None else hp.max_clusters
    return LaCoreAssignment(
        epsilon=epsilon,
        target_ratio=target_ratio,
        min_size=min_size,
        max_clusters=max_clusters,
    )


def load_dataset(
    dataset_name: str,
    args,
    dataset_config: Dict[str, dict],
    method: str,
):
    sparse = method != "diffpool"
    extra_transform = None
    if method == "lacore":
        extra_transform = build_lacore_assignment(dataset_name, args)
    data_seed = args.data_seed
    if data_seed is not None:
        seed_everything(data_seed)
    return get_dataset(
        dataset_name,
        sparse=sparse,
        extra_transform=extra_transform,
        dataset_config=dataset_config,
        dataset_root=args.dataset_root,
    )


def build_model(method: str, dataset, args):
    if method == "diffpool":
        return DiffPool(
            dataset,
            num_layers=args.diffpool_layers,
            hidden=args.hidden,
            ratio=args.pool_ratio,
        )
    if method == "gcn":
        pool_name = "none"
        readout = "global"
    elif method == "set2set":
        pool_name = "none"
        readout = "set2set"
    else:
        pool_name = method
        readout = "global"
    return FixedBackbonePoolNet(
        dataset=dataset,
        hidden=args.hidden,
        dropout=args.dropout,
        pool_name=pool_name,
        pool_ratio=args.pool_ratio,
        readout=readout,
    )


def run_method_on_dataset(
    method: str,
    dataset_name: str,
    dataset,
    args,
):
    if isinstance(dataset, tuple):
        train_dataset, test_dataset = dataset
        dataset_for_model = train_dataset
        runner = "single_split"
    else:
        dataset_for_model = dataset
        runner = "cross_val"

    accs: List[float] = []
    for seed in args.seeds:
        seed_everything(seed)
        model = build_model(method, dataset_for_model, args)
        if runner == "single_split":
            _, acc, _ = single_split_train_eval(
                train_dataset,
                test_dataset,
                model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=args.weight_decay,
                val_ratio=args.val_ratio,
                selection_metric=args.selection_metric,
                seed=seed,
            )
        else:
            _, acc, _ = cross_validation_with_val_set(
                dataset,
                model,
                folds=args.folds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=args.weight_decay,
                selection_metric=args.selection_metric,
                kfold_seed=seed,
                use_inner_val=(method == "lacore"),
            )
        accs.append(float(acc))
    mean = float(np.mean(accs)) if accs else 0.0
    std = float(np.std(accs)) if accs else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Table A: fixed-backbone pooling comparison")
    parser.add_argument("--datasets", type=str, default="ModelNet40,ScanObjectNN",
                        help="Comma-separated dataset names.")
    parser.add_argument("--methods", type=str, default=",".join(POOLING_METHODS),
                        help="Comma-separated pooling/readout methods.")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated random seeds.")
    parser.add_argument("--data_seed", type=int, default=0,
                        help="Seed used when materializing datasets.")
    parser.add_argument("--dataset_root", type=str, default=None)

    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--no-precompute", action="store_false", dest="precompute",
                        help="Disable dataset materialization (uses transforms per access).")
    parser.set_defaults(precompute=True)

    parser.add_argument("--scan_variant", type=str, default="PB_T50_RS")
    parser.add_argument("--scan_split_dir", type=str, default="main_split")
    parser.add_argument("--scan_resample", type=str, default="irregular",
                        choices=["irregular", "take_first"])
    parser.add_argument("--scan_bias", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_step_size", type=int, default=50)
    parser.add_argument("--pool_ratio", type=float, default=0.5)
    parser.add_argument("--diffpool_layers", type=int, default=4,
                        help="Number of DiffPool layers (>=4 to use pooled features).")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--selection_metric", type=str, default="acc",
                        choices=["acc", "loss"])
    parser.add_argument("--folds", type=int, default=10)

    parser.add_argument("--lacore_epsilon", type=float, default=None)
    parser.add_argument("--lacore_target_ratio", type=float, default=None)
    parser.add_argument("--lacore_min_size", type=int, default=None)
    parser.add_argument("--lacore_max_clusters", type=int, default=None)

    args = parser.parse_args()
    args.datasets = parse_csv_list(args.datasets) or ["ModelNet40", "ScanObjectNN"]
    args.methods = parse_csv_list(args.methods) or POOLING_METHODS
    args.seeds = parse_seeds(args.seeds, default=range(1, 11))

    dataset_config = build_dataset_config(args)
    results: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for dataset_name in args.datasets:
        results[dataset_name] = {}
        for method in args.methods:
            if method not in POOLING_METHODS:
                raise ValueError(f"Unknown method: {method}")
            print(f"=== {dataset_name} | {method} ===")
            dataset = load_dataset(dataset_name, args, dataset_config, method)
            mean, std = run_method_on_dataset(method, dataset_name, dataset, args)
            results[dataset_name][method] = (mean, std)
            print(f"{dataset_name} | {method}: {mean*100:.2f}% ± {std*100:.2f}%")

    print("\nSummary (mean ± std):")
    for dataset_name in args.datasets:
        print(f"{dataset_name}:")
        for method in args.methods:
            mean, std = results[dataset_name][method]
            print(f"  {method}: {mean*100:.2f}% ± {std*100:.2f}%")


if __name__ == "__main__":
    main()

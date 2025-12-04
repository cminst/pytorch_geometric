import argparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from graph_classif_utils import Accumulator, LaCoreWrapped, seed_all, parse_seed_list
try:
    import wandb
except ImportError:
    wandb = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb_run(run_name: str, config: dict):
    if wandb is None:
        raise ImportError("wandb is required unless --save_only is set. Install wandb or rerun with --save_only.")
    return wandb.init(project="lacore_pooling", name=run_name, config=config, group="Vanilla-Exp")


def wandb_log(run, metrics: dict, prefix: Optional[str] = None, step: Optional[int] = None):
    if run is None:
        return
    payload = metrics if prefix is None else {f"{prefix}/{k}": v for k, v in metrics.items()}
    run.log(payload, step=step)


def make_modelnet40_graph_datasets(root: str, num_points: int = 1024, k: int = 16, precompute: bool = True):
    """
    Build ModelNet40 train/test datasets as kNN graphs with x = pos.
    If precompute is True, materialize the transformed graphs once to avoid per-epoch
    KNN/SamplePoints overhead (mirrors the LaCore script behavior).
    """

    class PosToX(object):
        def __call__(self, data):
            data.x = data.pos
            return data

    transform = T.Compose(
        [
            T.SamplePoints(num_points),
            T.NormalizeScale(),
            T.KNNGraph(k=k, force_undirected=True),
            PosToX(),
        ]
    )

    train_raw = ModelNet(root=root, name="40", train=True, transform=transform)
    test_raw = ModelNet(root=root, name="40", train=False, transform=transform)

    if not precompute:
        return train_raw, test_raw

    print("Precomputing ModelNet graphs once (avoids per-epoch transforms)...")
    train_ds = LaCoreWrapped([train_raw[i] for i in range(len(train_raw))])
    test_ds = LaCoreWrapped([test_raw[i] for i in range(len(test_raw))])
    return train_ds, test_ds


def stratified_train_val_split(labels: torch.Tensor, val_ratio: float, seed: int):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1).")
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, stratify=labels.cpu().numpy(), random_state=seed)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long)


class VanillaGCNNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout

        self.lin = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, num_classes),
        )

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x0 = F.relu(self.conv1(x, ei))
        x0 = F.dropout(x0, p=self.dropout, training=self.training)

        g0_mean = global_mean_pool(x0, batch)
        g0_max = global_max_pool(x0, batch)

        x1 = F.relu(self.conv2(x0, ei))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        g1_mean = global_mean_pool(x1, batch)
        g1_max = global_max_pool(x1, batch)

        g = torch.cat([g0_mean, g0_max, g1_mean, g1_max], dim=-1)
        out = self.lin(g)
        return out


def run_fold(
    train_loader,
    val_loader,
    test_loader,
    in_dim,
    num_classes,
    hidden=64,
    lr=1e-3,
    wd=5e-4,
    epochs=500,
    dropout=0.2,
    log_progress=True,
    wandb_run=None,
    log_prefix: Optional[str] = None,
    step_offset: int = 0,
):
    model = VanillaGCNNet(in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0.0
    best_epoch = 0
    final_acc = Accumulator()
    final_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            out = model(data)
            loss = cross_entropy(out, data.y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs

        model.eval()

        def _eval(loader, compute_loss=False):
            correct = 0
            total = 0
            total_loss_val = 0.0
            with torch.no_grad():
                for d in loader:
                    d = d.to(device)
                    logits = model(d)
                    if compute_loss:
                        loss = cross_entropy(logits, d.y)
                        total_loss_val += loss.item() * d.num_graphs

                    pred = logits.argmax(dim=-1)
                    correct += int((pred == d.y).sum())
                    total += d.num_graphs

            acc = correct / total
            return acc, total_loss_val / len(loader.dataset) if compute_loss else acc

        train_acc, _ = _eval(train_loader)
        val_acc, val_loss = _eval(val_loader, compute_loss=True)
        test_acc, _ = _eval(test_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            final_acc.update_best(test_acc)
            final_metrics = {"train_acc": train_acc, "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc, "epoch": epoch}

        if log_progress and (epoch % 20 == 0 or epoch == 1):
            print(
                f"[{epoch:03d}] loss={total_loss/len(train_loader.dataset):.4f} "
                f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}% test_acc={test_acc*100:.2f}% "
                f"(test_at_best_val={final_acc.value*100:.2f}%)"
            )
        wandb_log(
            wandb_run,
            {
                "epoch": step_offset + epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "test_acc": test_acc,
            },
            prefix=log_prefix,
            step=step_offset + epoch,
        )

    wandb_log(
        wandb_run,
        {
            "test_acc@val": final_acc.value,
            "test_acc@val%": 100 * final_acc.value,
            "best_epoch": best_epoch,
        },
        prefix=log_prefix,
        step=step_offset + epochs,
    )

    return final_acc.value, best_epoch, final_metrics


def run_modelnet_experiment(
    dataset_root: str = "data/ModelNet",
    dataset_name: str = "ModelNet40",
    num_points: int = 1024,
    knn_k: int = 16,
    seed: int = 42,
    batch_size: int = 32,
    hidden: int = 64,
    lr: float = 0.0001,
    wd: float = 0.001,
    epochs: int = 300,
    dropout: float = 0.2,
    val_ratio: float = 0.1,
    precompute_graphs: bool = True,
    save_only: bool = False,
):
    """
    Run a vanilla 2-layer GCN on ModelNet40 (canonical split) with no LaCore pooling.
    """
    config_dict = locals()
    config = edict(config_dict)

    print("Using configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nLoading ModelNet40 (train/test splits)...")
    train_dataset, test_dataset = make_modelnet40_graph_datasets(
        config.dataset_root, num_points=config.num_points, k=config.knn_k, precompute=config.precompute_graphs
    )
    print(f"Train graphs: {len(train_dataset)} | Test graphs: {len(test_dataset)}")

    num_classes = int(train_dataset.num_classes)
    assert train_dataset.num_features > 0, "ModelNet graphs must carry node features."
    in_dim = train_dataset.num_features

    labels = torch.tensor([data.y.item() for data in train_dataset])
    seed_values = parse_seed_list(config.seed)

    results = []
    for seed_value in seed_values:
        seed_all(seed_value)
        run = None
        if not config.save_only:
            wandb_config = dict(config)
            wandb_config.update({"seed": seed_value, "in_dim": in_dim, "num_classes": num_classes})
            run = init_wandb_run(run_name=f"Vanilla-seed{seed_value}", config=wandb_config)

        train_idx, val_idx = stratified_train_val_split(labels, config.val_ratio, seed_value)

        train_loader = DataLoader(train_dataset.index_select(train_idx), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset.index_select(val_idx), batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        best, best_epoch, metrics = run_fold(
            train_loader,
            val_loader,
            test_loader,
            in_dim,
            num_classes,
            hidden=config.hidden,
            lr=config.lr,
            wd=config.wd,
            epochs=config.epochs,
            dropout=config.dropout,
            log_progress=len(seed_values) == 1,
            wandb_run=run,
        )
        results.append(best)

        print(f"\nSeed {seed_value}: test_acc@val={100*best:.2f}% at epoch {best_epoch}")
        print(f"  train_acc={100*metrics.get('train_acc', 0):.2f}% val_acc={100*metrics.get('val_acc', 0):.2f}%")

        if run is not None:
            wandb_log(
                run,
                {
                    "final_seed_test_acc@val": best,
                    "final_seed_test_acc@val%": 100 * best,
                    "best_epoch": best_epoch,
                },
                step=config.epochs + 1,
            )
            run.finish()

    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results)) if len(results) > 1 else 0.0
    print(f"\nTest accuracy (mean over seeds): {100*mean_acc:.2f}% ± {100*std_acc:.2f}%")
    return mean_acc, std_acc


def main():
    default_config = {
        "dataset_root": "data/ModelNet",
        "dataset_name": "ModelNet40",
        "num_points": 1024,
        "knn_k": 16,
        "seed": 42,
        "batch_size": 32,
        "hidden": 64,
        "lr": 0.0001,
        "wd": 0.001,
        "epochs": 500,
        "dropout": 0.2,
        "val_ratio": 0.1,
        "save_only": False,
    }

    parser = argparse.ArgumentParser(description="Vanilla GCN on ModelNet40 (canonical split, no LaCore pooling)")

    parser.add_argument("--dataset_root", type=str, default=default_config["dataset_root"], help="Root directory for ModelNet")
    parser.add_argument("--dataset_name", type=str, default=default_config["dataset_name"], help="Dataset name (kept for logging)")
    parser.add_argument("--num_points", type=int, default=default_config["num_points"], help="Number of points sampled per mesh")
    parser.add_argument("--knn_k", type=int, default=default_config["knn_k"], help="k for KNN graph construction")
    parser.add_argument("--seed", type=str, default=str(default_config["seed"]), help="Random seed(s), comma-separated for multiple")
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"], help="Batch size for training")
    parser.add_argument("--hidden", type=int, default=default_config["hidden"], help="Hidden dimension for GCN layers")
    parser.add_argument("--lr", type=float, default=default_config["lr"], help="Learning rate")
    parser.add_argument("--wd", type=float, default=default_config["wd"], help="Weight decay")
    parser.add_argument("--epochs", type=int, default=default_config["epochs"], help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=default_config["dropout"], help="Dropout probability")
    parser.add_argument("--val_ratio", type=float, default=default_config["val_ratio"], help="Validation ratio taken from the train split")
    parser.add_argument("--save_only", action="store_true", help="Skip WandB logging")

    args = parser.parse_args()

    config_dict = {key: getattr(args, key) for key in default_config.keys()}

    run_modelnet_experiment(**config_dict)


if __name__ == "__main__":
    main()

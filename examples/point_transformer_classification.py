import argparse
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter

_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
_BENCH_POINTS = osp.join(_ROOT, 'benchmark', 'points')
if _BENCH_POINTS not in sys.path:
    sys.path.insert(0, _BENCH_POINTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.custom_datasets import ScanObjectNN
from utils.transforms import IrregularResample, RandomIrregularResample

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ModelNet10',
    choices=['ModelNet10', 'ModelNet40', 'ScanObjectNN'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument(
    '--train_mode',
    type=str,
    default='clean',
    choices=['clean', 'aug'],
    help='Training transform: clean or aug (IrregularResample beta~U[0,4]).',
)

NUM_POINTS = 1024
TRAIN_AUG_DENSE_POINTS = 2048
TRAIN_AUG_MAX_BIAS = 4.0
STRESS_DENSE_POINTS = 2048
STRESS_BETAS = [0, 1, 2, 3, 4, 5]
STRESS_SEEDS = list(range(1, 11))
RUN_STRESS_EVAL = True
SCAN_VARIANT = 'PB_T50_RS'
SCAN_SPLIT_DIR = 'main_split'

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _preserve_rng_state(fn):
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        return fn()
    finally:
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.random.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

def _modelnet_variant(name: str) -> str:
    return '10' if name == 'ModelNet10' else '40'

def _build_transforms(dataset_name: str, train_mode: str):
    if dataset_name in {'ModelNet10', 'ModelNet40'}:
        if train_mode == 'aug':
            train_transform = T.Compose([
                T.SamplePoints(TRAIN_AUG_DENSE_POINTS),
                RandomIrregularResample(
                    num_points=NUM_POINTS,
                    max_bias=TRAIN_AUG_MAX_BIAS,
                ),
            ])
        else:
            train_transform = T.Compose([T.SamplePoints(NUM_POINTS)])
        test_transform = T.Compose([T.SamplePoints(NUM_POINTS)])
        return train_transform, test_transform

    if train_mode == 'aug':
        train_transform = T.Compose([
            RandomIrregularResample(
                num_points=NUM_POINTS,
                max_bias=TRAIN_AUG_MAX_BIAS,
            ),
        ])
    else:
        train_transform = T.Compose([
            IrregularResample(num_points=NUM_POINTS, bias_strength=0.0),
        ])
    test_transform = T.Compose([
        IrregularResample(num_points=NUM_POINTS, bias_strength=0.0),
    ])
    return train_transform, test_transform


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return F.log_softmax(out, dim=-1)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def _build_stress_loader(root, dataset_name, beta, batch_size):
    if dataset_name in {'ModelNet10', 'ModelNet40'}:
        stress_transform = T.Compose([
            T.SamplePoints(STRESS_DENSE_POINTS),
            IrregularResample(num_points=NUM_POINTS, bias_strength=float(beta)),
        ])
        stress_dataset = ModelNet(
            root,
            _modelnet_variant(dataset_name),
            False,
            stress_transform,
            pre_transform=T.NormalizeScale(),
        )
    else:
        stress_transform = T.Compose([
            IrregularResample(num_points=NUM_POINTS, bias_strength=float(beta)),
        ])
        stress_dataset = ScanObjectNN(
            root=root,
            train=False,
            transform=stress_transform,
            pre_transform=T.NormalizeScale(),
            variant=SCAN_VARIANT,
            split_dir=SCAN_SPLIT_DIR,
        )
    return DataLoader(stress_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0)

def eval_stress_sweep(root, dataset_name, batch_size):
    results = {}
    for beta in STRESS_BETAS:
        loader = _build_stress_loader(root, dataset_name, beta, batch_size)
        accs = []
        for seed in STRESS_SEEDS:
            def _run():
                _seed_all(seed)
                return test(loader)
            acc = float(_preserve_rng_state(_run))
            accs.append(acc)
            print(f"stress beta={beta:.1f} seed={seed} acc={acc:.4f}")
        mean = float(np.mean(accs)) if accs else 0.0
        std = float(np.std(accs)) if accs else 0.0
        results[beta] = (mean, std)
        print(f"stress beta={beta:.1f} mean={mean:.4f} std={std:.4f}")
    return results


if __name__ == '__main__':
    args = parser.parse_args()

    num_epochs = args.epochs
    num_workers = args.num_workers
    batch_size = args.batch_size
    root = osp.join(args.dataset_dir, args.dataset)
    pre_transform = T.NormalizeScale()
    train_transform, test_transform = _build_transforms(
        args.dataset,
        args.train_mode,
    )
    if args.dataset in {'ModelNet10', 'ModelNet40'}:
        variant = _modelnet_variant(args.dataset)
        train_dataset = ModelNet(
            root,
            variant,
            True,
            train_transform,
            pre_transform,
        )
        test_dataset = ModelNet(
            root,
            variant,
            False,
            test_transform,
            pre_transform,
        )
    else:
        train_dataset = ScanObjectNN(
            root=root,
            train=True,
            transform=train_transform,
            pre_transform=pre_transform,
            variant=SCAN_VARIANT,
            split_dir=SCAN_SPLIT_DIR,
        )
        test_dataset = ScanObjectNN(
            root=root,
            train=False,
            transform=test_transform,
            pre_transform=pre_transform,
            variant=SCAN_VARIANT,
            split_dir=SCAN_SPLIT_DIR,
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(0, train_dataset.num_classes,
                dim_model=[32, 64, 128, 256, 512], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.5)

    for epoch in range(1, num_epochs + 1):
        loss = train()
        test_acc = test(test_loader)
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
        scheduler.step()

    if RUN_STRESS_EVAL:
        eval_stress_sweep(root, args.dataset, batch_size=batch_size)

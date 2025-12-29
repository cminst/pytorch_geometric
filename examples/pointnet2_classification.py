import argparse
import os.path as osp
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
_BENCH_POINTS = osp.join(_ROOT, 'benchmark', 'points')
if _BENCH_POINTS not in sys.path:
    sys.path.insert(0, _BENCH_POINTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.transforms import IrregularResample

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ModelNet10',
    choices=['ModelNet10', 'ModelNet40'],
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
parser.add_argument('--epochs', type=int, default=201)
NUM_POINTS = 1024
STRESS_DENSE_POINTS = 2048
STRESS_BETAS = [0, 1, 2, 3, 4, 5]
STRESS_SEEDS = list(range(1, 11))
RUN_STRESS_EVAL = True

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

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class Net(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)

def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def _build_stress_loader(root, variant, beta, batch_size):
    stress_transform = T.Compose([
        T.SamplePoints(STRESS_DENSE_POINTS),
        IrregularResample(num_points=NUM_POINTS, bias_strength=float(beta)),
    ])
    stress_dataset = ModelNet(root, variant, False, stress_transform,
                              pre_transform=T.NormalizeScale())
    return DataLoader(stress_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0)

def eval_stress_sweep(root, variant, batch_size):
    results = {}
    for beta in STRESS_BETAS:
        loader = _build_stress_loader(root, variant, beta, batch_size)
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

    variant = '10' if args.dataset == 'ModelNet10' else '40'

    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(NUM_POINTS)
    train_dataset = ModelNet(root, variant, True, transform, pre_transform)
    test_dataset = ModelNet(root, variant, False, transform, pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs):
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')

    if RUN_STRESS_EVAL:
        eval_stress_sweep(root, variant, batch_size=batch_size)

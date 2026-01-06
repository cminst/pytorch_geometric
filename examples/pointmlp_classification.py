import argparse
import importlib.util
import os.path as osp
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
_BENCH_POINTS = osp.join(_ROOT, 'benchmark', 'points')
_POINTMLP_ROOT = osp.join(_ROOT, 'pointMLP-pytorch')
_POINTMLP_MODELS = osp.join(_POINTMLP_ROOT, 'classification_ModelNet40')
_POINTNET2_OPS = osp.join(_POINTMLP_ROOT, 'pointnet2_ops_lib')
for _path in [_POINTNET2_OPS, _POINTMLP_MODELS]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
for _path in [_ROOT, _BENCH_POINTS]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

_BENCH_TRANSFORMS = osp.join(_ROOT, 'benchmark', 'points', 'utils',
                             'transforms.py')
if not osp.isfile(_BENCH_TRANSFORMS):
    raise FileNotFoundError(
        f"Missing benchmark transforms at {_BENCH_TRANSFORMS}."
    )


def _load_irregular_resample():
    spec = importlib.util.spec_from_file_location("bench_transforms",
                                                  _BENCH_TRANSFORMS)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load benchmark transforms module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bench_transforms = _load_irregular_resample()
IrregularResample = _bench_transforms.IrregularResample
RandomIrregularResample = _bench_transforms.RandomIrregularResample

from utils.custom_datasets import ScanObjectNN

try:
    import pointnet2_ops  # noqa: F401
except Exception as exc:
    raise ImportError(
        "PointMLP requires the pointnet2_ops extension. "
        "Build it from pointMLP-pytorch/pointnet2_ops_lib before running."
    ) from exc

try:
    from models import pointMLP
except Exception as exc:
    raise ImportError(
        "Failed to import PointMLP from pointMLP-pytorch/classification_ModelNet40."
    ) from exc

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


def _to_dense_pos(data):
    pos, mask = to_dense_batch(
        data.pos,
        data.batch,
        max_num_nodes=NUM_POINTS,
    )
    if not bool(mask.all()):
        raise ValueError(
            f"Expected {NUM_POINTS} points per example, "
            f"but got a smaller point cloud in the batch."
        )
    return pos.permute(0, 2, 1).contiguous()


class Net(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.model = pointMLP(num_classes=out_channels)

    def forward(self, data):
        x = _to_dense_pos(data)
        return self.model(x).log_softmax(dim=-1)


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


def _modelnet_variant(name: str) -> str:
    return '10' if name == 'ModelNet10' else '40'


def _load_dataset(dataset_name, root, train, transform, pre_transform):
    if dataset_name in {'ModelNet10', 'ModelNet40'}:
        return ModelNet(
            root,
            _modelnet_variant(dataset_name),
            train,
            transform,
            pre_transform,
        )
    if dataset_name == 'ScanObjectNN':
        return ScanObjectNN(
            root=root,
            train=train,
            transform=transform,
            pre_transform=pre_transform,
            variant=SCAN_VARIANT,
            split_dir=SCAN_SPLIT_DIR,
        )
    raise ValueError(f'Unknown dataset: {dataset_name}')


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


def _build_stress_loader(root, dataset_name, beta, batch_size):
    if dataset_name in {'ModelNet10', 'ModelNet40'}:
        stress_transform = T.Compose([
            T.SamplePoints(STRESS_DENSE_POINTS),
            IrregularResample(num_points=NUM_POINTS, bias_strength=float(beta)),
        ])
    else:
        stress_transform = T.Compose([
            IrregularResample(num_points=NUM_POINTS, bias_strength=float(beta)),
        ])
    stress_dataset = _load_dataset(
        dataset_name,
        root,
        False,
        stress_transform,
        pre_transform=T.NormalizeScale(),
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
    train_dataset = _load_dataset(
        args.dataset,
        root,
        True,
        train_transform,
        pre_transform,
    )
    test_dataset = _load_dataset(
        args.dataset,
        root,
        False,
        test_transform,
        pre_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}')

    if RUN_STRESS_EVAL:
        eval_stress_sweep(root, args.dataset, batch_size=batch_size)

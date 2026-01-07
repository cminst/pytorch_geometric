import argparse
import os.path as osp
import random
import sys

import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import MedShapeNet, ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
_BENCH_POINTS = osp.join(_ROOT, 'benchmark', 'points')
if _BENCH_POINTS not in sys.path:
    sys.path.insert(0, _BENCH_POINTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.custom_datasets import ScanObjectNN
from utils.transforms import IrregularResample, RandomIrregularResample

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ModelNet10',
    choices=['ModelNet10', 'ModelNet40', 'MedShapeNet', 'ScanObjectNN'],
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

args = parser.parse_args()

num_epochs = args.epochs
num_workers = args.num_workers
batch_size = args.batch_size
root = osp.join(args.dataset_dir, args.dataset)

print('The root is: ', root)

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

    if dataset_name == 'ScanObjectNN':
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

    train_transform = T.Compose([T.SamplePoints(NUM_POINTS)])
    test_transform = T.Compose([T.SamplePoints(NUM_POINTS)])
    return train_transform, test_transform

pre_transform = T.NormalizeScale()
train_transform, test_transform = _build_transforms(
    args.dataset,
    args.train_mode,
)

print('The Dataset is: ', args.dataset)
if args.dataset == 'ModelNet40':
    print('Loading training data')
    train_dataset = ModelNet(root, '40', True, train_transform, pre_transform)
    print('Loading test data')
    test_dataset = ModelNet(root, '40', False, test_transform, pre_transform)
elif args.dataset == 'MedShapeNet':
    print('Loading dataset')
    dataset = MedShapeNet(root=root, size=50, pre_transform=pre_transform,
                          transform=train_transform, force_reload=False)

    random.seed(42)

    train_indices = []
    test_indices = []
    for label in range(dataset.num_classes):
        by_class = [
            i for i, data in enumerate(dataset) if int(data.y) == label
        ]
        random.shuffle(by_class)

        split_point = int(0.7 * len(by_class))
        train_indices.extend(by_class[:split_point])
        test_indices.extend(by_class[split_point:])

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

elif args.dataset == 'ModelNet10':
    print('Loading training data')
    train_dataset = ModelNet(root, '10', True, train_transform, pre_transform)
    print('Loading test data')
    test_dataset = ModelNet(root, '10', False, test_transform, pre_transform)

elif args.dataset == 'ScanObjectNN':
    print('Loading training data')
    train_dataset = ScanObjectNN(
        root=root,
        train=True,
        transform=train_transform,
        pre_transform=pre_transform,
        variant=SCAN_VARIANT,
        split_dir=SCAN_SPLIT_DIR,
    )
    print('Loading test data')
    test_dataset = ScanObjectNN(
        root=root,
        train=False,
        transform=test_transform,
        pre_transform=pre_transform,
        variant=SCAN_VARIANT,
        split_dir=SCAN_SPLIT_DIR,
    )

else:
    raise ValueError(
        f"Unknown dataset name '{args.dataset}'. "
        f"Available options: 'ModelNet10', 'ModelNet40', 'MedShapeNet', "
        f"'ScanObjectNN'.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)

print('Running model')

class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(train_dataset.num_classes, k=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
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
    elif dataset_name == 'ScanObjectNN':
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
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
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

for epoch in range(1, num_epochs + 1):
    loss = train()
    test_acc = test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    scheduler.step()

if RUN_STRESS_EVAL and args.dataset in {"ModelNet10", "ModelNet40", "ScanObjectNN"}:
    eval_stress_sweep(root, args.dataset, batch_size)

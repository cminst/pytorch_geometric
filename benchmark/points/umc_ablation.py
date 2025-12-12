import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, BaseTransform, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
import argparse
import os
import numpy as np


class BiasedSamplePoints(BaseTransform):
    """Sample points biased towards a random focus point."""
    def __init__(self, num_points):
        self.num_points = num_points

    def forward(self, data):
        pos = data.pos
        num_nodes = pos.size(0)

        # Random focus point
        focus_idx = np.random.randint(num_nodes)
        focus_point = pos[focus_idx]

        # Inverse-distance weighting
        dists = torch.norm(pos - focus_point, dim=1)
        weights = 1.0 / (dists + 0.05).pow(2)
        weights = weights / weights.sum()

        # Sample indices
        choice = torch.multinomial(weights, self.num_points, replacement=True)

        data.pos = pos[choice]
        data.batch = torch.zeros(self.num_points, dtype=torch.long)

        # Preserve label if present
        if data.y is not None:
            data.y = data.y

        return data


class ComputeSpectralConfig(BaseTransform):
    """Compute spectral configuration and optionally apply UMC weighting."""
    def __init__(self, K=64, use_umc=True, steps=100, lr=0.01):
        self.K = K
        self.use_umc = use_umc
        self.steps = steps
        self.lr = lr

    def solve_umc(self, phi):
        """Optimize diagonal weighting to approximate identity."""
        N, K = phi.shape
        device = phi.device
        phi = phi.detach()

        w = torch.ones(N, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.lr)
        I_K = torch.eye(K, device=device)

        for _ in range(self.steps):
            optimizer.zero_grad()
            W_mat = torch.diag(torch.relu(w))
            gram = phi.T @ W_mat @ phi
            loss = torch.norm(gram - I_K) ** 2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                w.clamp_(min=1e-4)
        return w.detach()

    def forward(self, data):
        # Compute random-walk Laplacian for geometry preservation
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='rw', num_nodes=data.num_nodes)

        # Compute symmetric Laplacian for stable eigen-decomposition
        edge_index_sym, edge_weight_sym = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L_sym = to_dense_adj(edge_index_sym, edge_attr=edge_weight_sym, max_num_nodes=data.num_nodes).squeeze(0)

        try:
            _, evecs = torch.linalg.eigh(L_sym)
            phi = evecs[:, :self.K]
        except Exception:
            phi = torch.zeros(data.num_nodes, self.K, device=data.pos.device)

        w = self.solve_umc(phi) if self.use_umc else torch.ones(data.num_nodes)

        data.phi = phi
        data.umc_weights = w
        return data


class SpectralProjectionNet(nn.Module):
    """Spectral projection network."""
    def __init__(self, in_channels, num_classes, K):
        super().__init__()
        self.K = K
        self.spectral_filter = nn.Parameter(torch.ones(1, K, in_channels))

        self.mlp = nn.Sequential(
            nn.Linear(K * in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, phi, w, batch_size):
        B = batch_size
        N = x.shape[0] // B
        C = x.shape[1]

        x = x.view(B, N, C)
        phi = phi.view(B, N, self.K)
        w = w.view(B, N)

        weighted_x = x * w.unsqueeze(-1)

        f_hat = torch.bmm(phi.transpose(1, 2), weighted_x)
        y = f_hat * self.spectral_filter
        y = torch.abs(y)  # Sign invariance
        y = y.view(B, -1)
        return F.log_softmax(self.mlp(y), dim=1)


def run_experiment(use_umc, K, batch_size, epochs):
    """Run training and evaluation with given hyperparameters."""
    print("\n==========================================")
    print(f"STARTING RUN: UMC = {use_umc}")
    print("==========================================")

    train_transform = Compose([
        SamplePoints(1024),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=K, use_umc=use_umc)
    ])

    test_clean_transform = train_transform

    test_corrupt_transform = Compose([
        BiasedSamplePoints(1024),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=K, use_umc=use_umc)
    ])

    dataset_root = os.path.join('..', 'data', 'ModelNet_Robustness')

    train_dataset = ModelNet(dataset_root, '10', train=True, transform=train_transform)
    test_clean_ds = ModelNet(dataset_root, '10', train=False, transform=test_clean_transform)
    test_corrupt_ds = ModelNet(dataset_root, '10', train=False, transform=test_corrupt_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    clean_loader = DataLoader(test_clean_ds, batch_size=batch_size, shuffle=False)
    corrupt_loader = DataLoader(test_corrupt_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralProjectionNet(in_channels=3, num_classes=10, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.pos, data.phi, data.umc_weights, data.num_graphs)
            loss = F.nll_loss(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.3f}")

    model.eval()

    def evaluate(loader, name):
        correct = 0
        total = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                out = model(data.pos, data.phi, data.umc_weights, data.num_graphs)
            pred = out.max(1)[1]
            correct += pred.eq(data.y.squeeze()).sum().item()
            total += data.num_graphs
        acc = correct / total
        print(f"--> {name} Accuracy: {acc*100:.2f}%")
        return acc

    acc_clean = evaluate(clean_loader, "CLEAN (Uniform)")
    acc_corrupt = evaluate(corrupt_loader, "CORRUPT (Biased)")

    drop = acc_clean - acc_corrupt
    print(f"Performance Drop: {drop*100:.2f}%")
    return acc_clean, acc_corrupt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UMC ablation experiment')
    parser.add_argument('--K', type=int, default=32, help='Number of eigenvectors to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    naive_clean, naive_corrupt = run_experiment(use_umc=False, K=args.K, batch_size=args.batch_size, epochs=args.epochs)
    umc_clean, umc_corrupt = run_experiment(use_umc=True, K=args.K, batch_size=args.batch_size, epochs=args.epochs)

    print("\n\n================ FINAL RESULTS ================")
    print(f"Naive: Clean {naive_clean*100:.1f}% -> Corrupt {naive_corrupt*100:.1f}% (Drop: {(naive_clean-naive_corrupt)*100:.1f}%)")
    print(f"UMC  : Clean {umc_clean*100:.1f}% -> Corrupt {umc_corrupt*100:.1f}% (Drop: {(umc_clean-umc_corrupt)*100:.1f}%)")
    print("===============================================")

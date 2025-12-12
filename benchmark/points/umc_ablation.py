import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, BaseTransform, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
import argparse
import os
import shutil
import numpy as np

# --- 1. Custom Transform: Biased Sampling ---
class BiasedSamplePoints(BaseTransform):
    def __init__(self, num_points):
        self.num_points = num_points

    def forward(self, data):
        pos = data.pos
        num_nodes = pos.size(0)

        # Pick a random "Sensor Focus" point from the existing points
        focus_idx = np.random.randint(num_nodes)
        focus_point = pos[focus_idx]

        # Calculate distance to focus
        dists = torch.norm(pos - focus_point, dim=1)

        # Probability is inverse distance (Higher prob near focus)
        # Add epsilon to avoid division by zero
        # Power of 2 makes the bias stronger (very dense cluster)
        weights = 1.0 / (dists + 0.05).pow(2)

        # Normalize to probability
        weights = weights / weights.sum()

        # Sample indices based on weights
        choice = torch.multinomial(weights, self.num_points, replacement=True)

        data.pos = pos[choice]
        data.batch = torch.zeros(self.num_points, dtype=torch.long)

        if data.y is not None:
            data.y = data.y

        return data

# --- 2. UMC Logic ---
class ComputeSpectralConfig(BaseTransform):
    def __init__(self, K=64, use_umc=True, steps=100, lr=0.01):
        self.K = K
        self.use_umc = use_umc
        self.steps = steps
        self.lr = lr

    def solve_umc(self, phi):
        N, K = phi.shape
        device = phi.device

        # Detach phi - we don't need gradients w.r.t. eigenvectors
        phi = phi.detach()

        # Force enable gradients for this optimization, even if called inside torch.no_grad()
        with torch.enable_grad():
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
        # Use RW for geometry preservation
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='rw', num_nodes=data.num_nodes)

        # Use Sym for solver stability
        edge_index_sym, edge_weight_sym = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L_sym = to_dense_adj(edge_index_sym, edge_attr=edge_weight_sym, max_num_nodes=data.num_nodes).squeeze(0)

        try:
            evals, evecs = torch.linalg.eigh(L_sym)
            phi = evecs[:, :self.K]
        except:
            # Fallback for degenerate graphs
            phi = torch.zeros(data.num_nodes, self.K, device=data.pos.device)

        if self.use_umc:
            w = self.solve_umc(phi)
        else:
            w = torch.ones(data.num_nodes)

        data.phi = phi
        data.umc_weights = w
        return data

# --- 3. Network ---
class SpectralProjectionNet(nn.Module):
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

        w_expanded = w.unsqueeze(-1)
        weighted_x = x * w_expanded

        f_hat = torch.bmm(phi.transpose(1, 2), weighted_x)
        y = f_hat * self.spectral_filter
        y = torch.abs(y) # Sign Invariance
        y = y.view(B, -1)
        return F.log_softmax(self.mlp(y), dim=1)

def run_experiment(use_umc):
    K = 32 # Lower K for speed/stability
    BATCH_SIZE = 32
    EPOCHS = 20 # Short run

    print(f"\n==========================================")
    print(f"STARTING RUN: UMC = {use_umc}")
    print(f"==========================================")

    # 1. Clean Data (Uniform Sampling) for Training
    # Note: We put SamplePoints in 'transform' so it resamples every epoch (Data Augmentation)
    # This might slow down training but reduces overfitting
    train_transform = Compose([
        SamplePoints(1024),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=K, use_umc=use_umc)
    ])

    # 2. Corrupted Data (Biased Sampling) for Testing
    test_clean_transform = train_transform

    test_corrupt_transform = Compose([
        BiasedSamplePoints(1024), # <--- THE STRESS TEST
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=K, use_umc=use_umc)
    ])

    dataset_root = os.path.join('data', 'ModelNet_Robustness')
    if os.path.exists(dataset_root): shutil.rmtree(dataset_root) # Force fresh start

    # Load Datasets
    # We use 'pre_transform=None' and put everything in 'transform'
    # so we can have different transforms for train/test
    train_dataset = ModelNet(dataset_root, '10', train=True, transform=train_transform)
    test_clean_ds = ModelNet(dataset_root, '10', train=False, transform=test_clean_transform)
    test_corrupt_ds = ModelNet(dataset_root, '10', train=False, transform=test_corrupt_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    clean_loader = DataLoader(test_clean_ds, batch_size=BATCH_SIZE, shuffle=False)
    corrupt_loader = DataLoader(test_corrupt_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralProjectionNet(in_channels=3, num_classes=10, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(1, EPOCHS + 1):
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

    # Evaluate
    model.eval()

    def evaluate(loader, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
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
    # Run Naive
    naive_clean, naive_corrupt = run_experiment(use_umc=False)

    # Run UMC
    umc_clean, umc_corrupt = run_experiment(use_umc=True)

    print("\n\n================ FINAL RESULTS ================")
    print(f"Naive: Clean {naive_clean*100:.1f}% -> Corrupt {naive_corrupt*100:.1f}% (Drop: {(naive_clean-naive_corrupt)*100:.1f}%)")
    print(f"UMC  : Clean {umc_clean*100:.1f}% -> Corrupt {umc_corrupt*100:.1f}% (Drop: {(umc_clean-umc_corrupt)*100:.1f}%)")
    print("===============================================")

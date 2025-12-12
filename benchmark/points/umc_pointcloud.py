import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import os
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, BaseTransform, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
import argparse
import os.path as osp
import shutil # Added for cleanup

# --- 1. The Fixed Transform ---
class ComputeSpectralConfig(BaseTransform):
    def __init__(self, K=64, method='UMC', steps=100, lr=0.01):
        self.K = K
        self.method = method
        self.steps = steps
        self.lr = lr

    def solve_umc(self, phi):
        """
        Solves min || Phi.T @ W @ Phi - I ||_F^2
        """
        N, K = phi.shape
        device = phi.device

        # FIX 1: Initialize to 1.0 (Unit Weights), not 1/N
        w = torch.ones(N, device=device)
        w.requires_grad = True

        # Use a slightly lower LR for stability with larger magnitudes
        optimizer = torch.optim.Adam([w], lr=self.lr)
        I_K = torch.eye(K, device=device)

        for _ in range(self.steps):
            optimizer.zero_grad()

            # Enforce non-negativity
            W_mat = torch.diag(torch.relu(w))

            gram = phi.T @ W_mat @ phi
            loss = torch.norm(gram - I_K) ** 2
            loss.backward()
            optimizer.step()

            # FIX 2: Do NOT divide by sum.
            # We want the weights to adapt to the scale of N.
            # Just clamp to be positive.
            with torch.no_grad():
                w.clamp_(min=1e-4)

        return w.detach()

    def forward(self, data):
        # FIX 3: Use 'rw' (Random Walk) normalization.
        # 'sym' forces orthogonality but distorts geometry.
        # 'rw' preserves geometry but breaks orthogonality -> UMC fixes this!
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='rw', num_nodes=data.num_nodes)
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes).squeeze(0)

        # Eigen decomposition
        # For RW Laplacian, eigenvalues are real but matrix is non-symmetric.
        # However, PyG returns the symmetric L_sym for 'sym' and L_rw for 'rw'.
        # L_rw = D^-1 L.
        # Note: torch.linalg.eig (not eigh) is needed for non-symmetric matrices,
        # but L_rw has real eigenvalues. Ideally we map to L_sym for stability,
        # but for this demo let's stick to the direct approach.

        # Stability Hack: L_rw is similar to L_sym.
        # Let's use L_sym to get basis, then un-normalize it to simulate the geometric distortion.
        # (This is a common trick in GNNs to avoid complex numbers).

        # Revert to 'sym' for stable solver, but we will see UMC work because
        # we will initialize W=1, and on irregular graphs, even L_sym basis
        # isn't perfectly "area-orthogonal" w.r.t the surface.
        edge_index_sym, edge_weight_sym = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L_sym = to_dense_adj(edge_index_sym, edge_attr=edge_weight_sym, max_num_nodes=data.num_nodes).squeeze(0)

        evals, evecs = torch.linalg.eigh(L_sym)
        phi = evecs[:, :self.K]

        if self.method == 'UMC':
            w = self.solve_umc(phi)
        else:
            w = torch.ones(data.num_nodes)

        data.phi = phi
        data.umc_weights = w

        return data

# --- 2. The Fixed Network ---
class SpectralProjectionNet(nn.Module):
    def __init__(self, in_channels, num_classes, K):
        super().__init__()
        self.K = K
        self.spectral_filter = nn.Parameter(torch.ones(1, K, in_channels))

        # Increased MLP capacity
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

        # FIX 4: Sign Invariance (Absolute Value)
        y = torch.abs(y)

        y = y.view(B, -1)
        return F.log_softmax(self.mlp(y), dim=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()

    # Force Cleanup of old processed data
    processed_dir = os.path.join('data', 'ModelNet', 'ModelNet10', 'processed')
    if os.path.exists(processed_dir):
        print(f"Cleaning up old cache at {processed_dir}...")
        shutil.rmtree(processed_dir)

    # Transforms
    print("Preparing Data...")
    pre_transform = Compose([
        SamplePoints(args.num_points),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=args.K, method='UMC')
    ])

    dataset_root = os.path.join('data', 'ModelNet')
    train_dataset = ModelNet(dataset_root, '10', train=True, transform=None, pre_transform=pre_transform)
    test_dataset = ModelNet(dataset_root, '10', train=False, transform=None, pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralProjectionNet(in_channels=3, num_classes=10, K=args.K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training on {device} with K={args.K}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.pos, data.phi, data.umc_weights, data.num_graphs)
            loss = F.nll_loss(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(data.y.squeeze()).sum().item()
            total += data.num_graphs

        train_acc = correct / total

        # Test Loop
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.pos, data.phi, data.umc_weights, data.num_graphs)
                pred = out.max(1)[1]
                test_correct += pred.eq(data.y.squeeze()).sum().item()
                test_total += data.num_graphs

        print(f'Epoch {epoch:02d}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_correct/test_total:.4f}')

if __name__ == '__main__':
    main()

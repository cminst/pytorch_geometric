import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, BaseTransform, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj
import argparse
import os.path as osp

# Precompute UMC weights
class ComputeSpectralConfig(BaseTransform):
    def __init__(self, K=64, method='UMC', steps=100, lr=0.1):
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

        # Init weights uniformly
        w = torch.ones(N, device=device) / N
        w.requires_grad = True
        optimizer = torch.optim.Adam([w], lr=self.lr)
        I_K = torch.eye(K, device=device)

        # Optimization loop
        for _ in range(self.steps):
            optimizer.zero_grad()
            # Enforce non-negativity proxy during training
            W_diag = torch.relu(w)
            W_mat = torch.diag(W_diag)

            # Gram matrix: Phi^T W Phi
            gram = phi.T @ W_mat @ phi
            loss = torch.norm(gram - I_K) ** 2
            loss.backward()
            optimizer.step()

            # Project to simplex (sum = 1, non-negative)
            with torch.no_grad():
                w.clamp_(min=1e-6) # Avoid true zero for stability
                w.div_(w.sum())

        return w.detach()

    def __call__(self, data):
        # 1. Compute Laplacian and Eigenvectors
        # L = D - A (normalized)
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes).squeeze(0)

        # Eigen decomposition (Hermitian)
        # Note: This is O(N^3), which is why we do it in PreTransform!
        evals, evecs = torch.linalg.eigh(L)

        # Take first K low-freq modes
        # evecs are sorted ascending by eigenvalue
        phi = evecs[:, :self.K] # (N, K)

        # 2. Compute UMC Weights
        if self.method == 'UMC':
            w = self.solve_umc(phi)
        else:
            w = torch.ones(data.num_nodes) / data.num_nodes

        # Save to data object
        data.phi = phi
        data.umc_weights = w

        return data

# Model Architecture
class SpectralProjectionNet(nn.Module):
    def __init__(self, in_channels, num_classes, K):
        super().__init__()
        self.K = K

        # A learnable spectral filter (diagonal)
        # We process each input channel (x,y,z) independently in frequency domain
        self.spectral_filter = nn.Parameter(torch.ones(1, K, in_channels))

        # MLP to classify the spectral features
        # Input size: K frequencies * 3 channels
        self.mlp = nn.Sequential(
            nn.Linear(K * in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, phi, w, batch_size):
        # x: (Batch*N, C)
        # phi: (Batch*N, K)
        # w: (Batch*N, )

        # Reshape to (Batch, N, ...)
        # Assuming fixed N=1024 due to SamplePoints transform
        B = batch_size
        N = x.shape[0] // B
        C = x.shape[1]

        x = x.view(B, N, C)
        phi = phi.view(B, N, self.K)
        w = w.view(B, N)

        # Create Diagonal Weight Matrix W
        # We can just broadcast multiply instead of full diag matrix construction
        # w_x: (B, N, C) weighting the input features
        w_expanded = w.unsqueeze(-1) # (B, N, 1)

        # 1. Projection: f_hat = Phi^T * (W * x)
        # (B, K, N) @ (B, N, C) -> (B, K, C)
        weighted_x = x * w_expanded
        f_hat = torch.bmm(phi.transpose(1, 2), weighted_x)

        # 2. Spectral Filtering
        # Element-wise multiplication in frequency domain
        y = f_hat * self.spectral_filter

        # 3. Flatten and Classify
        y = y.view(B, -1) # (B, K*C)
        return F.log_softmax(self.mlp(y), dim=1)

# Training

def parse_args():
    parser = argparse.ArgumentParser(description='UMC Point Cloud Benchmark')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory for datasets (default: benchmark/data/ModelNet)')
    parser.add_argument('--K', type=int, default=64,
                        help='Number of spectral components')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from each point cloud')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--knn_k', type=int, default=20,
                        help='Number of neighbors for KNN graph')
    return parser.parse_args()


def main():
    args = parse_args()

    K = args.K
    NUM_POINTS = args.num_points
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    KNN_K = args.knn_k

    # Determine dataset root (same logic as benchmark/kernel/datasets.py)
    default_root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ModelNet')
    dataset_root = args.dataset_root if args.dataset_root else default_root

    # Transforms
    print("Preparing Data... (This may take a moment for eigen decomposition)")
    pre_transform = Compose([
        SamplePoints(NUM_POINTS),
        NormalizeScale(),
        KNNGraph(k=KNN_K),
        ComputeSpectralConfig(K=K, method='UMC')
    ])

    train_dataset = ModelNet(dataset_root, '10', train=True, transform=None,
                            pre_transform=pre_transform)
    test_dataset = ModelNet(dataset_root, '10', train=False, transform=None,
                           pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralProjectionNet(in_channels=3, num_classes=10, K=K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {device} with K={K}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            x = data.pos
            out = model(x, data.phi, data.umc_weights, data.num_graphs)
            loss = F.nll_loss(out, data.y.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.max(1)[1]
            correct += pred.eq(data.y.squeeze()).sum().item()
            total += data.num_graphs

        train_acc = correct / total
        print(f'Epoch {epoch:02d}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')

        if epoch % 5 == 0:
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
            print(f'---> Test Acc: {test_correct/test_total:.4f}')

if __name__ == '__main__':
    main()

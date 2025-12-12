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

class ComputeSpectralConfig(BaseTransform):
    def __init__(self, K=64, use_umc=True, steps=100, lr=0.01):
        self.K = K
        self.use_umc = use_umc
        self.steps = steps
        self.lr = lr

    def solve_umc(self, phi):
        N, K = phi.shape
        device = phi.device
        w = torch.ones(N, device=device)
        w.requires_grad = True
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
        # Use RW normalization to preserve geometry
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='rw', num_nodes=data.num_nodes)
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes).squeeze(0)

        # Use Sym for stability in eigendecomposition, then project back
        edge_index_sym, edge_weight_sym = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L_sym = to_dense_adj(edge_index_sym, edge_attr=edge_weight_sym, max_num_nodes=data.num_nodes).squeeze(0)

        evals, evecs = torch.linalg.eigh(L_sym)
        phi = evecs[:, :self.K]

        if self.use_umc:
            w = self.solve_umc(phi)
        else:
            # NAIVE BASELINE: Uniform weights
            w = torch.ones(data.num_nodes)

        data.phi = phi
        data.umc_weights = w
        return data

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
            nn.Dropout(0.5), # Added extra dropout
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

        # Sign Invariance
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
    parser.add_argument('--no_umc', action='store_true', help="Disable UMC (Use uniform weights)")
    parser.add_argument('--use_const_feature', action='store_true', help="Use constant 1s instead of XYZ as input")
    return parser.parse_args()

def main():
    args = parse_args()

    # Force Cleanup
    processed_dir = os.path.join('data', 'ModelNet', 'ModelNet10', 'processed')
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    use_umc = not args.no_umc
    print(f"Preparing Data... Mode: {'UMC' if use_umc else 'NAIVE (Uniform)'}")

    pre_transform = Compose([
        SamplePoints(args.num_points),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=args.K, use_umc=use_umc)
    ])

    dataset_root = os.path.join('data', 'ModelNet')
    train_dataset = ModelNet(dataset_root, '10', train=True, transform=None, pre_transform=pre_transform)
    test_dataset = ModelNet(dataset_root, '10', train=False, transform=None, pre_transform=pre_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Input channels: 3 if using XYZ, 1 if using Constant
    in_channels = 1 if args.use_const_feature else 3

    model = SpectralProjectionNet(in_channels=in_channels, num_classes=10, K=args.K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4) # Added weight decay

    print(f"Training on {device} with K={args.K}, Features={'Const' if args.use_const_feature else 'XYZ'}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Feature Selection
            if args.use_const_feature:
                x = torch.ones((data.pos.shape[0], 1), device=device)
            else:
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

        if epoch % 1 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    if args.use_const_feature:
                        x = torch.ones((data.pos.shape[0], 1), device=device)
                    else:
                        x = data.pos
                    out = model(x, data.phi, data.umc_weights, data.num_graphs)
                    pred = out.max(1)[1]
                    test_correct += pred.eq(data.y.squeeze()).sum().item()
                    test_total += data.num_graphs

            print(f'Epoch {epoch:02d}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_correct/test_total:.4f}')

if __name__ == '__main__':
    main()

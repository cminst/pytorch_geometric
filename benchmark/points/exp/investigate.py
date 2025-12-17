import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import shutil
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import Compose, SamplePoints, KNNGraph, BaseTransform, NormalizeScale
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj

# --- 1. Re-use your UMC Logic ---
class ComputeSpectralConfig(BaseTransform):
    def __init__(self, K=16, method='UMC', steps=200, lr=0.1):
        self.K = K
        self.method = method
        self.steps = steps
        self.lr = lr

    def solve_umc(self, phi):
        N, K = phi.shape
        device = phi.device
        w = torch.ones(N, device=device) / N
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
                w.clamp_(min=1e-6)
                w.div_(w.sum())
        return w.detach()

    def forward(self, data):
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=data.num_nodes)
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes).squeeze(0)
        evals, evecs = torch.linalg.eigh(L)
        phi = evecs[:, :self.K]
        
        if self.method == 'UMC':
            w = self.solve_umc(phi)
        else:
            w = torch.ones(data.num_nodes) / data.num_nodes
            
        data.phi = phi
        data.umc_weights = w
        return data

# --- 2. The Investigation Logic ---

def investigate():
    K = 16 # Keep K small for visualization clarity
    NUM_POINTS = 1024
    
    # FORCE CLEANUP to avoid caching bugs
    dataset_path = 'data/ModelNet_Debug'
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    
    print(f"Generating diagnostic plots for K={K}...")
    
    # 1. Setup Data with UMC
    pre_transform = Compose([
        SamplePoints(NUM_POINTS),
        NormalizeScale(),
        KNNGraph(k=20),
        ComputeSpectralConfig(K=K, method='UMC', steps=200)
    ])
    
    # We only need the test set (unseen data)
    dataset = ModelNet(dataset_path, '10', train=False, transform=None, pre_transform=pre_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get one sample
    data = next(iter(loader))
    
    # Extract vars
    phi = data.phi.squeeze(0)      # (N, K)
    w_umc = data.umc_weights.squeeze(0) # (N,)
    w_unif = torch.ones_like(w_umc) / NUM_POINTS
    
    # --- EXPERIMENT A: The Gram Matrix (Orthogonality Check) ---
    # Ideally, Phi^T * W * Phi should be Identity
    
    # 1. Naive (Uniform Weights)
    gram_naive = phi.T @ torch.diag(w_unif) @ phi
    # 2. UMC (Learned Weights)
    gram_umc = phi.T @ torch.diag(w_umc) @ phi
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(gram_naive.abs().numpy(), ax=axes[0], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f"Naive Gram Matrix\nOff-Diag Error: {torch.norm(gram_naive - torch.eye(K)):.4f}")
    
    sns.heatmap(gram_umc.abs().numpy(), ax=axes[1], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"UMC Gram Matrix\nOff-Diag Error: {torch.norm(gram_umc - torch.eye(K)):.4f}")
    
    plt.tight_layout()
    plt.savefig('investigation_1_orthogonality.png')
    print("Saved investigation_1_orthogonality.png")
    
    # --- EXPERIMENT B: Weight Distribution ---
    # Does UMC actually change anything? Or is it just staying near 1/N?
    
    plt.figure(figsize=(10, 5))
    w_values = w_umc.numpy() * NUM_POINTS # Scale so 1.0 = uniform average
    plt.hist(w_values, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', label='Uniform Weight')
    plt.title("Distribution of Learned UMC Weights (Scaled)")
    plt.xlabel("Weight Multiplier (1.0 = Average)")
    plt.ylabel("Count of Points")
    plt.legend()
    plt.savefig('investigation_2_weights.png')
    print("Saved investigation_2_weights.png")

    # --- EXPERIMENT C: Visualizing the "Loud" Points ---
    # Let's see which points get down-weighted.
    # Low weight = Dense area (usually). High weight = Sparse area.
    
    pos = data.pos.numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by their learned weight
    p = ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=w_values, cmap='coolwarm', s=20)
    fig.colorbar(p, label='UMC Weight (Red=High/Sparse, Blue=Low/Dense)')
    ax.set_title("Point Cloud Colored by UMC Weight")
    plt.savefig('investigation_3_geometry.png')
    print("Saved investigation_3_geometry.png")

if __name__ == '__main__':
    investigate()

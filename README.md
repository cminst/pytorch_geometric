<p align="center">
  <img height="150" src="https://via.placeholder.com/400x150/4F8BF9/FFFFFF?text=PyTorch+Geometric+Logo" alt="PyTorch Geometric Logo" />
</p>

<h1 align="center">PyTorch Geometric</h1>

<p align="center">
  Geometric deep learning with PyG, with UMC, PointWavelet, and LaCore pooling
</p>

______________________________________________________________________

<div align="center">

[![License](https://img.shields.io/github/license/pyg-team/pytorch_geometric)](LICENSE)

</div>

PyTorch Geometric is a library built upon [PyTorch](https://pytorch.org/) to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. This fork focuses on advanced geometric deep learning techniques including UMC, PointWavelet (a stronger backbone for 3D point cloud processing), and LaCore pooling.

- `benchmark/points/`: Implementation of UMC and PointWavelet methods for 3D point cloud processing
- `benchmark/kernel/`: Implementation of LaCore pooling methods
- `torch_geometric/`: Core PyTorch Geometric library components

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy, SciPy, scikit-learn

### Installation

```bash
pip install torch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric
```

## Usage Examples

### UMC and PointWavelet

3D point cloud processing with UMC and PointWavelet implementations in the `benchmark/points/` directory:

```bash
# Example command for running UMC experiments
python3 run_all_umc_experiments.py --root ../data/ModelNet10 --num_points 512 --dense_points 2048 --train_mode clean --methods umc --lambda_ortho_grid 0 --no_cache_eval

# Example command for PointWavelet
python3 umc_pointwavelet.py --data_root ../data/ModelNet10/ --num_workers 4 --wf_learnable --modelnet 10 --lr 1e-4 --methods umc --epochs 100 --stress_beta 5
```

### LaCore Pooling

LaCore pooling methods are implemented in the `benchmark/kernel/` directory for advanced graph neural network pooling operations.

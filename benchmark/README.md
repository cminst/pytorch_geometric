# PyG Benchmark Suite

This benchmark suite provides evaluation scripts for **[semi-supervised node classification](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/citation)**, **[graph classification](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/kernel)**, and **[point cloud classification](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/points)** and **[runtimes](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/runtime)** in order to compare various methods in homogeneous evaluation scenarios.
In particular, we take care to avoid to perform hyperparameter and model selection on the test set and instead use an additional validation set.

## Installation

```
$ pip install -e .
```

## PointWavelet UMC Examples

From `benchmark/points`, train and save a checkpoint:

```
python3 umc_pointwavelet.py --data_root ../data/ModelNet10/ --modelnet 10 --methods umc --save_ckpt --ckpt_dir checkpoints
```

Evaluate stress accuracy across multiple betas using a saved checkpoint:

```
python3 umc_pointwavelet_stress_eval.py --ckpt checkpoints/pointwavelet_umc_modelnet10_n1024_seed0.pt --betas 1,2,3,4 --data_root ../data/ModelNet10
```

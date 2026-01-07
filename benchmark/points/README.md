# Point Cloud classification

Evaluation scripts for various methods on the ModelNet10 dataset:

- **[MPNN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/mpnn.py)**: `python mpnn.py`
- **[PointNet++](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/point_net.py)**: `python point_net.py`
- **[EdgeCNN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/edge_cnn.py)**: `python edge_cnn.py`
- **[SplineCNN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/spline_cnn.py)**: `python spline_cnn.py`
- **[PointCNN](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/points/point_cnn.py)**: `python point_cnn.py`

## PointWavelet UMC Examples

From `./`, train and save a checkpoint:

```
python3 umc_pointwavelet.py \
    --data_root ../data/ModelNet10/ \
    --num_workers 4 \
    --wf_learnable \
    --modelnet 10 \
    --lr 1e-4 \
    --methods umc \
    --epochs 100 \
    --stress_beta 5 \
    --seeds 1 \
    --save_ckpt \
    --ckpt_dir checkpoints
```

Evaluate stress accuracy across multiple betas using a saved checkpoint:

```
python3 umc_pointwavelet_stress_eval.py --ckpt checkpoints/pointwavelet_umc_modelnet10_n1024_seed0.pt --betas 1,2,3,4 --data_root ../data/ModelNet10
```

## UMC Ablation Grid (ModelNet10/40 + ScanObjectNN)

Run the UMC-only ablation grid across datasets with selectable feature sets:

```
python benchmark/points/run_umc_ablation_grid.py \
  --datasets ModelNet10,ModelNet40,ScanObjectNN \
  --umc_configs full,no_coords,md_only,deg_only \
  --degree_features log
```

UMC ablation configs:

- `full`: `[x_i, md_i, log md_i, log deg_i]`
- `no_coords`: `[md_i, log md_i, log deg_i]` (rotation-invariant)
- `md_only`: `[md_i, log md_i]`
- `deg_only`: `[log deg_i]` (or `[deg_i, log deg_i]` with `--degree_features both`)

You can limit runs to a single dataset with `--dataset ModelNet10` and choose subsets of configs with `--umc_configs`.

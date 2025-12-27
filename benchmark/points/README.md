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

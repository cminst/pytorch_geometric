# UMC Integration in PointWavelet

This document describes exactly how Universal Measure Correction (UMC) is wired into the PointWavelet implementation in this repo, with pointers to the concrete code paths. The conceptual UMC layer is described in `UMC.tex`, and the WaveletFormer/PointWavelet backbone is described in `PointWavelet.tex`. This note focuses on the *integration*: where UMC is inserted in the PointWavelet pipeline and how its weights are computed/used.

## 1) Where UMC lives in the codebase

UMC is implemented inside the WaveletFormer layer and exposed through the PointWavelet model configs:

- UMC weight network + weighting logic: `pointwavelet/layers/waveletformer.py`
- Config pass-through from PointWavelet models: `pointwavelet/models/pointwavelet_cls.py` (and `pointwavelet/models/pointwavelet_partseg.py` for segmentation)
- Set-abstraction wrapper that instantiates WaveletFormer: `pointwavelet/layers/pointnet2_modules.py`
- Training/CLI that enables UMC and logs diagnostics: `umc_pointwavelet.py`

In short: **UMC is a per-patch weighting layer inside each WaveletFormer block** attached to the PointNet++ set abstraction (SA) stages.

## 2) How UMC is threaded through the model

### Config flow (from CLI to WaveletFormer)

`umc_pointwavelet.py` builds the model with UMC knobs and passes them all the way down:

- CLI args: `--umc_hidden`, `--umc_knn`, `--umc_min_weight`, `--umc_no_inverse`
- `build_model()` sets in `PointWaveletClsConfig`:
  - `wf_use_umc`, `wf_umc_hidden`, `wf_umc_knn`, `wf_umc_min_weight`, `wf_umc_use_inverse`
- `PointWaveletClassifier` forwards these into each SA block
- `PointNetSetAbstractionWavelet` passes them into `WaveletFormerConfig`

Result: **all four WaveletFormer layers (SA1–SA4) use UMC when enabled**.

### Placement in the pipeline

Within each SA block (see `pointwavelet/layers/pointnet2_modules.py`):

1. Sample centroids via FPS and group kNN neighborhoods.
2. Run pointwise MLP on each local patch.
3. **Apply WaveletFormer to each patch**.
4. Max-pool over the k neighbors to produce the centroid feature.

UMC lives inside **step 3**, i.e., inside WaveletFormer.

## 3) UMC mechanics inside WaveletFormer

### 3.1 Patch inputs

WaveletFormer receives local patches as:

- `x`: `(P, n, C)` where `P = B * npoint` and `n = k` (neighbors)
- `xyz`: `(P, n, 3)` local coordinates, centered at the centroid

`xyz` is *always* passed in from the SA block; it is required when UMC is enabled.

### 3.2 Weight prediction (learned quadrature)

UMC predicts a **nonnegative weight per node in each patch**. The steps in
`pointwavelet/layers/waveletformer.py` are:

1. Compute pairwise distances within the patch.
2. For each node, compute mean distance to its `umc_knn` nearest neighbors:
   - `md_i = mean_kNN_distance(i)`
3. Build a 6D geometric feature per node:

   `g_i = [x_i, y_i, z_i, r_i, md_i, log(md_i)]`

   where `r_i = ||xyz_i||`.
4. Run a small MLP with Softplus output (`UMCWeightNet`) to get weights:

   `w_i = psi_theta(g_i) >= 0`

5. Normalize weights within each patch so that `sum_i w_i = n`.
6. Clamp weights by `umc_min_weight` to avoid zeros/instability.

Notes vs `UMC.tex`:
- The paper’s descriptor uses `[x_i, md_i, log(md_i), log(deg_i)]`.
- The implementation uses **`r_i` instead of `log(deg_i)`**, and does not compute `deg_i`.
- The normalization is **per patch** (size `n = k`), not over an entire shape.

### 3.3 Where the weights are applied

UMC changes the spectral transform **at the vertex-domain input**. In
`WaveletFormer.forward()`:

1. If enabled, apply weights to the node features:

   `x <- w ⊙ x`

2. Proceed with the standard wavelet pipeline:
   - spectral projection
   - multi-scale wavelet filters
   - transformer across scales
   - pseudoinverse reconstruction

3. If `umc_use_inverse=True`, *undo the weighting* after reconstruction:

   `out <- out / w`

This corresponds to the UMC formulation in `UMC.tex` where coefficients are
computed as `Psi_s (W x)` and reconstruction uses an additional `W^{-1}` term.

### 3.4 Storage for diagnostics

UMC stores the latest weights and mean distances:

- `self.last_umc_raw`: tensors used for training regularizers
- `self.last_umc`: detached tensors for logging

These are read by training utilities in `umc_pointwavelet.py`.

## 4) Training integration (umc_pointwavelet.py)

### Enabling UMC

- `--methods umc` or `--methods both` enables a UMC variant.
- `build_model(use_umc=True, ...)` turns on `wf_use_umc`.

### Additional UMC-specific losses

`umc_pointwavelet.py` adds an optional correlation regularizer (not the
orthogonality regularizer from `UMC.tex`):

- `_umc_corr_reg_loss()` computes Pearson corr between `w` and `mean_dist`.
- Training adds `umc_corr_reg * corr` to the loss.
- With a positive `umc_corr_reg`, minimizing the loss encourages **negative
  correlation** between weights and mean distance (i.e., upweight sparse regions).

### Diagnostics / logging

`_collect_umc_stats()` uses `last_umc` to log:

- variance/std/min/max of weights
- correlation with mean distance and inverse mean distance

These are printed per epoch and optionally logged to W&B.

## 5) Default hyperparameters (CLI defaults)

These are the **exact defaults** used by `umc_pointwavelet.py` when no CLI flags
are provided (i.e., "use all default CLI args").

### Data + loader

- `--data_root=data`
- `--modelnet=40`
- `--force_reload=False`
- `--num_points=1024`
- `--batch_size=16`
- `--num_workers=4`

### Optimization + schedule

- `--epochs=200`
- `--lr=1e-3`
- `--weight_decay=1e-4`
- `--lr_step=20`
- `--lr_gamma=0.7`
- `--amp=False` (only enables CUDA AMP when explicitly set)
- `--seeds=0` (single run)

### Which methods run

- `--methods=both` (runs PointWavelet and PointWavelet+UMC)
- `--wf_learnable=True` (PointWavelet-L)

### WaveletFormer backbone (from model defaults)

- SA hierarchy: `(npoint, k, mlp)` = `(512,32,[64,64,128])`, `(128,32,[128,128,256])`, `(32,32,[256,256,512])`, `(1,32,[512,512,512])`
- Wavelet scales: `J=5`, `scales=None` (uses default dyadic scales)
- Transformer in WaveletFormer: `depth=2`, `heads=4`
- Wavelet: Mexican hat, `sigma_mode="mean"`
- Learnable basis regularizer: `beta=0.05` (active only for learnable basis)

### UMC settings

- `--umc_hidden=128,32` (UMC MLP hidden widths)
- `--umc_knn=20`
- `--umc_min_weight=1e-4`
- `--umc_no_inverse=False` (so `umc_use_inverse=True`)
- `--umc_corr_reg=0.0` (correlation regularizer disabled by default)

### Diagnostics / stress eval

- `--umc_stats_batches=10`
- `--stress_beta=2.0`
- `--stress_interval=10` (stress eval enabled every 10 epochs)
- `--stress_dense_points=2048`

### Logging / outputs

- `--wandb=True`
- `--save_ckpt=False`
- `--ckpt_dir=checkpoints`
- `--out_csv=pointwavelet_umc_results.csv`

## 6) Summary of the plug-in points

UMC modifies PointWavelet *only* inside each WaveletFormer:

- **Input weighting:** `x <- w ⊙ x` before spectral transforms.
- **Optional inverse correction:** `out <- out / w` after reconstruction.
- **Weight prediction:** local geometric MLP per patch using `xyz`, `r`, and mean kNN distance.

All other components (PointNet++ SA layout, WaveletFormer transform, transformer
across scales, classifier head) are unchanged. UMC is thus a **drop-in
quadrature correction** inside the existing spectral pipeline.

## 7) Key file references

- `pointwavelet/layers/waveletformer.py` (UMCWeightNet, weighting, inverse correction)
- `pointwavelet/layers/pointnet2_modules.py` (passes `xyz` into WaveletFormer)
- `pointwavelet/models/pointwavelet_cls.py` (config wiring for all SA blocks)
- `umc_pointwavelet.py` (CLI, enablement, regularizer, logging)

# PointWavelet (PyTorch, from-paper implementation)

This is a **from-paper** PyTorch implementation of **PointWavelet / PointWavelet-L** (WaveletFormer + PointNet++ hierarchy), based on the paper.

What is implemented:

- **Graph wavelet transform** (Mexican hat wavelets):
  - scaling function `h(x) = exp(-x^4)`
  - wavelet function `g(x) = x * exp(-x)`
- **WaveletFormer layer**:
  1. multi-scale wavelet filtering
  2. Transformer encoder across *scales* (scales = tokens)
  3. reconstruction via the **pseudoinverse** (frame-based inverse) as described in the paper
- Two variants:
  - **PointWavelet**: eigendecomposition per local patch (slow, but direct)
  - **PointWavelet-L**: **learnable orthogonal basis** `U` constructed from a learnable vector `q` (Eq. 13-14) and learned eigenvalues `lambda`
- **PointNet++-style** backbone:
  - SA1: (512, k=32, C=128)
  - SA2: (128, k=32, C=256)
  - SA3: ( 32, k=32, C=512)
  - SA4: (  1, k=32, C=512)
  - WaveletFormer is applied **inside** each SA layer on each local patch (k nodes), then max-pooled.

## Quick usage (graph classification)

```python
import torch
from pointwavelet import PointWaveletClassifier, PointWaveletClsConfig

model = PointWaveletClassifier(
    PointWaveletClsConfig(
        num_classes=40,
        wf_learnable=True,   # PointWavelet-L (fast)
        wf_J=5,
    )
)

xyz = torch.randn(2, 1024, 3)
logits = model(xyz)  # (2, 40)
reg = model.regularization_loss()  # beta * sum_i ||q_eps||_1
```

## Notes about wavelet scales

The paper does not explicitly list the numeric scale values `{s1,...,sJ}`.
By default this implementation uses **dyadic scales**: `[1, 2, 4, ..., 2^(J-1)]`.

You can override scales via `wf_scales=[...]` in the model config.

## Performance notes

- The eigendecomposition version (`wf_learnable=False`) is **very slow**, because it does `torch.linalg.eigh`
  per local patch.
- The learnable version (`wf_learnable=True`) is much faster and is the practical choice.
- For real training runs you likely want to run on GPU.

## Files

- `pointwavelet/layers/waveletformer.py` – WaveletFormer layer
- `pointwavelet/layers/graph_wavelet.py` – Laplacian construction + learnable spectral basis
- `pointwavelet/layers/pointnet2_modules.py` – SA+WF and FP layers
- `pointwavelet/models/pointwavelet_cls.py` – classification network
- `pointwavelet/models/pointwavelet_partseg.py` – part segmentation network

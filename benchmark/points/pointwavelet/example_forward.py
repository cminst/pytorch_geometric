import os
import sys
import torch

# Add the parent directory of this file so that `import pointwavelet` works.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(THIS_DIR))

from pointwavelet import PointWaveletClassifier, PointWaveletClsConfig


def main():
    model = PointWaveletClassifier(
        PointWaveletClsConfig(
            num_classes=40,
            wf_learnable=True,  # PointWavelet-L
            wf_J=5,
        )
    )
    model.eval()

    xyz = torch.randn(1, 1024, 3)
    with torch.no_grad():
        logits = model(xyz)
        print("logits:", logits.shape)
        print("reg_loss:", float(model.regularization_loss()))


if __name__ == "__main__":
    main()

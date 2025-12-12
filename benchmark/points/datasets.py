import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet


DEFAULT_MODELNET_ROOT = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'ModelNet'
)


def get_dataset(num_points, root=None):
    path = root or DEFAULT_MODELNET_ROOT
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(path, name='10', train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name='10', train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset

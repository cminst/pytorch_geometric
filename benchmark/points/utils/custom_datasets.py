import os
import os.path as osp

import h5py
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import ModelNet
from typing import Optional

DEFAULT_MODELNET_ROOT = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'ModelNet'
)

class ScanObjectNN(InMemoryDataset):
    r"""The ScanObjectNN dataset from the `"ScanObjectNN: A Dataset and
    Benchmark for 3D Object Classification"
    <https://arxiv.org/abs/1908.04616>`_ paper.

    This dataset provides point clouds for several ScanObjectNN variants
    hosted on Hugging Face, including object-only and perturbed background
    settings.

    Args:
        root (str): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        variant (str, optional): The dataset variant to load. Options are
            :obj:`"OBJ_ONLY"`, :obj:`"PB_T25"`, :obj:`"PB_T25_R"`,
            :obj:`"PB_T50_R"`, or :obj:`"PB_T50_RS"`. (default:
            :obj:`"PB_T50_RS"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in
            the final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    base_url = 'https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main'
    default_split_dir = 'main_split_nobg'
    variants = {
        'OBJ_ONLY': 'objectdataset.h5',
        'PB_T25': 'objectdataset_augmented25_norot.h5',
        'PB_T25_R': 'objectdataset_augmented25rot.h5',
        'PB_T50_R': 'objectdataset_augmentedrot.h5',
        'PB_T50_RS': 'objectdataset_augmentedrot_scale75.h5',
    }

    def __init__(self, root, train=True, variant='PB_T50_RS', split_dir: Optional[str] = None, transform=None,
                 pre_transform=None, pre_filter=None, force_reload: bool = False):
        self.train = train
        self.variant = self._canonical_variant(variant)
        self.split_dir = split_dir or self.default_split_dir
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @classmethod
    def _canonical_variant(cls, variant):
        if not isinstance(variant, str):
            raise TypeError('ScanObjectNN variant must be a string')
        key = variant.upper()
        if key not in cls.variants:
            options = ', '.join(sorted(cls.variants.keys()))
            raise ValueError(f'Unknown ScanObjectNN variant "{variant}". '
                             f'Available variants: {options}')
        return key

    @property
    def raw_file_names(self):
        filename = self.variants[self.variant]
        return [
            osp.join(self.split_dir, f'training_{filename}'),
            osp.join(self.split_dir, f'test_{filename}'),
        ]

    @property
    def processed_file_names(self):
        return [
            f'{self.split_dir}_{self.variant}_training.pt',
            f'{self.split_dir}_{self.variant}_test.pt',
        ]

    def download(self):
        raw_split_dir = osp.join(self.raw_dir, self.split_dir)
        os.makedirs(raw_split_dir, exist_ok=True)
        filename = self.variants[self.variant]
        for split in ('training', 'test'):
            url = f'{self.base_url}/{self.split_dir}/{split}_{filename}'
            download_url(url, raw_split_dir)

    def process(self):
        self.save(self.process_set('training'), self.processed_paths[0])
        self.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, split):
        filename = self.variants[self.variant]
        h5_path = osp.join(self.raw_dir, self.split_dir, f'{split}_{filename}')

        with h5py.File(h5_path, 'r') as f:
            data = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64')

        data_list = []
        for i in range(data.shape[0]):
            pos = torch.from_numpy(data[i])
            y = torch.tensor(labels[i]).view(1)

            d = Data(pos=pos, y=y)
            data_list.append(d)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return data_list


if __name__ == '__main__':
    dataset = ScanObjectNN(root='data/ScanObjectNN', train=True)
    print(f'Dataset: {dataset}')
    print(f'First graph: {dataset[0]}')

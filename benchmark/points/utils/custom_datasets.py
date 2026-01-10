import os
import os.path as osp
from typing import Callable, List, Optional

import h5py
import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.datasets import ModelNet
from torch_geometric.io import fs, read_txt_array

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

class S3DISPatched(InMemoryDataset):
    r"""The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
    the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
    <https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf>`_
    paper, containing point clouds of six large-scale indoor parts in three
    buildings with 12 semantic elements (and one clutter class).

    Args:
        root (str): Root directory where the dataset should be saved.
        test_area (int, optional): Which area to use for testing (1-6).
            (default: :obj:`6`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
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
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://huggingface.co/datasets/cminst/S3DIS/resolve/main/indoor3d_sem_seg_hdf5_data.zip'

    def __init__(
        self,
        root: str,
        test_area: int = 6,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert test_area >= 1 and test_area <= 6
        self.test_area = test_area
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['all_files.txt', 'room_filelist.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{split}_{self.test_area}.pt' for split in ['train', 'test']]

    def download(self) -> None:
        zip_path = download_url(self.url, self.root)

        extract_zip(zip_path, self.root)
        os.unlink(zip_path)
        fs.rm(self.raw_dir)

        zip_name = self.url.split('/')[-1]
        extracted_dir = osp.splitext(zip_name)[0]

        os.rename(osp.join(self.root, extracted_dir), self.raw_dir)

    def process(self) -> None:
        with open(self.raw_paths[0]) as f:
            filenames = [x.split('/')[-1] for x in f.read().split('\n')[:-1]]

        with open(self.raw_paths[1]) as f:
            rooms = f.read().split('\n')[:-1]

        xs: List[Tensor] = []
        ys: List[Tensor] = []
        for filename in filenames:
            h5 = h5py.File(osp.join(self.raw_dir, filename))
            xs += torch.from_numpy(h5['data'][:]).unbind(0)
            ys += torch.from_numpy(h5['label'][:]).to(torch.long).unbind(0)

        test_area = f'Area_{self.test_area}'
        train_data_list, test_data_list = [], []
        for i, (x, y) in enumerate(zip(xs, ys)):
            data = Data(pos=x[:, :3], x=x[:, 3:], y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if test_area not in rooms[i]:
                train_data_list.append(data)
            else:
                test_data_list.append(data)

        self.save(train_data_list, self.processed_paths[0])
        self.save(test_data_list, self.processed_paths[1])

class ShapeNetPatched(InMemoryDataset):
    r"""The ShapeNet part level segmentation dataset from the `"A Scalable
    Active Framework for Region Annotation in 3D Shape Collections"
    <http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
    paper, containing about 17,000 3D shape point clouds from 16 shape
    categories.
    Each category is annotated with 2 to 6 parts.

    Args:
        root (str): Root directory where the dataset should be saved.
        categories (str or [str], optional): The category of the CAD models
            (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
            :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
            :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
            :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
            :obj:`"Skateboard"`, :obj:`"Table"`).
            Can be explicitly set to :obj:`None` to load all categories.
            (default: :obj:`None`)
        include_normals (bool, optional): If set to :obj:`False`, will not
            include normal vectors as input features to :obj:`data.x`.
            As a result, :obj:`data.x` will be :obj:`None`.
            (default: :obj:`True`)
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"trainval"`, loads the training and validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"trainval"`)
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
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 16,881
          - ~2,616.2
          - 0
          - 3
          - 50
    """

    url = ('https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/'
           'shapenetcore_partanno_segmentation_benchmark_v0_normal.zip')

    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }

    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

    def __init__(
        self,
        root: str,
        categories: Optional[Union[str, List[str]]] = None,
        include_normals: bool = True,
        split: str = 'trainval',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        if categories is None:
            categories = list(self.category_ids.keys())
        if isinstance(categories, str):
            categories = [categories]
        assert all(category in self.category_ids for category in categories)
        self.categories = categories
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        elif split == 'trainval':
            path = self.processed_paths[3]
        else:
            raise ValueError(f'Split {split} found, but expected either '
                             'train, val, trainval or test')

        self.load(path)

        assert isinstance(self._data, Data)
        self._data.x = self._data.x if include_normals else None

        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50),
                                  dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    @property
    def num_classes(self) -> int:
        return self.y_mask.size(-1)

    @property
    def raw_file_names(self) -> List[str]:
        return list(self.category_ids.values()) + ['train_test_split']

    @property
    def processed_file_names(self) -> List[str]:
        cats = '_'.join([cat[:3].lower() for cat in self.categories])
        return [
            osp.join(f'{cats}_{split}.pt')
            for split in ['train', 'val', 'test', 'trainval']
        ]

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        fs.rm(self.raw_dir)
        name = self.url.split('/')[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process_filenames(self, filenames: List[str]) -> List[Data]:
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in filenames:
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            tensor = read_txt_array(osp.join(self.raw_dir, name))
            pos = tensor[:, :3]
            x = tensor[:, 3:6]
            y = tensor[:, -1].type(torch.long)
            data = Data(pos=pos, x=x, y=y, category=cat_idx[cat])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self) -> None:
        trainval = []
        for i, split in enumerate(['train', 'val', 'test']):
            path = osp.join(self.raw_dir, 'train_test_split',
                            f'shuffled_{split}_file_list.json')
            with open(path) as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + '.txt'
                    for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == 'train' or split == 'val':
                trainval += data_list
            self.save(data_list, self.processed_paths[i])
        self.save(trainval, self.processed_paths[3])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, '
                f'categories={self.categories})')

if __name__ == '__main__':
    dataset = ScanObjectNN(root='data/ScanObjectNN', train=True)
    print(f'Dataset: {dataset}')
    print(f'First graph: {dataset[0]}')

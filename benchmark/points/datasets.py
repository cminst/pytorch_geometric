import os.path as osp
import os
import torch
import h5py
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data

DEFAULT_MODELNET_ROOT = osp.join(
    osp.dirname(osp.realpath(__file__)), '..', 'data', 'ModelNet'
)

class ScanObjectNN(InMemoryDataset):
    url = 'https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/scanobjectnn_PB_T50_RS_h5.zip'

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, force_reload: bool = False):
        self.train = train
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.load(path)

    @property
    def raw_file_names(self):
        return [
            osp.join('main_split', 'training_objectdataset_augmentedrot_scale75.h5'),
            osp.join('main_split', 'test_objectdataset_augmentedrot_scale75.h5')
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        self.save(self.process_set('training'), self.processed_paths[0])
        self.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, split):
        filename = f'{split}_objectdataset_augmentedrot_scale75.h5'

        h5_path = osp.join(self.raw_dir, 'main_split', filename)

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


def get_dataset(num_points, root=None):
    path = root or DEFAULT_MODELNET_ROOT
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(num_points)

    train_dataset = ModelNet(path, name='10', train=True, transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ModelNet(path, name='10', train=False, transform=transform,
                            pre_transform=pre_transform)

    return train_dataset, test_dataset


if __name__ == '__main__':
    dataset = ScanObjectNN(root='data/ScanObjectNN', train=True)
    print(f'Dataset: {dataset}')
    print(f'First graph: {dataset[0]}')

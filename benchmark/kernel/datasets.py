import os.path as osp

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ModelNet
from torch_geometric.utils import degree
from torch_geometric.data import InMemoryDataset


class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def _select_config(dataset_config, name):
    if not isinstance(dataset_config, dict):
        return {}
    if name in dataset_config and isinstance(dataset_config[name], dict):
        return dataset_config[name]
    return dataset_config


def get_dataset(
    name,
    sparse=True,
    cleaned=False,
    extra_transform=None,
    dataset_config=None,
):
    # ModelNet10/40 support: build point-cloud graphs the same way as
    # lacore_3d_pooling.py (SamplePoints -> NormalizeScale -> KNNGraph,
    # with x=pos).
    if name in {'ModelNet40', 'ModelNet10'}:
        num = '40' if name.endswith('40') else '10'
        cfg = _select_config(dataset_config, name)
        canonical_split = cfg.get('canonical_split', True)
        precompute = cfg.get('precompute', True)

        class PosToX:
            def __call__(self, data):
                data.x = data.pos
                return data

        base_transform = T.Compose([
            T.SamplePoints(cfg.get('num_points', 1024)),
            T.NormalizeScale(),
            T.KNNGraph(k=cfg.get('knn_k', 16), force_undirected=True),
            PosToX(),
        ])
        full_transform = base_transform if extra_transform is None else T.Compose([base_transform, extra_transform])

        root = cfg.get(
            'root',
            osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ModelNet'),
        )

        def _materialize(ds):
            data_list = [full_transform(ds[i]) for i in range(len(ds))]

            class _Materialized(InMemoryDataset):
                def __init__(self, dl):
                    super().__init__(root=None)
                    self.data, self.slices = self.collate(dl)

            out = _Materialized(data_list)
            out.transform = None
            return out

        if precompute:
            train_raw = ModelNet(root=root, name=num, train=True, transform=None)
            test_raw = ModelNet(root=root, name=num, train=False, transform=None)
            train_ds = _materialize(train_raw)
            test_ds = _materialize(test_raw)
        else:
            train_ds = ModelNet(root=root, name=num, train=True, transform=full_transform)
            test_ds = ModelNet(root=root, name=num, train=False, transform=full_transform)

        if canonical_split:
            return train_ds, test_ds

        data_list = [train_ds[i] for i in range(len(train_ds))]
        data_list += [test_ds[i] for i in range(len(test_ds))]

        class _ModelNetMerged(InMemoryDataset):
            def __init__(self, dl):
                super().__init__(root=None)
                self.data, self.slices = self.collate(dl)

        dataset = _ModelNetMerged(data_list)
        dataset.transform = None
        return dataset

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset._data.edge_attr = None

    if dataset._data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    if extra_transform is not None:
        # Avoid recomputing costly transforms (e.g., LaCore assignment) on every
        # __getitem__ by materializing them once. TUDataset may not expose
        # `.map` on older PyG versions, so fall back to manual materialization.
        full_transform = extra_transform if dataset.transform is None else T.Compose([dataset.transform, extra_transform])
        if hasattr(dataset, 'map'):
            dataset = dataset.map(full_transform)
            dataset.transform = None
        else:
            data_list = [full_transform(dataset[i]) for i in range(len(dataset))]

            class _Mapped(InMemoryDataset):
                def __init__(self, dl):
                    super().__init__(root=None)
                    self.data, self.slices = self.collate(dl)

            dataset = _Mapped(data_list)

    return dataset

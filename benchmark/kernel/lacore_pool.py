import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from dataclasses import dataclass
from typing import Optional

from graph_classif_utils import compute_lacore_cover_for_graph
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, LaCorePooling


@dataclass
class HyperParams:
    hidden: int = 64
    lr: float = 1e-4
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 500
    weight_decay: float = 1e-3
    epsilon: float = 0.1
    target_ratio: float = 0.25
    min_size: int = 4
    max_clusters: Optional[int] = None


class LaCoreAssignment:
    def __init__(self, epsilon=0.1, target_ratio=0.5, min_size=4, max_clusters=None):
        self.epsilon = epsilon
        self.target_ratio = target_ratio
        self.min_size = min_size
        self.max_clusters = max_clusters

    def __call__(self, data):
        cluster, num_clusters, _ = compute_lacore_cover_for_graph(
            data,
            epsilon=self.epsilon,
            target_ratio=self.target_ratio,
            min_size=self.min_size,
            max_clusters=self.max_clusters,
        )
        data.cluster = cluster
        data.num_clusters = torch.tensor([num_clusters], dtype=torch.long)
        return data


class LaCore(torch.nn.Module):
    # Expose an extra transform so the dataset loader can append it.
    extra_transform = LaCoreAssignment()

    @staticmethod
    def default_hparams(dataset_name: str):
        """
        Return a HyperParams instance with default values, overridden by
        dataset‑specific settings if they exist.
        """
        # Start with the defaults defined in the HyperParams dataclass.
        hp = HyperParams()

        # Dataset‑specific overrides.
        overrides = {
            'DD': {'hidden': 128, 'lr': 1e-4, 'dropout': 0.15,
                   'batch_size': 64, 'epochs': 500},
            'PROTEINS': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.1,
                         'batch_size': 128, 'epochs': 500},
            'ModelNet40': {'batch_size': 32, 'epsilon': 1e3, 'target_ratio': 0.5},
            'ModelNet10': {'batch_size': 32, 'epsilon': 1e3, 'target_ratio': 0.5},
            'NCI1': {'hidden': 256, 'lr': 5e-4, 'dropout': 0.4,
                     'batch_size': 256, 'epochs': 500},
            'NCI109': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.2,
                       'batch_size': 64, 'epochs': 500},
            'FRANKENSTEIN': {'hidden': 128, 'lr': 1e-3, 'dropout': 0.2,
                             'batch_size': 128, 'epochs': 500},
        }

        # Apply any overrides for the given dataset.
        for key, value in overrides.get(dataset_name, {}).items():
            setattr(hp, key, value)

        return hp

    def __init__(self, dataset, num_layers, hidden, dropout=0.2):
        super().__init__()
        assert num_layers >= 2, "LaCore needs to have >= 2 layers!"
        self.conv1 = GCNConv(dataset.num_features, hidden)

        # Remaining layers operate after pooling.
        self.post_pool_convs = torch.nn.ModuleList()
        for _ in range(max(num_layers - 1, 0)):
            self.post_pool_convs.append(GCNConv(hidden, hidden))

        self.pool = LaCorePooling(aggregate='mean')

        self.lin = Sequential(
            Linear(4*hidden, 2*hidden),
            ReLU(),
            Dropout(dropout),
            Linear(2*hidden, dataset.num_classes),
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.post_pool_convs:
            conv.reset_parameters()
        self.pool.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data):
        if not hasattr(data, 'cluster') or not hasattr(data, 'num_clusters'):
            raise ValueError("LaCorePooling requires 'cluster' and 'num_clusters' attributes. "
                             "Ensure LaCoreAssignment transform was applied.")

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        pre_mean = global_mean_pool(x, batch)
        pre_max = global_max_pool(x, batch)

        x, edge_index, _, batch_pooled, _, _ = self.pool(
            x, edge_index, batch, data.cluster, data.num_clusters)

        for conv in self.post_pool_convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        post_mean = global_mean_pool(x, batch_pooled)
        post_max = global_max_pool(x, batch_pooled)

        g = torch.cat([pre_mean, pre_max, post_mean, post_max], dim=-1)
        # Return log-probabilities to match the benchmark training loop,
        # which optimizes with NLL loss.
        out = F.log_softmax(self.lin(g), dim=-1)
        return out

    def __repr__(self):
        return self.__class__.__name__

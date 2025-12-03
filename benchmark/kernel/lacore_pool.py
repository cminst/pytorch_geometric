import torch
import torch.nn.functional as F
from torch.nn import Linear

from graph_classif_utils import compute_lacore_cover_for_graph
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, LaCorePooling


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
        base = {
            'hidden': 64,
            'lr': 1e-4,
            'dropout': 0.2,
            'batch_size': 64,
            'epochs': 500,
            'weight_decay': 1e-3,
            'epsilon': 0.1,
            'target_ratio': 0.25,
            'min_size': 4,
            'max_clusters': None,
        }

        table = {
            'DD': {'hidden': 128, 'lr': 1e-4, 'dropout': 0.15,
                   'batch_size': 64, 'epochs': 500},
            'PROTEINS': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.1,
                         'batch_size': 128, 'epochs': 500},
            'NCI1': {'hidden': 256, 'lr': 5e-4, 'dropout': 0.4,
                     'batch_size': 256, 'epochs': 500},
            'NCI109': {'hidden': 128, 'lr': 5e-4, 'dropout': 0.2,
                       'batch_size': 64, 'epochs': 500},
            'FRANKENSTEIN': {'hidden': 128, 'lr': 1e-3, 'dropout': 0.2,
                             'batch_size': 128, 'epochs': 500},
        }
        overrides = table.get(dataset_name, {})
        merged = {**base, **overrides}
        return merged

    def __init__(self, dataset, num_layers, hidden, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)

        # Remaining layers operate after pooling.
        self.post_pool_convs = torch.nn.ModuleList()
        for _ in range(max(num_layers - 1, 0)):
            self.post_pool_convs.append(GCNConv(hidden, hidden))

        self.pool = LaCorePooling(aggregate='mean')
        # We concatenate pre/post pooled global mean and max: 4 * hidden.
        self.lin1 = Linear(4 * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.post_pool_convs:
            conv.reset_parameters()
        self.pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

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

        x = torch.cat([pre_mean, pre_max, post_mean, post_max], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

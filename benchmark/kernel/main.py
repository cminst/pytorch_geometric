import argparse
from itertools import product

from asap import ASAP
from datasets import get_dataset
from diff_pool import DiffPool
from edge_pool import EdgePool
from gcn import GCN, GCNWithJK
from gin import GIN, GIN0, GIN0WithJK, GINWithJK
from global_attention import GlobalAttentionNet
from graclus import Graclus
from graph_sage import GraphSAGE, GraphSAGEWithJK
from sag_pool import SAGPool
from set2set import Set2SetNet
from sort_pool import SortPool
from top_k import TopK
from lacore_pool import LaCore, LaCoreAssignment
from train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2, 3, 4, 5]
hiddens = [16, 32, 64, 128]
datasets = ['PROTEINS'] #['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']  # , 'COLLAB']
nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    # TopK,
    # SAGPool,
    # DiffPool,
    # EdgePool,
    # GCN,
    # GraphSAGE,
    # GIN0,
    # GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    # SortPool,
    # ASAP,
    LaCore,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    if Net is LaCore:
        print("Overriding hyperparams with LaCore optimal settings")
        params = LaCore.default_hparams(dataset_name)
        print(params)
        layer_grid = [2]  # Script uses one conv pre-pool and one post-pool.
        hidden_grid = [params['hidden']]
        lr = params['lr']
        weight_decay = params['weight_decay']
        epochs = params['epochs']
        batch_size = params['batch_size']
        lr_decay_factor = 1.0  # Script trains with a fixed LR.
        lr_decay_step_size = epochs + 1  # Disable decay.
        dropout = params['dropout']
        extra_transform = LaCoreAssignment(
            epsilon=params.get('epsilon', 0.1),
            target_ratio=params.get('target_ratio', 0.25),
            min_size=params.get('min_size', 4),
            max_clusters=params.get('max_clusters', None),
        )
    else:
        layer_grid = layers
        hidden_grid = hiddens
        lr = args.lr
        weight_decay = 0
        epochs = args.epochs
        batch_size = args.batch_size
        lr_decay_factor = args.lr_decay_factor
        lr_decay_step_size = args.lr_decay_step_size
        dropout = None
        extra_transform = getattr(Net, 'extra_transform', None)

    for num_layers, hidden in product(layer_grid, hidden_grid):
        dataset = get_dataset(
            dataset_name,
            sparse=Net != DiffPool,
            extra_transform=extra_transform,
        )
        model_kwargs = {}
        if Net is LaCore:
            model_kwargs['dropout'] = dropout
        model = Net(dataset, num_layers, hidden, **model_kwargs)
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            folds=10,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_decay_factor=lr_decay_factor,
            lr_decay_step_size=lr_decay_step_size,
            weight_decay=weight_decay,
            logger=None,
            selection_metric='acc' if Net is LaCore else 'loss',
            kfold_seed=42 if Net is LaCore else 12345,
            use_inner_val=(Net is LaCore),
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')

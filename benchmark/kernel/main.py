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
from train_eval import cross_validation_with_val_set, single_split_train_eval

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Logs training stats')
parser.add_argument('--datasets', type=str, default=None,
                    help="Comma-separated dataset names (e.g., 'PROTEINS,ModelNet40').")
parser.add_argument('--modelnet_num_points', type=int, default=1024,
                    help='Number of points to sample per mesh for ModelNet datasets.')
parser.add_argument('--modelnet_knn_k', type=int, default=16,
                    help='k for KNN graph construction on ModelNet datasets.')
parser.add_argument('--modelnet_val_ratio', type=float, default=0.1,
                    help='Validation ratio taken from the ModelNet train split.')
args = parser.parse_args()

layers = [2, 3, 4, 5]
hiddens = [16, 32, 64, 128]

# NOTE: datasets are stored in ../data/DATASET_NAME/DATASET_NAME/[processed|raw]
default_datasets = ['PROTEINS'] # ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']
if args.datasets:
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
else:
    datasets = default_datasets

# Optional per-dataset configuration (currently used for ModelNet).
dataset_config = {}
if any(name.startswith('ModelNet') for name in datasets):
    modelnet_cfg = {
        'num_points': args.modelnet_num_points,
        'knn_k': args.modelnet_knn_k,
        'canonical_split': True,
        'val_ratio': args.modelnet_val_ratio,
    }
    dataset_config['ModelNet40'] = modelnet_cfg
    dataset_config['ModelNet10'] = modelnet_cfg

nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    Graclus,
    TopK,
    # SAGPool,
    # DiffPool,
    # EdgePool,
    GCN,
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
    train_acc = info['train_acc']
    val_acc = info['val_acc']
    val_loss = info['val_loss']
    test_acc = info['test_acc']
    best_test = info.get('test_acc_at_best_val', test_acc)
    best_epoch = info.get('best_epoch', epoch)

    if args.verbose:
        print(f'{fold:02d}/{epoch:03d}: '
            f'Train Acc: {train_acc:.3f}, '
            f'Val Acc: {val_acc:.3f}, '
            f'Val Loss: {val_loss:.4f}, '
            f'Test Acc: {test_acc:.3f}, '
            f'Test@Best Val: {best_test:.3f} (epoch {best_epoch})')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    if Net is LaCore:
        params = LaCore.default_hparams(dataset_name)
        print("Overriding hyperparams with LaCore optimal settings")
        print(params)

        hidden_grid = [params.hidden]
        lr = params.lr
        weight_decay = params.weight_decay
        epochs = params.epochs
        batch_size = params.batch_size
        lr_decay_factor = args.lr_decay_factor
        lr_decay_step_size = args.lr_decay_step_size
        dropout = params.dropout
        extra_transform = LaCoreAssignment(
            epsilon=params.epsilon,
            target_ratio=params.target_ratio,
            min_size=params.min_size,
            max_clusters=params.max_clusters,
        )
    else:
        hidden_grid = hiddens
        lr = args.lr
        weight_decay = 0
        epochs = args.epochs
        batch_size = args.batch_size
        lr_decay_factor = args.lr_decay_factor
        lr_decay_step_size = args.lr_decay_step_size
        dropout = None
        extra_transform = getattr(Net, 'extra_transform', None)

    layer_grid = layers

    dataset = get_dataset(
        dataset_name,
        sparse=Net != DiffPool,
        extra_transform=extra_transform,
        dataset_config=dataset_config,
    )

    for num_layers, hidden in product(layer_grid, hidden_grid):
        use_single_split = isinstance(dataset, tuple)
        if use_single_split:
            train_dataset, test_dataset = dataset
            dataset_for_model = train_dataset
        else:
            dataset_for_model = dataset

        model_kwargs = {}
        if Net is LaCore:
            model_kwargs['dropout'] = dropout
        model = Net(dataset_for_model, num_layers, hidden, **model_kwargs)

        selection_metric = 'loss'
        if use_single_split:
            val_ratio = dataset_config.get(dataset_name, {}).get('val_ratio', 0.1)
            loss, acc, std = single_split_train_eval(
                train_dataset,
                test_dataset,
                model,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                lr_decay_factor=lr_decay_factor,
                lr_decay_step_size=lr_decay_step_size,
                weight_decay=weight_decay,
                val_ratio=val_ratio,
                logger=logger,
                selection_metric=selection_metric,
                seed=42,
            )
        else:
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
                logger=logger,
                selection_metric=selection_metric,
                kfold_seed=42,
                use_inner_val=(Net is LaCore),
            )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')

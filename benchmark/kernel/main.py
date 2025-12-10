import argparse
from dataclasses import replace
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

LAYERS = [2, 3, 4, 5]
HIDDEN_SIZES = [64, 128]
LACORE_EPSILONS = [0.001, 100, 10000]
DEFAULT_DATASETS = ['PROTEINS']
DEFAULT_NETS = [LaCore]

def build_parser():
    """Create the command-line parser for benchmark configuration."""
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
    parser.add_argument('--nets', type=str, default=None,
                        help="Comma-separated net names (e.g., 'GCN,LaCore'). Case sensitive.")
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Root directory that holds the datasets.')
    return parser


def resolve_dataset_names(dataset_arg):
    """Translate the raw datasets flag into an explicit list of names."""
    if dataset_arg:
        parsed = [name.strip() for name in dataset_arg.split(',') if name.strip()]
        if parsed:
            return parsed
    return DEFAULT_DATASETS


def build_dataset_config(dataset_names, args):
    """Prepare dataset-specific overrides such as ModelNet sampling."""
    config = {}
    if any(name.startswith('ModelNet') for name in dataset_names):
        modelnet_cfg = {
            'num_points': args.modelnet_num_points,
            'knn_k': args.modelnet_knn_k,
            'canonical_split': True,
            'val_ratio': args.modelnet_val_ratio,
        }
        config['ModelNet40'] = modelnet_cfg
        config['ModelNet10'] = modelnet_cfg
    return config


def resolve_nets(nets_arg):
    if nets_arg:
        parsed = [name.strip() for name in nets_arg.split(',') if name.strip()]
        if parsed:
            try:
                return [eval(name) for name in parsed]
            except NameError as exc:
                raise ValueError(f'Unknown net: {exc}')
    return DEFAULT_NETS


def create_logger(verbose):
    """Produce a logger that prints per-epoch metrics when verbose mode is on."""
    def logger(info):
        fold, epoch = info['fold'] + 1, info['epoch']
        train_acc = info['train_acc']
        val_acc = info['val_acc']
        val_loss = info['val_loss']
        test_acc = info['test_acc']
        best_test = info.get('test_acc_at_best_val', test_acc)
        best_epoch = info.get('best_epoch', epoch)
        if verbose:
            print(f'{fold:02d}/{epoch:03d}: '
                  f'Train Acc: {train_acc:.3f}, '
                  f'Val Acc: {val_acc:.3f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Test Acc: {test_acc:.3f}, '
                  f'Test@Best Val: {best_test:.3f} (epoch {best_epoch})')
    return logger


def get_training_setup(Net, dataset_name, epsilon, args, base_params=None):
    """Decide hyperparameters and transforms depending on the selected network."""
    if Net is LaCore:
        params = replace(base_params, epsilon=epsilon) if base_params else replace(
            LaCore.default_hparams(dataset_name), epsilon=epsilon)
        extra_transform = LaCoreAssignment(
            epsilon=params.epsilon,
            target_ratio=params.target_ratio,
            min_size=params.min_size,
            max_clusters=params.max_clusters,
        )
        return {
            'epochs': params.epochs,
            'weight_decay': params.weight_decay,
            'dropout': params.dropout,
            'extra_transform': extra_transform,
            'params': params,
        }
    return {
        'epochs': args.epochs,
        'weight_decay': 0,
        'dropout': None,
        'extra_transform': getattr(Net, 'extra_transform', None),
        'params': None,
    }


def run_experiments(dataset_names, dataset_config, args):
    """Iterate dataset / network combinations, train, and collect results."""
    logger = create_logger(args.verbose)
    results = []
    lr = args.lr
    batch_size = args.batch_size
    lr_decay_factor = args.lr_decay_factor
    lr_decay_step_size = args.lr_decay_step_size
    for dataset_name, Net in product(dataset_names, args.nets):
        best_result = (float('inf'), 0, 0)
        best_config = None
        print(f'----------\n{dataset_name} - {Net.__name__}')
        epsilon_grid = LACORE_EPSILONS if Net is LaCore else [None]
        base_params = None
        if Net is LaCore:
            base_params = LaCore.default_hparams(dataset_name)
            print("Overriding hyperparams with LaCore optimal settings:")
            print(base_params)
        for epsilon in epsilon_grid:
            training_setup = get_training_setup(Net, dataset_name, epsilon, args, base_params)
            params = training_setup['params']
            if Net is LaCore:
                print(f"Running epsilon={params.epsilon}")
            print(f"Loading dataset {dataset_name}...")
            dataset = get_dataset(
                dataset_name,
                sparse=Net != DiffPool,
                extra_transform=training_setup['extra_transform'],
                dataset_config=dataset_config,
                dataset_root=args.dataset_root,
            )
            print("Dataset loaded")
            for num_layers, hidden in product(LAYERS, HIDDEN_SIZES):
                use_single_split = isinstance(dataset, tuple)
                dataset_for_model = dataset[0] if use_single_split else dataset
                model_kwargs = {}
                if training_setup['dropout'] is not None:
                    model_kwargs['dropout'] = training_setup['dropout']
                model = Net(dataset_for_model, num_layers, hidden, **model_kwargs)
                selection_metric = 'loss'
                if use_single_split:
                    train_dataset, test_dataset = dataset
                    val_ratio = dataset_config.get(dataset_name, {}).get('val_ratio', 0.1)
                    loss, acc, std = single_split_train_eval(
                        train_dataset,
                        test_dataset,
                        model,
                        epochs=training_setup['epochs'],
                        batch_size=batch_size,
                        lr=lr,
                        lr_decay_factor=lr_decay_factor,
                        lr_decay_step_size=lr_decay_step_size,
                        weight_decay=training_setup['weight_decay'],
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
                        epochs=training_setup['epochs'],
                        batch_size=batch_size,
                        lr=lr,
                        lr_decay_factor=lr_decay_factor,
                        lr_decay_step_size=lr_decay_step_size,
                        weight_decay=training_setup['weight_decay'],
                        logger=logger,
                        selection_metric=selection_metric,
                        kfold_seed=42,
                        use_inner_val=(Net is LaCore),
                    )
                if loss < best_result[0]:
                    best_result = (loss, acc, std)
                    best_config = {
                        'num_layers': num_layers,
                        'hidden': hidden,
                    }
                    if Net is LaCore:
                        best_config['epsilon'] = params.epsilon
        desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
        if best_config is not None:
            print(f'Best config: {best_config}')
        print(f'Best result - {desc}')
        results.append(f'{dataset_name} - {Net.__name__}: {desc}')
    return results


def main():
    """Parse flags, resolve datasets and nets, and kick off the benchmark run."""
    args = build_parser().parse_args()
    try:
        args.nets = resolve_nets(args.nets)
    except ValueError as exc:
        raise SystemExit(exc)
    dataset_names = resolve_dataset_names(args.datasets)
    dataset_config = build_dataset_config(dataset_names, args)
    results = run_experiments(dataset_names, dataset_config, args)
    print(f'--\n{"\\n".join(results)}')


if __name__ == '__main__':
    main()

import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None,
                                  selection_metric='loss',
                                  kfold_seed=12345,
                                  use_inner_val=False):

    def _label_hist(indices):
        lbl = dataset.y[indices].view(-1).long()
        num_classes = int(dataset.y.max()) + 1
        return torch.bincount(lbl, minlength=num_classes).tolist()

    val_losses, val_accs, accs, durations = [], [], [], []
    if use_inner_val:
        # Match the splitting scheme in lacorepool_graph_classification.py:
        # outer 10-fold CV (train/test), with an inner stratified split on
        # the training portion to build the validation set. We do not carve
        # out the "previous fold" as an additional validation set (unlike
        # the original kernel benchmark), so the effective train/val/test
        # proportions mirror the reference script.
        outer_skf = StratifiedKFold(folds, shuffle=True, random_state=kfold_seed)
        fold_splits = list(outer_skf.split(torch.zeros(len(dataset)), dataset.y))
    else:
        fold_splits = list(zip(*k_fold(dataset, folds, seed=kfold_seed)))

    for fold, split in enumerate(fold_splits):
        if use_inner_val:
            train_idx_np, test_idx_np = split
            train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
            test_idx = torch.as_tensor(test_idx_np, dtype=torch.long)

            # Inner stratified split on the training subset.
            labels = dataset.y[train_idx]
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=kfold_seed)
            inner_train_sub, inner_val_sub = next(skf.split(torch.arange(len(train_idx)), labels))
            train_ids = train_idx[inner_train_sub]
            val_ids = train_idx[inner_val_sub]

            # Helpful debug signal to ensure we match the reference script.
            print(
                f"[LaCore split] fold {fold + 1}/{folds}: "
                f"train={len(train_ids)} val={len(val_ids)} test={len(test_idx)} | "
                f"train_hist={_label_hist(train_ids)} "
                f"val_hist={_label_hist(val_ids)} "
                f"test_hist={_label_hist(test_idx)}"
            )
        else:
            train_idx, test_idx, val_idx = split
            train_ids = train_idx
            val_ids = val_idx

        train_dataset = dataset[train_ids]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_ids]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except ImportError:
                pass

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            val_accs.append(eval_acc(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'val_acc': val_accs[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, v_acc, acc, duration = tensor(val_losses), tensor(val_accs), tensor(accs), tensor(durations)
    loss, v_acc, acc = loss.view(folds, epochs), v_acc.view(folds, epochs), acc.view(folds, epochs)
    if selection_metric == 'acc':
        best_val, argbest = v_acc.max(dim=1)
    else:
        best_val, argbest = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argbest]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Metric: {best_val.mean().item():.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds, seed=12345):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)

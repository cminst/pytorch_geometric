import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
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
        # Outer 10-fold CV (train/test), with an inner stratified split on
        # the training portion to build the validation set. We do not carve
        # out the "previous fold" as an additional validation set (unlike
        # the original kernel benchmark)
        outer_skf = StratifiedKFold(folds, shuffle=True, random_state=kfold_seed)
        fold_splits = list(outer_skf.split(torch.zeros(len(dataset)), dataset.y))
    else:
        fold_splits = list(zip(*k_fold(dataset, folds, seed=kfold_seed)))

    fold_iter = tqdm(
        fold_splits,
        total=len(fold_splits),
        desc="Folds",
        ascii=True,
    )

    for fold, split in enumerate(fold_iter):
        best_val_metric = -float('inf') if selection_metric == 'acc' else float('inf')
        best_test_at_val = 0.0
        best_epoch = 0
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
            train_acc = eval_acc(model, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            val_accs.append(eval_acc(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            val_metric = val_accs[-1] if selection_metric == 'acc' else val_losses[-1]
            if (selection_metric == 'acc' and val_metric > best_val_metric) or (
                    selection_metric != 'acc' and val_metric < best_val_metric):
                best_val_metric = val_metric
                best_test_at_val = accs[-1]
                best_epoch = epoch
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_losses[-1],
                'val_acc': val_accs[-1],
                'test_acc': accs[-1],
                'test_acc_at_best_val': best_test_at_val,
                'best_epoch': best_epoch,
            }

            if logger is not None and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
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


def single_split_train_eval(
    train_dataset,
    test_dataset,
    model,
    epochs,
    batch_size,
    lr,
    lr_decay_factor,
    lr_decay_step_size,
    weight_decay,
    val_ratio=0.1,
    logger=None,
    selection_metric='acc',
    seed=42,
):
    """Train/val/test on a single split (used for ModelNet canonical splits)."""
    labels = train_dataset.y.view(-1).cpu().numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_dataset)),
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )
    train_subset = train_dataset[torch.as_tensor(train_idx, dtype=torch.long)]
    val_subset = train_dataset[torch.as_tensor(val_idx, dtype=torch.long)]

    if hasattr(train_subset[0], 'adj'):
        train_loader = DenseLoader(train_subset, batch_size, shuffle=True)
        val_loader = DenseLoader(val_subset, batch_size, shuffle=False)
        test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
    else:
        train_loader = DataLoader(train_subset, batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except ImportError:
            pass

    val_losses, val_accs, test_accs, durations = [], [], [], []
    best_val_metric = -float('inf') if selection_metric == 'acc' else float('inf')
    best_test_at_val = 0.0
    best_epoch = 0

    t_start = time.perf_counter()
    epoch_iter = tqdm(
        range(1, epochs + 1),
        total=epochs,
        desc="Train epochs",
        ascii=True,
    )

    for epoch in epoch_iter:
        train_loss = train(model, optimizer, train_loader)
        train_acc = eval_acc(model, train_loader)
        val_loss = eval_loss(model, val_loader)
        val_acc = eval_acc(model, val_loader)
        test_acc = eval_acc(model, test_loader)

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        val_metric = val_acc if selection_metric == 'acc' else val_loss
        if (selection_metric == 'acc' and val_metric > best_val_metric) or (
                selection_metric != 'acc' and val_metric < best_val_metric):
            best_val_metric = val_metric
            best_test_at_val = test_acc
            best_epoch = epoch

        eval_info = {
            'fold': 0,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_acc_at_best_val': best_test_at_val,
            'best_epoch': best_epoch,
        }
        if logger is not None and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    val_losses_t = tensor(val_losses)
    val_accs_t = tensor(val_accs)
    test_accs_t = tensor(test_accs)
    duration_t = tensor(durations)

    if selection_metric == 'acc':
        best_val = val_accs_t.max()
        loss_out = (1.0 - best_val.item())  # smaller is better for selection
        test_at_best = test_accs_t[val_accs_t.argmax()].item()
    else:
        best_val = val_losses_t.min()
        loss_out = best_val.item()
        test_at_best = test_accs_t[val_losses_t.argmin()].item()

    acc_mean = test_at_best
    acc_std = 0.0
    duration_mean = duration_t.mean().item()
    print(f'Val Metric: {best_val.item():.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_out, acc_mean, acc_std


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

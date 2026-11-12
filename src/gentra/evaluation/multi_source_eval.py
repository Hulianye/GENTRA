"""Zero-shot evaluation helpers for multi-source MI pretraining."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from gentra.training.channel_encoding import unpack_batch
from gentra.training.metrics import get_metrics
from gentra.data.multi_source_dataset import MultiSourceChannelEEGDataset, multi_source_collate_fn
from gentra.training.trainer import _model_forward


def _class_to_local_map(eval_class_indices: tuple[int, ...] | list[int]) -> dict[int, int]:
    return {int(c): i for i, c in enumerate(eval_class_indices)}


def _remap_labels_for_eval(labels, class_to_local: dict[int, int]) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_classes = len(class_to_local)
    uniq = {int(y) for y in arr}
    # Cache labels already in 0..K-1 (e.g. BNCI2015001 binary 0/1 maps to pretraining classes 1,2).
    if uniq <= set(range(n_classes)):
        return arr.copy()
    out = np.empty(arr.shape[0], dtype=np.int64)
    for i, y in enumerate(arr):
        key = int(y)
        if key not in class_to_local:
            raise ValueError(
                f'label {key} not in eval_class_indices '
                f'{tuple(sorted(class_to_local))}'
            )
        out[i] = class_to_local[key]
    return out


def evaluate_loader(model, loader, device, criterion=None, eval_class_indices=None):
    model.eval()
    all_probs, all_labels = [], []
    running_loss = 0.0
    n = 0
    class_to_local = (
        _class_to_local_map(eval_class_indices) if eval_class_indices is not None else None
    )
    with torch.no_grad():
        for batch in loader:
            inputs, labels, channel_ids, channel_mask = unpack_batch(batch, device)
            outputs = _model_forward(model, inputs, channel_ids, channel_mask)
            label_np = labels.cpu().numpy()
            if eval_class_indices is not None:
                outputs = outputs[:, eval_class_indices]
                label_np = _remap_labels_for_eval(label_np, class_to_local)
            if criterion is not None:
                loss_labels = torch.as_tensor(label_np, device=device, dtype=labels.dtype)
                running_loss += criterion(outputs, loss_labels).item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(label_np)
            n += inputs.size(0)
    metric_labels = (
        list(range(len(eval_class_indices)))
        if eval_class_indices is not None
        else list(range(model.nb_classes))
    )
    metrics = get_metrics(
        np.array(all_probs),
        np.array(all_labels),
        metrics=['Acc', 'BAC', "Cohen's Kappa", 'weighted-F1'],
        labels=metric_labels,
    )
    if criterion is not None and n > 0:
        metrics['loss'] = running_loss / n
    return metrics


def evaluate_by_subject(
    model,
    x,
    y,
    subject_ids,
    channel_names,
    vocab,
    device,
    batch_size=64,
    eval_class_indices=None,
):
    from gentra.data.multi_source_mi import SourceBundle
    bundle = SourceBundle('test', x, y, subject_ids, channel_names, 0)
    subject_results = []
    for subject_id in sorted(np.unique(subject_ids), key=lambda v: float(v)):
        mask = subject_ids == subject_id
        ds = MultiSourceChannelEEGDataset([bundle], vocab, np.where(mask)[0])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=multi_source_collate_fn)
        metrics = evaluate_loader(model, loader, device, eval_class_indices=eval_class_indices)
        subject_results.append({
            'subject_id': subject_id,
            'balanced_accuracy': metrics['balanced_accuracy'],
            'accuracy': metrics['accuracy'],
            'n_trials': int(mask.sum()),
        })
    avg_bac = float(np.mean([r['balanced_accuracy'] for r in subject_results])) if subject_results else 0.0
    avg_acc = float(np.mean([r['accuracy'] for r in subject_results])) if subject_results else 0.0
    return {
        'average_subject_balanced_accuracy': avg_bac,
        'average_subject_accuracy': avg_acc,
        'subject_results': subject_results,
    }

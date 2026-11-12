"""Training and evaluation helpers for GENTRA fine-tuning."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from gentra.training.channel_encoding import unpack_batch
from gentra.training.metrics import get_metrics
from gentra.training.optimizer_schedule import (
    apply_optimizer_schedule,
    current_schedule_values,
    setup_optimizer_schedules,
)


def _model_forward(model, inputs, channel_ids=None, channel_mask=None):
    return model(inputs, channel_ids=channel_ids, channel_mask=channel_mask)


def _is_prototype_model(model):
    return hasattr(model, 'is_prototype_head') and model.is_prototype_head()


def _loss_targets(model, labels):
    if _is_prototype_model(model):
        return model.prototype_targets(labels)
    return labels


def _build_optimizer(model, args):
    weight_decay = args.weight_decay
    opt_cls = torch.optim.AdamW if args.optimizer_type == 'adamw' else torch.optim.Adam
    opt_kwargs = {
        'lr': args.lr,
        'eps': args.opt_eps,
        'betas': (args.b1, args.b2),
    }
    gate_lr_scale = getattr(args, 'moe_gate_lr_scale', 1.0)

    if _is_prototype_model(model):
        encoder_decay = []
        encoder_no_decay = []
        head_decay = []
        head_no_decay = []
        prototype_params = []
        head_prefix = 'main_model.head.'
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_proto = name.endswith('.prototypes')
            is_head = name.startswith(head_prefix)
            is_no_decay = getattr(args, 'filter_bias_and_bn', False) and (
                param.ndim <= 1
                or name.endswith('.bias')
                or 'bn' in name.lower()
                or 'norm' in name.lower()
            )
            if is_proto:
                prototype_params.append(param)
            elif is_head:
                (head_no_decay if is_no_decay else head_decay).append(param)
            else:
                (encoder_no_decay if is_no_decay else encoder_decay).append(param)
        param_groups = []
        proto_lr_scale = getattr(args, 'prototype_lr_scale', 3.0)
        if encoder_decay:
            param_groups.append({
                'params': encoder_decay,
                'weight_decay': weight_decay,
                'lr': args.lr,
                'group': 'encoder',
            })
        if encoder_no_decay:
            param_groups.append({
                'params': encoder_no_decay,
                'weight_decay': 0.0,
                'lr': args.lr,
                'group': 'encoder',
            })
        if head_decay:
            param_groups.append({
                'params': head_decay,
                'weight_decay': weight_decay,
                'lr': args.lr,
                'group': 'head',
            })
        if head_no_decay:
            param_groups.append({
                'params': head_no_decay,
                'weight_decay': 0.0,
                'lr': args.lr,
                'group': 'head',
            })
        if prototype_params:
            param_groups.append({
                'params': prototype_params,
                'weight_decay': 0.0,
                'lr': args.lr * proto_lr_scale,
                'group': 'prototype',
            })
        return opt_cls(param_groups, **opt_kwargs)

    gate_params = []
    base_decay = []
    base_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_gate = 'gate_mlp' in name or 'shared_filter_weight' in name or 'private_filter_weight' in name
        is_no_decay = param.ndim <= 1 or name.endswith('.bias') or 'bn' in name.lower() or 'norm' in name.lower()
        bucket = (gate_params if is_gate else base_no_decay if is_no_decay else base_decay)
        bucket.append(param)

    if gate_params and gate_lr_scale != 1.0:
        params = []
        if base_decay:
            params.append({'params': base_decay, 'weight_decay': weight_decay, 'lr': args.lr})
        if base_no_decay:
            params.append({'params': base_no_decay, 'weight_decay': 0.0, 'lr': args.lr})
        params.append({'params': gate_params, 'weight_decay': weight_decay, 'lr': args.lr * gate_lr_scale})
        return opt_cls(params, **opt_kwargs)

    params = model.parameters()
    if getattr(args, 'filter_bias_and_bn', False) and weight_decay > 0:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith('.bias') or 'bn' in name.lower() or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        params = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        weight_decay = 0.0

    return opt_cls(params, weight_decay=weight_decay, **opt_kwargs)


def _state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _train_labels_from_loader(train_loader):
    dataset = train_loader.dataset
    if hasattr(dataset, 'y'):
        labels = dataset.y.cpu().numpy()
    elif hasattr(dataset, 'tensors'):
        labels = dataset.tensors[1].cpu().numpy()
    else:
        labels = np.concatenate([
            batch[1].cpu().numpy() for batch in train_loader
        ], axis=0)
    return labels.astype(np.int64)


def _class_weights_from_labels(labels, device, nb_classes=None, max_ratio=None):
    labels = np.asarray(labels, dtype=np.int64)
    if nb_classes is None:
        nb_classes = int(labels.max()) + 1 if labels.size else 1
    class_counts = np.bincount(labels, minlength=nb_classes).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    weights = len(labels) / (nb_classes * class_counts)
    if max_ratio is not None and max_ratio > 0:
        w_max = weights.min() * float(max_ratio)
        weights = np.minimum(weights, w_max)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _checkpoint_score(metrics, checkpoint_metric):
    if checkpoint_metric == 'balanced_accuracy':
        return metrics.get('balanced_accuracy', 0.0)
    return metrics.get('accuracy', 0.0)


def evaluate_model(model, dataloader, criterion, device, logger=None, quiet=False, return_predictions=False):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        iterator = dataloader if quiet else tqdm(dataloader, desc='Evaluating', leave=False)
        for batch in iterator:
            inputs, labels, channel_ids, channel_mask = unpack_batch(batch, device)
            outputs = _model_forward(model, inputs, channel_ids, channel_mask)
            loss = criterion(outputs, _loss_targets(model, labels))
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = running_loss / len(dataloader.dataset)
    metrics = get_metrics(
        np.array(all_probs),
        np.array(all_labels),
        metrics=['Acc', 'BAC', "Cohen's Kappa", 'weighted-F1'],
        labels=list(range(model.nb_classes)),
    )
    metrics['loss'] = avg_loss
    if logger is not None and not quiet:
        logger.log(f'Eval Loss: {avg_loss:.4f}')
        for key, value in metrics.items():
            logger.log(f'Eval {key}: {value:.4f}')
    if return_predictions:
        return metrics, np.asarray(all_probs), np.asarray(all_labels)
    return metrics


def train_model(model, train_loader, test_loader, device, args, logger):
    quiet = getattr(args, 'quiet', False)
    patience = getattr(args, 'patience', 0) or 0
    final_epoch_only = getattr(args, 'checkpoint_metric', '') == 'final_epoch'
    if final_epoch_only and patience:
        raise ValueError('Fixed final-epoch reporting forbids test-driven early stopping')
    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'

    weight = None
    if getattr(args, 'class_weights', False):
        train_labels = _train_labels_from_loader(train_loader)
        weight = _class_weights_from_labels(train_labels, device)
        if logger and not quiet:
            logger.log(f'Calculated class weights: {weight.cpu().numpy()}')
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=weight)
    if hasattr(model, 'init_prototypes_from_loader'):
        model.init_prototypes_from_loader(train_loader, device, proto_init=getattr(args, 'proto_init', 'mean'))
    model.apply_finetune_strategy('full')
    optimizer = _build_optimizer(model, args)
    args.global_step = 0
    setup_optimizer_schedules(optimizer, args, train_loader, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    checkpoint_metric = getattr(args, 'checkpoint_metric', 'accuracy')
    best_val_score = 0.0
    best_val_metrics = None
    best_state = None
    stale_epochs = 0
    train_log = {'epochs': [], 'loss': [], 'metrics': [], 'gate_stats': []}
    gate_log = getattr(args, 'moe_gate_log', False)

    for epoch in range(args.epochs):
        if not quiet:
            logger.log(f'\nEpoch {epoch + 1}/{args.epochs}')
            logger.log('-' * 10)

        model.train()
        running_loss = 0.0
        train_iterator = train_loader if quiet else tqdm(train_loader, desc='Training', leave=False)

        for batch in train_iterator:
            apply_optimizer_schedule(optimizer, args)
            inputs, labels, channel_ids, channel_mask = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = _model_forward(model, inputs, channel_ids, channel_mask)
                loss = criterion(outputs, _loss_targets(model, labels))
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, test_loader, criterion, device, logger=None, quiet=True)
        val_score = _checkpoint_score(val_metrics, checkpoint_metric)

        gate_stats = {}
        if gate_log and hasattr(model, 'collect_gate_stats'):
            with torch.no_grad():
                batch = next(iter(test_loader))
                inputs, labels, channel_ids, channel_mask = unpack_batch(batch, device)
                _model_forward(model, inputs, channel_ids, channel_mask)
                gate_stats = model.collect_gate_stats(labels=labels)
                gate_stats['epoch'] = epoch + 1
            train_log['gate_stats'].append(gate_stats)

        if not quiet:
            logger.log(f'Train Loss: {epoch_loss:.4f}')
            logger.log(f'Eval BAC: {val_metrics.get("balanced_accuracy", 0.0):.4f}')

        train_log['epochs'].append(epoch + 1)
        train_log['loss'].append(epoch_loss)
        lr_min, lr_max, weight_decay = current_schedule_values(optimizer)
        train_log['metrics'].append({
            **val_metrics,
            'train_loss': epoch_loss,
            'test_loss': val_metrics['loss'],
            'lr_min': lr_min,
            'lr_max': lr_max,
            'weight_decay': weight_decay,
        })

        if final_epoch_only:
            # Test curves are diagnostic only: no selection, stopping or rollback.
            best_val_metrics = val_metrics
            best_val_score = val_metrics.get('balanced_accuracy', 0.0)
            continue
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_metrics = val_metrics
            best_state = _state_dict_cpu(model)
            stale_epochs = 0
            if not quiet:
                logger.log(f'New best validation {checkpoint_metric}: {best_val_score:.4f}')
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                if not quiet:
                    logger.log(f'Early stop at epoch {epoch + 1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if not quiet:
        logger.log(f'\nTraining completed. Selected {checkpoint_metric}: {best_val_score:.4f}')
    return model, best_val_score, train_log, best_val_metrics

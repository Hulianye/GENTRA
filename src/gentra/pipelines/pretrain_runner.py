#!/usr/bin/env python3
"""Multi-source MI mixed pretraining with MOE+RoPE and zero-shot BNCI test."""

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gentra.config.paths import project_root
PROJECT_ROOT = str(project_root())

from gentra.config.build import (
    build_pretrain_config,
    save_config_snapshot,
    source_pkl_path_from_cfg,
    train_pre_cfg_from_config,
    zs_pre_cfg_from_config,
)
from gentra.data.registry import SOURCE_CLASS_SPECS, source_pkl_path
from gentra.data.source_channels import channel_names_for_source
from gentra.models.loader import GENTRAModel
from gentra.training.channel_encoding import vocab_size_from_vocab
from gentra.training.logger import Logger
from gentra.training.multi_source_da_train import train_one_epoch_da
from gentra.data.multi_source_dataset import (
    MultiSourceChannelEEGDataset,
    labels_from_bundles,
    multi_source_collate_fn,
    source_ids_from_bundles,
)
from gentra.evaluation.multi_source_eval import evaluate_by_subject, evaluate_loader
from gentra.data.multi_source_mi import build_multi_source_pool
from gentra.data.multi_source_sampling import (
    ClassBalancedBatchSampler,
    stratified_train_val_indices,
    stratified_train_val_indices_by_source,
)
from gentra.data.zs_test_cache import build_all_zs_tests
from gentra.training.optimizer_schedule import current_schedule_values, setup_optimizer_schedules
from gentra.training.trainer import (
    _build_optimizer,
    _class_weights_from_labels,
    _is_prototype_model,
    _state_dict_cpu,
)


def preprocess_band_tag(pre_cfg):
    return f"{pre_cfg.get('l_freq')}-{pre_cfg.get('h_freq')}"


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataloader_kwargs(cfg, device, seed=0):
    workers = int(cfg.get('dataloader_workers', 0))
    use_cuda = device.type == 'cuda'
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kw = {
        'num_workers': workers,
        'pin_memory': use_cuda,
        'generator': generator,
    }
    if workers > 0:
        kw['worker_init_fn'] = _seed_worker
        kw['persistent_workers'] = bool(cfg.get('persistent_workers', True))
        kw['prefetch_factor'] = int(cfg.get('prefetch_factor', 2))
    return kw


def _test_specs_from_cfg(cfg):
    custom = cfg.get('zs_test_specs')
    if custom:
        return list(custom)
    return []


def set_seed(seed, cudnn_benchmark=False):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn_benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    if not cudnn_benchmark:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def model_kwargs_from_config(cfg):
    kwargs = {
        'emb_size': cfg['emb_size'],
        'depth': cfg['depth'],
        'conv_filters': cfg['conv_filters'],
        'conv_temporal_kernel': cfg['conv_temporal_kernel'],
        'conv_temporal_kernel2': cfg.get('conv_temporal_kernel2', 15),
        'pool_kernel': cfg['pool_kernel'],
        'pool_stride': cfg['pool_stride'],
        'dropout_rate': cfg.get('dropout_rate', 0.1),
        'head_type': cfg['head_type'],
        'activation': cfg['activation'],
        'moe_shared_filters': cfg.get('moe_shared_filters', 16),
        'moe_private_filters': cfg.get('moe_private_filters', 48),
        'moe_top_k': cfg.get('moe_top_k', 16),
        'moe_apply_topk_weight': cfg.get('moe_apply_topk_weight', True),
        'moe_gate_feature_mode': cfg.get('moe_gate_feature_mode', 'layout'),
        'moe_mixer_mode': cfg.get('moe_mixer_mode', 'stack_gated_topk'),
        'moe_stack_kernel': cfg.get('moe_stack_kernel', 1),
        'moe_stack_merge_mode': cfg.get('moe_stack_merge_mode', 'full_1x1'),
        'shallow_norm_type': cfg.get('shallow_norm_type', 'batch'),
        'use_rope': cfg.get('use_rope', True),
    }
    if cfg.get('num_heads'):
        kwargs['num_heads'] = cfg['num_heads']
    if cfg.get('head_type') == 'cls_prototype':
        kwargs['proj_dim'] = cfg.get('proj_dim') or cfg['emb_size']
        kwargs['temperature'] = cfg.get('temperature', 0.1)
        kwargs['num_prototypes'] = cfg.get('num_prototypes', 1)
        kwargs['prototype_logit_aggregation'] = cfg.get(
            'prototype_logit_aggregation', 'logmeanexp'
        )
    kwargs['rope_mode'] = cfg.get('rope_mode', 'attention_qk')
    kwargs['attention_scale'] = cfg.get('attention_scale', 'emb_size')
    return kwargs


def build_args(cfg, seed, gpuid, timestamp):
    ns = SimpleNamespace(**cfg)
    ns.seeds = [seed]
    ns.gpuid = gpuid
    ns.timestamp = timestamp
    ns.label_smoothing = cfg.get('label_smoothing', 0.0)
    ns.clip_grad_norm = cfg.get('clip_grad_norm', 1.0)
    ns.optimizer_type = cfg.get('optimizer_type', 'adamw')
    ns.opt_eps = float(cfg.get('opt_eps', 1e-8))
    ns.b1 = float(cfg.get('b1', 0.9))
    ns.b2 = float(cfg.get('b2', 0.999))
    ns.filter_bias_and_bn = bool(cfg.get('filter_bias_and_bn', True))
    ns.finetune_strategy = cfg.get('finetune_strategy', 'full')
    ns.class_weights = cfg.get('class_weights', False)
    ns.max_class_weight_ratio = cfg.get('max_class_weight_ratio', None)
    ns.checkpoint_metric = cfg.get('checkpoint_metric', 'balanced_accuracy')
    ns.quiet = cfg.get('quiet', True)
    ns.cudnn_benchmark = cfg.get('cudnn_benchmark', False)
    ns.use_amp = cfg.get('use_amp', False)
    ns.filter_sparsity_l1_weight = cfg.get('filter_sparsity_l1_weight', 0.0)
    ns.filter_sparsity_l1_target = cfg.get(
        'filter_sparsity_l1_target', 'normalized_only'
    )
    ns.data_workers = cfg.get('data_workers', 1)
    ns.dataloader_workers = cfg.get('dataloader_workers', 0)
    ns.persistent_workers = cfg.get('persistent_workers', False)
    ns.prefetch_factor = cfg.get('prefetch_factor', 2)
    ns.proto_init = cfg.get('proto_init', 'mean')
    ns.prototype_lr_scale = cfg.get('prototype_lr_scale', 3.0)
    ns.aux_method = cfg.get('aux_method', '') or ''
    ns.lambda_aux = cfg.get('lambda_aux', 0.0)
    ns.supcon_temperature = cfg.get('supcon_temperature', 0.1)
    ns.save_zs_eval_epochs = cfg.get('save_zs_eval_epochs') or []
    ns.train_classes = tuple(cfg.get('train_classes', (0, 1, 2)))
    ns.recon_target = cfg.get('recon_target', '') or ''
    ns.lambda_recon = float(cfg.get('lambda_recon', 0.0))
    ns.mask_ratio = float(cfg.get('mask_ratio', 0.0))
    ns.mask_block_tokens = int(cfg.get('mask_block_tokens', 8))
    ns.channel_mask_max_ratio = float(cfg.get('channel_mask_max_ratio', 0.0))
    ns.recon_warmup_epochs = int(cfg.get('recon_warmup_epochs', 0))
    ns.use_lr_scheduler = bool(cfg.get('use_lr_scheduler', False))
    ns.use_wd_scheduler = bool(cfg.get('use_wd_scheduler', False))
    ns.lr_warmup_epochs = int(cfg.get('lr_warmup_epochs', cfg.get('warmup_epochs', 0)))
    ns.warmup_epochs = ns.lr_warmup_epochs
    ns.lr_decay_start_epoch = int(cfg.get('lr_decay_start_epoch', 1))
    ns.wd_decay_start_epoch = int(cfg.get('wd_decay_start_epoch', ns.lr_decay_start_epoch))
    ns.min_lr_ratio = float(cfg.get('min_lr_ratio', 0.0))
    ns.min_lr = float(cfg.get('min_lr', cfg.get('lr', 5e-5) * (ns.min_lr_ratio or 0.01)))
    ns.min_weight_decay = float(cfg.get('min_weight_decay', cfg.get('final_weight_decay', 0.0)))
    ns.final_weight_decay = float(cfg.get('final_weight_decay', ns.min_weight_decay))
    ns.global_step = 0
    ns.recon_hidden_dim = int(cfg.get('recon_hidden_dim', 128))
    ns.recon_samples_per_token = int(cfg.get('recon_samples_per_token', 16))
    ns.conv_temporal_kernel = int(cfg.get('conv_temporal_kernel', 31))
    ns.conv_temporal_kernel2 = int(cfg.get('conv_temporal_kernel2', 15))
    ns.pool_kernel = int(cfg.get('pool_kernel', 80))
    ns.pool_stride = int(cfg.get('pool_stride', 16))
    return ns


def _sanitize_cfg_for_save(cfg):
    return {k: v for k, v in cfg.items() if not str(k).startswith('_')}


def _split_train_val_indices(labels, source_ids, cfg, seed):
    val_ratio = float(cfg.get('val_ratio', 0.2))
    if cfg.get('source_aware_val', True):
        return stratified_train_val_indices_by_source(
            labels, source_ids, val_ratio, seed,
        )
    return stratified_train_val_indices(labels, val_ratio, seed)


def _build_train_loader(bundles, vocab, train_idx, labels, cfg, device, args):
    train_ds = MultiSourceChannelEEGDataset(bundles, vocab, train_idx)
    dl_kw = _dataloader_kwargs(cfg, device, seed=args.seeds[0])
    if cfg.get('class_balanced_sampling', False):
        class_ids = tuple(cfg.get('train_classes', [0, 1, 2]))
        train_sampler = ClassBalancedBatchSampler(
            labels[train_idx],
            cfg['batch_size'],
            class_ids=class_ids,
            seed=args.seeds[0],
        )
        return DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=multi_source_collate_fn,
            **dl_kw,
        )
    return DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=multi_source_collate_fn,
        **dl_kw,
    )


def _save_epoch_checkpoint(output_dir, model, vocab, cfg, epoch, min_zs=None):
    path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    payload = {
        'model_state': _state_dict_cpu(model),
        'vocab': vocab,
        'cfg': _sanitize_cfg_for_save(cfg),
        'epoch': epoch,
        'checkpoint_metric': 'epoch_snapshot',
    }
    if min_zs is not None:
        payload['min_zs_bac'] = min_zs
    torch.save(payload, path)


def make_dummy_eeg(vocab, n_classes=2):
    from gentra.data.eeg_data import EEGData
    labels = np.arange(n_classes, dtype=np.int64)
    n = max(2, n_classes)
    return EEGData(
        dataset_name='MI_multi',
        eeg_data=np.zeros((n, 8, 1000), dtype=np.float32),
        subject_ids=np.arange(1, n + 1),
        channel_names=['C3', 'C4', 'CZ', 'FZ'],
        sampling_rate=250,
        labels=labels,
        dataset_type='classification',
    )


def _log_epoch(msg, log_metrics=True):
    print(msg, flush=True)


def flush_progress(output_dir, results, summary_dir=None, write_summary_fn=None):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'multi_source_results.json')
    tmp_path = results_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp_path, results_path)

    jsonl_path = os.path.join(output_dir, 'train_log.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for row in results.get('train_log', []):
            f.write(json.dumps(row, default=str) + '\n')

    if summary_dir and write_summary_fn and os.path.isfile(results_path):
        write_summary_fn(results_path, summary_dir)


def format_epoch_log(epoch, epochs, record):
    parts = [f'Epoch {epoch}/{epochs} loss={record["train_loss"]:.4f}']
    if 'train_accuracy' in record:
        parts.append(f'train_acc={record["train_accuracy"]:.4f}')
    if 'train_balanced_accuracy' in record:
        parts.append(f'train_BAC={record["train_balanced_accuracy"]:.4f}')
    if 'val_balanced_accuracy' in record:
        parts.append(f'val_BAC={record["val_balanced_accuracy"]:.4f}')
    if 'test_BNCI2014004_bac' in record:
        parts.append(f'test_14004={record["test_BNCI2014004_bac"]:.4f}')
    if 'test_BNCI2015001_bac' in record:
        parts.append(f'test_15001={record["test_BNCI2015001_bac"]:.4f}')
    if 'test_BNCI2014001_bac' in record:
        parts.append(f'test_14001={record["test_BNCI2014001_bac"]:.4f}')
    if 'mean_zs_bac' in record:
        parts.append(f'mean_ZS={record["mean_zs_bac"]:.4f}')
    if 'min_zs_bac' in record:
        parts.append(f'min_ZS={record["min_zs_bac"]:.4f}')
    if 'test_Zhou2016_bac' in record:
        parts.append(f'test_Zhou2016={record["test_Zhou2016_bac"]:.4f}')
    if 'mean4_bac' in record:
        parts.append(f'mean4={record["mean4_bac"]:.4f}')
    return ' '.join(parts)


def _zs_test_bacs(record):
    return [
        v for k, v in record.items()
        if k.startswith('test_') and k.endswith('_bac') and v is not None
    ]


def _min_zs_from_record(record):
    vals = _zs_test_bacs(record)
    if not vals:
        return None
    return min(vals)


def _mean_zs_from_record(record):
    vals = _zs_test_bacs(record)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _zs_best_score_from_record(record, mode='min'):
    if mode == 'mean':
        return _mean_zs_from_record(record)
    return _min_zs_from_record(record)


def _annotate_zs_scores(record, mode='min'):
    min_zs = _min_zs_from_record(record)
    if min_zs is not None:
        record['min_zs_bac'] = min_zs
    mean_zs = _mean_zs_from_record(record)
    if mean_zs is not None:
        record['mean_zs_bac'] = mean_zs
    best = _zs_best_score_from_record(record, mode=mode)
    if best is not None:
        record['zs_best_bac'] = best
    return best


def _annotate_mean4_scores(record, datasets):
    if not datasets:
        return None
    vals = []
    for name in datasets:
        key = f'test_{name}_bac'
        if record.get(key) is None:
            return None
        vals.append(float(record[key]))
    mean4 = sum(vals) / len(vals)
    record['mean4_bac'] = mean4
    record['min4_bac'] = min(vals)
    return mean4


def _run_zero_shot_eval_record(
    model,
    record,
    *,
    vocab,
    test_specs,
    zs_bundles,
    zs_loo_bundles,
    cfg,
    device,
    zs_best_metric,
):
    from gentra.evaluation.zs_eval_protocol import resolve_zs_eval_class_indices

    batch_size = min(64, cfg['batch_size'])
    for entry in test_specs:
        test_name = entry[0]
        eval_class_indices = resolve_zs_eval_class_indices(test_name, entry, model, cfg)
        test_res = evaluate_by_subject(
            model,
            zs_bundles[test_name].x,
            zs_bundles[test_name].y,
            zs_bundles[test_name].subject_ids,
            zs_bundles[test_name].channel_names,
            vocab,
            device,
            batch_size=batch_size,
            eval_class_indices=eval_class_indices,
        )
        record[f'test_{test_name}_bac'] = test_res['average_subject_balanced_accuracy']

    loo_idx_map = cfg.get('zs_loo_eval_class_indices') or {}
    for source_name, bundle in zs_loo_bundles.items():
        eval_class_indices = loo_idx_map.get(source_name)
        if eval_class_indices is None:
            eval_class_indices = tuple(SOURCE_CLASS_SPECS[source_name]['allowed_classes'])
        test_res = evaluate_by_subject(
            model,
            bundle.x,
            bundle.y,
            bundle.subject_ids,
            bundle.channel_names,
            vocab,
            device,
            batch_size=batch_size,
            eval_class_indices=eval_class_indices,
        )
        record[f'test_{source_name}_bac'] = test_res['average_subject_balanced_accuracy']

    _annotate_zs_scores(record, mode=zs_best_metric)
    mean4_datasets = cfg.get('mean4_datasets')
    if mean4_datasets:
        _annotate_mean4_scores(record, mean4_datasets)
    return record


def _load_zs_loo_eval_bundles(cfg, logger):
    sources = list(cfg.get('zs_loo_eval_sources') or [])
    if not sources:
        return {}
    from gentra.evaluation.loo_loader import load_loo_preprocessed_bundle
    from gentra.data.multi_source_mi import SourceBundle

    bundles = {}
    for source_name in sources:
        x, y, sid, channel_names, allowed = load_loo_preprocessed_bundle(source_name, cfg)
        bundles[source_name] = SourceBundle(
            source_name, x, y, sid, channel_names, 0,
        )
        logger.log(f'  ZS LOO {source_name}: n={len(y)} trials (classes={allowed})')
    return bundles


def _peak_zs_score_from_log(train_log, key='min_zs_bac'):
    peak = -1.0
    peak_epoch = 0
    for record in train_log:
        score = record.get(key)
        if score is not None and score > peak:
            peak = score
            peak_epoch = record['epoch']
    if peak < 0:
        return None, None
    return peak, peak_epoch


def _peak_min_zs_from_log(train_log):
    return _peak_zs_score_from_log(train_log, key='min_zs_bac')


def _peak_mean_zs_from_log(train_log):
    return _peak_zs_score_from_log(train_log, key='mean_zs_bac')


def _metrics_at_epoch(train_log, epoch, *, strict: bool = False):
    for record in train_log:
        if record.get('epoch') == epoch:
            return dict(record)
    if strict:
        return {}
    if train_log and train_log[-1].get('epoch', 0) < epoch:
        return dict(train_log[-1])
    return {}


def run_training(cfg, output_dir, seed=0, gpuid=0, summary_dir=None, write_summary_fn=None):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed, cudnn_benchmark=cfg.get('cudnn_benchmark', False))
    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
    args = build_args(cfg, seed, gpuid, os.path.basename(output_dir))
    log_metrics = cfg.get('log_metrics', not args.quiet)
    logger = Logger(verbose=log_metrics)

    cache_dir = cfg.get('cache_dir', '')
    train_pre_cfg = train_pre_cfg_from_config(cfg)
    zs_pre_cfg = zs_pre_cfg_from_config(cfg)
    sources = cfg['sources']
    logger.log(
        f'Loading {len(sources)} sources... pre={preprocess_band_tag(train_pre_cfg)} '
        f'cache={cache_dir}'
    )
    pkl_fn = source_pkl_path_from_cfg(cfg) if cfg.get('pkl_cache_root') else source_pkl_path
    bundles, vocab = build_multi_source_pool(
        sources,
        pkl_fn,
        channel_names_for_source,
        train_pre_cfg,
        cache_dir,
        train_classes=tuple(cfg.get('train_classes', [0, 1])),
        data_workers=cfg.get('data_workers', 1),
        preprocess_source_workers=cfg.get('preprocess_source_workers', 0),
        source_class_overrides=cfg.get('source_class_overrides'),
    )
    labels = labels_from_bundles(bundles)
    source_ids = source_ids_from_bundles(bundles)
    logger.log(f'Total trials: {len(labels)} from {len(bundles)} sources, vocab size={len(vocab)}')

    train_idx, val_idx = _split_train_val_indices(labels, source_ids, cfg, seed)

    if cfg.get('class_balanced_sampling'):
        train_labels = labels[train_idx]
        class_ids = tuple(cfg.get('train_classes', [0, 1, 2]))
        counts = {int(c): int(np.sum(train_labels == c)) for c in class_ids}
        per_class = cfg['batch_size'] // len(class_ids)
        cfg['class_balanced_sampling_stats'] = {
            'train_class_counts': counts,
            'batch_size': cfg['batch_size'],
            'per_class_per_batch': per_class,
            'batches_per_epoch': max(counts.values()) // per_class if counts else 0,
        }
        logger.log(
            f'Class-balanced batch sampling: counts={counts}, '
            f'{per_class}/class per batch (batch_size={cfg["batch_size"]}), '
            f'~{cfg["class_balanced_sampling_stats"]["batches_per_epoch"]} batches/epoch'
        )

    val_ds = MultiSourceChannelEEGDataset(bundles, vocab, val_idx)
    train_loader = _build_train_loader(bundles, vocab, train_idx, labels, cfg, device, args)
    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg['batch_size'],
            shuffle=False,
            collate_fn=multi_source_collate_fn,
            **_dataloader_kwargs(cfg, device, seed=seed),
        )

    test_specs = _test_specs_from_cfg(cfg)
    zs_bundles = {}
    if test_specs and int(cfg.get('zs_eval_every') or 0) > 0:
        zs_cache_dir = cfg['zs_cache_dir']
        logger.log(f'Loading zero-shot test caches from {zs_cache_dir}...')
        zs_bundles = build_all_zs_tests(
            test_specs,
            zs_pre_cfg,
            vocab,
            zs_cache_dir,
            data_workers=cfg.get('data_workers', 1),
        )
        for name, bundle in zs_bundles.items():
            logger.log(f'  ZS {name}: n={len(bundle.y)} trials')
    zs_loo_bundles = _load_zs_loo_eval_bundles(cfg, logger)

    n_classes = len(cfg.get('train_classes', [0, 1]))
    dummy = make_dummy_eeg(vocab, n_classes=n_classes)
    model = GENTRAModel(
        dummy,
        finetune_strategy='full',
        channel_vocab=vocab,
        channel_vocab_size=vocab_size_from_vocab(vocab),
        **model_kwargs_from_config(cfg),
    ).to(device)

    recon_target = cfg.get('recon_target', '') or ''
    if recon_target and float(cfg.get('lambda_recon', 0.0)) > 0:
        from gentra.models.masked_reconstruction import build_reconstruction_head
        pe = getattr(model.main_model, 'patch_embedding', None)
        conv_dim = int(getattr(pe, 'conv_filters', cfg.get('conv_filters', 40)))
        model.recon_head = build_reconstruction_head(
            recon_target,
            emb_size=int(cfg.get('emb_size', 64)),
            conv_dim=conv_dim,
            channel_vocab_size=vocab_size_from_vocab(vocab),
            recon_hidden_dim=int(cfg.get('recon_hidden_dim', 128)),
            samples_per_token=int(cfg.get('recon_samples_per_token', 16)),
        ).to(device)
    else:
        model.recon_head = None

    if hasattr(model, 'init_prototypes_from_loader') and _is_prototype_model(model):
        logger.log(f'Initializing prototypes ({args.proto_init})...')
        model.init_prototypes_from_loader(train_loader, device, proto_init=args.proto_init)

    model.apply_finetune_strategy('full')

    class_weight = None
    if args.class_weights:
        train_labels = labels[train_idx]
        class_weight = _class_weights_from_labels(
            train_labels,
            device,
            nb_classes=n_classes,
            max_ratio=getattr(args, 'max_class_weight_ratio', None),
        )
        logger.log(f'Using class weights: {class_weight.detach().cpu().tolist()}')
    args.class_weight_tensor = class_weight
    criterion = nn.CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = _build_optimizer(model, args)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp and device.type == 'cuda')

    eval_every = cfg.get('eval_every', 5)
    train_eval_every = cfg.get('train_eval_every', 0)
    zs_eval_every = int(cfg.get('zs_eval_every') or 0) or eval_every
    if not test_specs and not zs_loo_bundles:
        zs_eval_every = 0
    flush_every_epoch = cfg.get('flush_every_epoch', False)
    epochs = cfg['epochs']
    setup_optimizer_schedules(optimizer, args, train_loader, epochs)
    save_zs_eval_epochs = set(cfg.get('save_zs_eval_epochs') or [])
    best_val_bac = -1.0
    best_val_epoch = 0
    zs_best_metric = str(cfg.get('zs_best_metric', 'min')).lower()
    if zs_best_metric not in ('min', 'mean'):
        raise ValueError(f'Unknown zs_best_metric={zs_best_metric!r}; expected min or mean')
    train_log = []

    results_base = {
        'params': _sanitize_cfg_for_save(cfg),
        'seed': seed,
        'vocab_size': len(vocab),
        'n_trials_total': len(labels),
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'status': 'running',
        'run_id': cfg.get('run_id', ''),
    }
    if cfg.get('class_balanced_sampling_stats'):
        results_base['class_balanced_sampling_stats'] = cfg['class_balanced_sampling_stats']

    for epoch in range(1, epochs + 1):

        train_out = train_one_epoch_da(
            model, train_loader, optimizer, device, args, scaler, epoch=epoch,
        )
        if isinstance(train_out, dict):
            record = {'epoch': epoch, **train_out}
        else:
            record = {'epoch': epoch, 'train_loss': train_out}

        if getattr(args, 'use_lr_scheduler', False) or getattr(args, 'use_wd_scheduler', False):
            lr_min, lr_max, wd = current_schedule_values(optimizer)
            record['lr_base'] = lr_min
            record['lr_max'] = lr_max
            record['weight_decay_active'] = wd

        should_eval_train = train_eval_every > 0 and (
            epoch % train_eval_every == 0 or epoch == epochs
        )
        if should_eval_train:
            train_metrics = evaluate_loader(model, train_loader, device, criterion)
            record['train_accuracy'] = train_metrics['accuracy']
            record['train_balanced_accuracy'] = train_metrics['balanced_accuracy']

        should_eval_val = val_loader is not None and (
            epoch % eval_every == 0 or epoch == epochs
        )
        should_eval_zs = (
            bool(zs_bundles) or bool(zs_loo_bundles)
        ) and (
            (zs_eval_every > 0 and epoch % zs_eval_every == 0)
            or epoch == epochs
            or epoch in save_zs_eval_epochs
        )

        if should_eval_val:
            val_metrics = evaluate_loader(model, val_loader, device, criterion)
            record['val_balanced_accuracy'] = val_metrics['balanced_accuracy']
            record['val_accuracy'] = val_metrics['accuracy']
            if val_metrics['balanced_accuracy'] > best_val_bac:
                best_val_bac = val_metrics['balanced_accuracy']
                best_val_epoch = epoch

        if should_eval_zs:
            _run_zero_shot_eval_record(
                model,
                record,
                vocab=vocab,
                test_specs=test_specs,
                zs_bundles=zs_bundles,
                zs_loo_bundles=zs_loo_bundles,
                cfg=cfg,
                device=device,
                zs_best_metric=zs_best_metric,
            )
            if epoch in save_zs_eval_epochs:
                _save_epoch_checkpoint(output_dir, model, vocab, cfg, epoch, min_zs=record.get('min_zs_bac'))

        train_log.append(record)
        if log_metrics:
            _log_epoch(format_epoch_log(epoch, epochs, record), log_metrics)

        if flush_every_epoch:
            results = {
                **results_base,
                'best_epoch': epoch,
                'best_val_epoch': best_val_epoch,
                'best_val_balanced_accuracy': best_val_bac,
                'primary_checkpoint_metric': 'final_epoch',
                'zs_best_metric': zs_best_metric,
                'train_log': train_log,
                'last_epoch': epoch,
            }
            flush_progress(output_dir, results, summary_dir, write_summary_fn)

    final_epoch = epochs
    peak_min_zs, peak_min_zs_epoch = _peak_min_zs_from_log(train_log)
    peak_mean_zs, peak_mean_zs_epoch = _peak_mean_zs_from_log(train_log)
    peak_zs_epoch = peak_mean_zs_epoch if zs_best_metric == 'mean' else peak_min_zs_epoch

    last_record = train_log[-1] if train_log else {}
    final_path = os.path.join(output_dir, 'checkpoint_final.pt')
    torch.save({
        'model_state': _state_dict_cpu(model),
        'vocab': vocab,
        'cfg': _sanitize_cfg_for_save(cfg),
        'epoch': final_epoch,
        'seed': seed,
        'checkpoint_metric': 'final_epoch',
        'metrics': {
            k: v for k, v in last_record.items()
            if k.startswith('test_') or k in ('mean4_bac', 'min4_bac', 'min_zs_bac', 'mean_zs_bac')
        },
    }, final_path)
    logger.log(f'Saved final checkpoint: {final_path}')

    results = {
        **results_base,
        'status': 'done',
        'best_epoch': final_epoch,
        'best_val_epoch': best_val_epoch,
        'best_val_balanced_accuracy': best_val_bac,
        'primary_checkpoint_metric': 'final_epoch',
        'zs_best_metric': zs_best_metric,
        'peak_min_zs_bac': peak_min_zs,
        'peak_mean_zs_bac': peak_mean_zs,
        'peak_zs_epoch': peak_zs_epoch,
        'train_log': train_log,
        'last_epoch': final_epoch,
        'checkpoint_final': final_path,
        'checkpoint': final_path,
    }

    flush_progress(output_dir, results, summary_dir, write_summary_fn)
    peak_min_zs_str = f'{peak_min_zs:.4f}' if peak_min_zs is not None else 'n/a'
    peak_mean_zs_str = f'{peak_mean_zs:.4f}' if peak_mean_zs is not None else 'n/a'
    logger.log(
        f'Done. best_val_BAC={best_val_bac:.4f} @ ep{best_val_epoch}; '
        f'peak_min_ZS={peak_min_zs_str} @ ep{peak_min_zs_epoch or "-"}; '
        f'peak_mean_ZS={peak_mean_zs_str} @ ep{peak_mean_zs_epoch or "-"}'
    )
    return results


def main():
    parser = argparse.ArgumentParser(description='GENTRA multi-source pretrain')
    parser.add_argument('--config', type=str, default=None, help='Path to pretrain YAML')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--build_cache_only', action='store_true')
    parser.add_argument('--build_zs_cache_only', action='store_true')
    cli = parser.parse_args()

    cfg = build_pretrain_config(cli.config)
    pre_cfg = train_pre_cfg_from_config(cfg)
    zs_pre_cfg = zs_pre_cfg_from_config(cfg)

    if cli.build_cache_only:
        pkl_fn = source_pkl_path_from_cfg(cfg)
        build_multi_source_pool(
            cfg['sources'], pkl_fn, channel_names_for_source, pre_cfg, cfg['cache_dir'],
            train_classes=tuple(cfg.get('train_classes', [0, 1, 2, 3, 4])),
            data_workers=cfg.get('data_workers', 4),
            preprocess_source_workers=cfg.get('preprocess_source_workers', 3),
            source_class_overrides=cfg.get('source_class_overrides'),
        )
        print(f'Built caches under {cfg["cache_dir"]}', flush=True)
        return

    if cli.build_zs_cache_only:
        pkl_fn = source_pkl_path_from_cfg(cfg)
        _, vocab = build_multi_source_pool(
            cfg['sources'], pkl_fn, channel_names_for_source, pre_cfg, cfg['cache_dir'],
            train_classes=tuple(cfg.get('train_classes', [0, 1, 2, 3, 4])),
            data_workers=cfg.get('data_workers', 4),
            preprocess_source_workers=cfg.get('preprocess_source_workers', 3),
            source_class_overrides=cfg.get('source_class_overrides'),
        )
        build_all_zs_tests(_test_specs_from_cfg(cfg), zs_pre_cfg, vocab, cfg['zs_cache_dir'],
                           data_workers=cfg.get('data_workers', 4))
        print(f'Built ZS caches under {cfg["zs_cache_dir"]}', flush=True)
        return

    from gentra.config.paths import output_root
    exp = cfg.get('experiment_name', 'pretrain_7source')
    out_base = cli.output_dir or str(output_root() / exp)
    seeds = cli.seeds if cli.seeds is not None else [int(cfg.get('seed', 1))]
    for seed in seeds:
        seed_dir = os.path.join(out_base, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        save_config_snapshot(cfg, seed_dir)
        run_training(cfg, seed_dir, seed=seed, gpuid=cli.gpuid)


if __name__ == '__main__':
    main()

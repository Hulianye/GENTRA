#!/usr/bin/env python3
"""Fine-tune a multi-source MI pretrained checkpoint on BNCI downstream tasks."""

import argparse
import copy
import gc
import json
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from gentra.models.loader import GENTRAModel, GENTRAPreprocessor
from gentra.pipelines.pretrain_runner import model_kwargs_from_config
from gentra.training.channel_encoding import channels_to_ids, vocab_size_from_vocab
from gentra.data.data_io import resolve_data_path
from gentra.training.dataset_split import create_dataloaders, split_dataset_cross, split_dataset_fewshot
from gentra.data.eeg_data import load_eeg_data_from_pkl
from gentra.training.logger import Logger
from gentra.data.preprocess_cache import load_preprocessed, save_preprocessed
from gentra.training.trainer import evaluate_model, train_model


from gentra.config.paths import project_root

PROJECT_ROOT = str(project_root())

CACHE_METADATA_ATTR = '_gentra_downstream_preprocess_metadata'
CACHE_METADATA_VERSION = 1


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
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if not cudnn_benchmark:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def load_pretrained_bundle(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('model_state', 'vocab', 'cfg'):
        if key not in ckpt:
            raise KeyError(f'Checkpoint missing required key: {key}')
    return {
        'state': ckpt['model_state'],
        'vocab': ckpt['vocab'],
        'cfg': {**ckpt['cfg'], 'seed': ckpt.get('seed', ckpt['cfg'].get('seed'))},
        'seed': ckpt.get('seed'),
        'epoch': ckpt.get('epoch'),
    }


def _filter_eeg_data_classes(eeg_data, classes):
    y = np.asarray(eeg_data.labels).reshape(-1)
    mask = np.ones(y.shape[0], dtype=bool) if classes is None else np.isin(
        y, np.asarray(classes, dtype=y.dtype)
    )
    eeg_data = copy.deepcopy(eeg_data)
    eeg_data.eeg_data = np.asarray(eeg_data.eeg_data)[mask]
    filtered_y = y[mask]
    class_ids = np.unique(filtered_y)
    if not np.array_equal(class_ids, np.arange(len(class_ids), dtype=class_ids.dtype)):
        remap = {old: new for new, old in enumerate(class_ids.tolist())}
        filtered_y = np.asarray([remap[int(v)] for v in filtered_y], dtype=np.int64)
    eeg_data.labels = filtered_y
    eeg_data.subject_ids = np.asarray(eeg_data.subject_ids).reshape(-1)[mask]
    if hasattr(eeg_data, 'usage') and eeg_data.usage is not None:
        eeg_data.usage = np.asarray(eeg_data.usage)[mask]
    return eeg_data


def build_training_args(cfg, overrides):
    merged = {**cfg, **overrides}
    defaults = {
        'seeds': [0],
        'gpuid': 0,
        'data_path': None,
        'label_smoothing': 0.0,
        'clip_grad_norm': 1.0,
        'finetune_strategy': 'full',
        'class_weights': False,
        'checkpoint_metric': 'balanced_accuracy',
        'quiet': False,
        'patience': 0,
        'cudnn_benchmark': False,
        'use_amp': False,
        'moe_gate_log': False,
        'moe_gate_lr_scale': 1.0,
        'data_workers': 1,
        'dataloader_workers': 0,
        'dropout_rate': 0.5,
        'optimizer_type': 'adamw',
        'lr': 2e-4,
        'weight_decay': 0.001,
        'use_lr_scheduler': False,
        'use_wd_scheduler': False,
        'lr_warmup_epochs': 0,
        'lr_decay_start_epoch': 1,
        'min_lr_ratio': 0.01,
        'wd_decay_start_epoch': 1,
        'final_weight_decay': None,
        'filter_bias_and_bn': True,
        'opt_eps': 1e-8,
        'b1': 0.9,
        'b2': 0.999,
        'target_fs': 250,
        'l_freq': 8.0,
        'h_freq': 32.0,
        'norm_method': 'car',
        'time_length': 4.0,
        'apply_EA': True,
        'use_channel_encoding': True,
    }
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    if merged.get('dropout_rate') is None:
        merged['dropout_rate'] = overrides.get('dropout_rate', 0.5)
    if merged.get('final_weight_decay') is None:
        merged['final_weight_decay'] = merged['weight_decay']
    # The loader passed to train_model is a TEST loader, never a validation set.
    merged['checkpoint_metric'] = 'final_epoch'
    merged['primary_checkpoint_metric'] = 'final_epoch'
    merged['patience'] = 0
    return SimpleNamespace(**merged)


def expected_cache_metadata(args):
    return {
        'metadata_version': CACHE_METADATA_VERSION,
        'dataset': args.dataset,
        'task_mode': args.task_mode,
        'train_percentage': float(args.train_percentage),
        'apply_EA': bool(args.apply_EA),
        'target_fs': args.target_fs,
        'time_length': args.time_length,
        'l_freq': args.l_freq,
        'h_freq': args.h_freq,
        'norm_method': (args.norm_method or 'none').lower(),
        'train_classes': list(args.train_classes) if args.train_classes else None,
        'num_classes': int(args.num_classes),
    }


def validate_cache_metadata(eeg_data, expected, cache_path):
    actual = getattr(eeg_data, CACHE_METADATA_ATTR, None)
    if actual is None:
        raise ValueError(
            f'Cache has no downstream preprocessing metadata: {cache_path}. '
            'Rebuild it with --build_preprocess_cache_only True.'
        )
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ', '.join(
            f'{key}: cached={cached!r}, expected={wanted!r}'
            for key, (cached, wanted) in mismatches.items()
        )
        raise ValueError(f'Preprocessed cache identity mismatch for {cache_path}: {details}')


def load_and_preprocess(args):
    cache_metadata = expected_cache_metadata(args)
    if args.preprocessed_cache and os.path.exists(args.preprocessed_cache):
        eeg_data = load_preprocessed(args.preprocessed_cache)
        validate_cache_metadata(eeg_data, cache_metadata, args.preprocessed_cache)
        split_method = getattr(eeg_data, 'split_method', 'LOSO')
        return eeg_data, split_method, None

    data_path = resolve_data_path(args.data_path, dataset_name=args.dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Dataset not found: {data_path}. Provide an EEGData bundle in data/downstream_raw/; see docs/data.md'
        )
    eeg_data = load_eeg_data_from_pkl(data_path)
    split_method = getattr(eeg_data, 'split_method', 'LOSO')

    train_classes = getattr(args, 'train_classes', None)
    if train_classes:
        eeg_data = _filter_eeg_data_classes(eeg_data, train_classes)

    preprocessor = GENTRAPreprocessor(
        target_fs=args.target_fs,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        normalize_method=args.norm_method,
        time_length=args.time_length,
        apply_EA=args.apply_EA,
        n_workers=args.data_workers,
    )
    eeg_data = copy.deepcopy(eeg_data)
    eeg_data.eeg_data = preprocessor.preprocess(
        eeg_data,
        task_mode=args.task_mode,
        train_percentage=args.train_percentage,
    )
    eeg_data.sampling_rate = preprocessor.target_fs

    if args.preprocessed_cache:
        setattr(eeg_data, CACHE_METADATA_ATTR, cache_metadata)
        save_preprocessed(eeg_data, args.preprocessed_cache)

    return eeg_data, split_method, preprocessor


def _state_dict_for_model(pretrained_state, model):
    """Load only keys with matching shapes (skip 2-class head when finetuning 4-class)."""
    current = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in pretrained_state.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped


def load_pretrained_model(eeg_data, bundle, args):
    vocab = bundle['vocab']
    model_kwargs = model_kwargs_from_config(bundle['cfg'])
    model_kwargs['dropout_rate'] = args.dropout_rate
    model = GENTRAModel(
        eeg_data,
        finetune_strategy=args.finetune_strategy,
        channel_vocab=vocab,
        channel_vocab_size=vocab_size_from_vocab(vocab),
        **model_kwargs,
    )
    filtered_state, skipped = _state_dict_for_model(bundle['state'], model)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    head_prefix = 'main_model.head.'
    skipped_head = [k for k in skipped if k.startswith(head_prefix)]
    other_skipped = [k for k in skipped if not k.startswith(head_prefix)]
    other_missing = [k for k in missing if not k.startswith(head_prefix)]
    if other_skipped and not getattr(args, 'quiet', False):
        print(f'Warning: skipped shape-mismatch keys ({len(other_skipped)}): {other_skipped[:5]}')
    if other_missing and not getattr(args, 'quiet', False):
        print(f'Warning: missing non-head keys ({len(other_missing)}): {other_missing[:5]}')
    if skipped_head and not getattr(args, 'quiet', False):
        print(f'Head keys skipped ({len(skipped_head)}); using freshly initialized head.')
    if unexpected and not getattr(args, 'quiet', False):
        print(f'Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}')
    if hasattr(model, 'apply_finetune_strategy') and getattr(args, 'finetune_strategy', None):
        model.apply_finetune_strategy(args.finetune_strategy)
    return model, vocab


def log_channel_vocab_coverage(eeg_data, vocab, logger=None):
    names = list(eeg_data.channel_names)
    ids = channels_to_ids(names, vocab, strict=True)
    unk = [names[i] for i, cid in enumerate(ids) if cid == 0]
    msg = f'Channel vocab coverage: {len(names) - len(unk)}/{len(names)} mapped, UNK={len(unk)}'
    if unk:
        msg += f' ({", ".join(unk[:8])}{"..." if len(unk) > 8 else ""})'
    if logger:
        logger.log(msg)
    elif not getattr(logger, 'quiet', True):
        print(msg)
    return unk


def run_single_seed(seed, args, eeg_data, split_method, device, output_dir, bundle):
    set_seed(seed, cudnn_benchmark=getattr(args, 'cudnn_benchmark', False))
    args.seed = int(seed)
    args.pretrain_seed = bundle.get('seed')
    os.makedirs(output_dir, exist_ok=True)
    config_snapshot = copy.deepcopy(vars(args))
    logger = Logger(verbose=not getattr(args, 'quiet', False))
    vocab = bundle['vocab']
    use_channel_encoding = True

    if args.task_mode == 'Cross':
        splits = split_dataset_cross(eeg_data, split_method)
        effective_split = split_method
    else:
        splits = split_dataset_fewshot(eeg_data, train_percentage=args.train_percentage)
        effective_split = f'Fewshot-{int(args.train_percentage * 100)}%'

    max_splits = getattr(args, 'max_splits', 0) or 0
    if max_splits > 0:
        splits = splits[:max_splits]

    split_results = []
    seed_metrics = {}
    pretrain_epoch = bundle.get('epoch')

    for split_idx, split in enumerate(splits):
        if not getattr(args, 'quiet', False):
            logger.log(f'\n===== Split {split_idx + 1}/{len(splits)} =====')
            if 'test_subject' in split:
                logger.log(f'Test subject: {split["test_subject"]}')

        train_loader, test_loader = create_dataloaders(
            eeg_data,
            split['train_mask'],
            split['test_mask'],
            args.batch_size,
            device,
            num_workers=args.dataloader_workers,
            use_channel_encoding=use_channel_encoding,
            channel_vocab=vocab,
        )

        model, _vocab = load_pretrained_model(eeg_data, bundle, args)
        model = model.to(device)
        if split_idx == 0:
            log_channel_vocab_coverage(eeg_data, vocab, logger)

        trained_model, best_val_acc, train_log, best_val_metrics = train_model(
            model, train_loader, test_loader, device, args, logger
        )

        assert args.checkpoint_metric == 'final_epoch'
        assert train_log['epochs'][-1] == args.epochs
        final_metrics = dict(best_val_metrics)
        # Saving predictions must not consume RNG used by the next fold.
        with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
            _, probabilities, labels = evaluate_model(
                trained_model, test_loader, torch.nn.CrossEntropyLoss(), device,
                quiet=True, return_predictions=True,
            )
        fold_dir = os.path.join(output_dir, f'fold_{split_idx:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        checkpoint_path = os.path.join(fold_dir, 'checkpoint_final.pt')
        if os.path.exists(checkpoint_path):
            raise FileExistsError(f'Refusing to overwrite {checkpoint_path}')
        torch.save({
            'model_state': {k: v.detach().cpu() for k, v in trained_model.state_dict().items()},
            'cfg': config_snapshot, 'vocab': vocab, 'epoch': args.epochs,
            'seed': int(seed), 'pretrain_seed': bundle.get('seed'),
            'checkpoint_metric': 'final_epoch', 'metrics': final_metrics,
        }, checkpoint_path + '.tmp')
        os.replace(checkpoint_path + '.tmp', checkpoint_path)
        test_indices = np.flatnonzero(split['test_mask'])
        assert np.array_equal(labels, np.asarray(eeg_data.labels)[test_indices])
        np.savez_compressed(os.path.join(fold_dir, 'predictions_final.npz'),
            probabilities=probabilities, labels=labels, predictions=probabilities.argmax(1),
            test_indices=test_indices, train_indices=np.flatnonzero(split['train_mask']),
            subject_ids=np.asarray(eeg_data.subject_ids)[test_indices])

        split_results.append({
            'split_idx': split_idx,
            'test_subject': split.get('test_subject'),
            'fold_idx': split.get('fold_idx'),
            'selected_epoch': args.epochs,
            'checkpoint_metric': 'final_epoch',
            'checkpoint_path': checkpoint_path,
            'final_metrics': final_metrics,
            'final_train_loss': train_log['loss'][-1],
            'last_epoch_test_metrics': train_log['metrics'][-1],
            'train_size': int(np.sum(split['train_mask'])),
            'test_size': int(np.sum(split['test_mask'])),
            'train_class_counts': {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(
                    np.asarray(eeg_data.labels)[split['train_mask']], return_counts=True
                ))
            },
            'test_class_counts': {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(
                    np.asarray(eeg_data.labels)[split['test_mask']], return_counts=True
                ))
            },
            'train_log': train_log,
        })
        logger.save_json(os.path.join(fold_dir, 'metrics.json'), split_results[-1])
        logger.save_json(os.path.join(output_dir, 'progress.json'), {
            'completed_splits': len(split_results), 'total_splits': len(splits),
            'checkpoint_metric': 'final_epoch', 'split_results': split_results,
        })

        for key, value in final_metrics.items():
            seed_metrics.setdefault(key, []).append(value)

        del model, trained_model, train_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    logger.experiment_dir = output_dir

    experiment_info = {
        'dataset': args.dataset,
        'model_name': 'GENTRA',
        'task_type': 'classification',
        'split_method': effective_split,
        'seed': seed,
        'pretrain_seed': bundle.get('seed'),
        'checkpoint_metric': 'final_epoch',
        'device': str(device),
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'pretrain_best_epoch': pretrain_epoch,
        'num_classes': int(getattr(args, 'num_classes', len(np.unique(eeg_data.labels)))),
        'train_classes': getattr(args, 'train_classes', None),
        'params': {
            k: (list(v) if isinstance(v, (list, tuple)) and k == 'train_classes' else v)
            for k, v in config_snapshot.items()
        },
        'channel_vocab_size': vocab_size_from_vocab(vocab),
    }
    logger.save_log(os.path.join(output_dir, 'training_log.txt'))
    results_payload = {
        **experiment_info,
        'split_results': split_results,
        'avg_metrics': {k: float(np.mean(v)) for k, v in seed_metrics.items()},
        'std_metrics': {k: float(np.std(v)) for k, v in seed_metrics.items()},
    }
    logger.save_json(os.path.join(output_dir, 'results.json'), results_payload)
    logger.log(f'\nSaved results to {output_dir}')
    return split_results, seed_metrics, results_payload


def build_parser():
    parser = argparse.ArgumentParser(description='Fine-tune pretrained multi-source MI checkpoint')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='BNCI2014001')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--task_mode', default='Cross', choices=['Cross', 'Fewshot'])
    parser.add_argument('--train_percentage', default=0.8, type=float)
    parser.add_argument('--num_classes', default=0, type=int, help='0 = infer from data')
    parser.add_argument('--train_classes', nargs='*', type=int, default=None,
                        help='Filter labels before preprocess, e.g. 0 1 for 2-class')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--seeds', default=[0], nargs='+', type=int)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--preprocessed_cache', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--timestamp', type=str, default=None)
    parser.add_argument('--build_preprocess_cache_only', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--max_splits', default=0, type=int, help='>0 limits splits for smoke tests')
    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_workers', default=1, type=int)
    parser.add_argument('--dataloader_workers', default=0, type=int)
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--lr', type=float, default=None, help='Override checkpoint cfg lr')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override checkpoint cfg weight_decay')
    parser.add_argument('--optimizer_type', type=str, default=None, choices=['adam', 'adamw'],
                        help='Override checkpoint cfg optimizer')
    parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--use_wd_scheduler', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--lr_warmup_epochs', type=int, default=None)
    parser.add_argument('--lr_decay_start_epoch', type=int, default=None)
    parser.add_argument('--min_lr_ratio', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--wd_decay_start_epoch', type=int, default=None)
    parser.add_argument('--final_weight_decay', type=float, default=None)
    parser.add_argument('--finetune_strategy', default=None, choices=['full'],
                        help='Override checkpoint cfg finetune_strategy')
    parser.add_argument('--label_smoothing', type=float, default=None, help='Override checkpoint cfg label_smoothing')
    parser.add_argument('--l_freq', type=float, default=None, help='Bandpass low cutoff Hz (default 8)')
    parser.add_argument('--h_freq', type=float, default=None, help='Bandpass high cutoff Hz (default 32)')
    parser.add_argument('--target_fs', type=float, default=None, help='Target sampling rate (default 250)')
    parser.add_argument('--time_length', type=float, default=None, help='Trial length in seconds (default 4)')
    parser.add_argument('--norm_method', type=str, default=None, help='Normalization: car, none, etc.')
    return parser


def main():
    parser = build_parser()
    cli = parser.parse_args()
    bundle = load_pretrained_bundle(cli.pretrained_checkpoint)

    scenario_overrides = {
        'dataset': cli.dataset,
        'data_path': cli.data_path,
        'task_mode': cli.task_mode,
        'train_percentage': cli.train_percentage,
        'batch_size': cli.batch_size,
        'epochs': cli.epochs,
        'dropout_rate': cli.dropout_rate,
        'seeds': cli.seeds,
        'gpuid': cli.gpuid,
        'preprocessed_cache': cli.preprocessed_cache,
        'max_splits': cli.max_splits,
        'quiet': cli.quiet,
        'data_workers': cli.data_workers,
        'dataloader_workers': cli.dataloader_workers,
        'use_amp': cli.use_amp,
        'cudnn_benchmark': cli.cudnn_benchmark,
        'pretrained_checkpoint': cli.pretrained_checkpoint,
        'num_classes': cli.num_classes,
        'train_classes': cli.train_classes,
        'build_preprocess_cache_only': cli.build_preprocess_cache_only,
    }
    for key in (
        'lr', 'weight_decay', 'optimizer_type', 'finetune_strategy',
        'use_lr_scheduler', 'use_wd_scheduler', 'lr_warmup_epochs',
        'lr_decay_start_epoch', 'min_lr_ratio', 'min_lr', 'wd_decay_start_epoch',
        'final_weight_decay', 'label_smoothing', 'l_freq', 'h_freq', 'target_fs',
        'time_length', 'norm_method',
    ):
        value = getattr(cli, key, None)
        if value is not None:
            scenario_overrides[key] = value
    args = build_training_args(bundle['cfg'], scenario_overrides)

    if args.build_preprocess_cache_only:
        if not args.preprocessed_cache:
            raise ValueError('--preprocessed_cache is required with --build_preprocess_cache_only')
        eeg_data, split_method, _ = load_and_preprocess(args)
        print(
            f'Built cache: {args.preprocessed_cache}, shape={eeg_data.eeg_data.shape}, '
            f'labels={np.unique(eeg_data.labels)}, split={split_method}'
        )
        return

    timestamp = cli.timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpuid}')
        torch.cuda.set_device(args.gpuid)
    else:
        device = torch.device('cpu')

    eeg_data, split_method, preprocessor = load_and_preprocess(args)
    inferred_classes = len(np.unique(eeg_data.labels))
    if not args.num_classes:
        args.num_classes = inferred_classes
    elif args.num_classes != inferred_classes:
        print(
            f'Warning: num_classes={args.num_classes} but data has {inferred_classes} unique labels'
        )

    if not getattr(args, 'quiet', False):
        print('\n===== Preprocessed dataset =====')
        print(f'Shape: {eeg_data.eeg_data.shape}, fs: {eeg_data.sampling_rate} Hz')
        print(f'Labels: {np.unique(eeg_data.labels)}, split: {split_method}')
        ea_info = preprocessor.apply_EA if preprocessor is not None else args.apply_EA
        print(f'Preprocessor EA: {ea_info}')
        print(f'Pretrain checkpoint epoch: {bundle.get("epoch")}')

    all_seed_metrics = {}
    for seed in args.seeds:
        if not getattr(args, 'quiet', False):
            print(f'\n\n===== Seed {seed} =====')
        output_dir = cli.output_dir or os.path.join(
            PROJECT_ROOT,
            'outputs',
            'hparam_search',
            f'pretrained_finetune_{timestamp}',
            f'seed_{seed}',
        )
        _, seed_metrics, _ = run_single_seed(
            seed, args, eeg_data, split_method, device, output_dir, bundle
        )
        for key, values in seed_metrics.items():
            all_seed_metrics.setdefault(key, []).extend(values)

    if not getattr(args, 'quiet', False):
        print('\n===== Overall (all splits, all seeds) =====')
        for key, values in all_seed_metrics.items():
            print(f'{key}: {np.mean(values):.4f} ± {np.std(values):.4f}')


if __name__ == '__main__':
    main()

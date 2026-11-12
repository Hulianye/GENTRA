"""Offline zero-shot evaluation from checkpoint_final.pt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gentra.config.build import build_pretrain_config, zs_pre_cfg_from_config
from gentra.config.paths import output_root
from gentra.data.multi_source_mi import SourceBundle
from gentra.data.zs_test_cache import build_all_zs_tests
from gentra.evaluation.loo_loader import load_loo_preprocessed_bundle
from gentra.evaluation.multi_source_eval import evaluate_by_subject
from gentra.evaluation.zs_eval_protocol import resolve_zs_eval_class_indices
from gentra.models.loader import GENTRAModel
from gentra.pipelines.pretrain_runner import model_kwargs_from_config, make_dummy_eeg
from gentra.models.masked_reconstruction import build_reconstruction_head
from gentra.training.channel_encoding import vocab_size_from_vocab
from gentra.pipelines.finetune_runner import set_seed


def _test_entry_for_name(cfg: dict, name: str) -> tuple:
    for entry in cfg.get('zs_test_specs', []):
        if entry[0] == name:
            return entry
    return (name, None, None, False, None)


def main():
    parser = argparse.ArgumentParser(description='GENTRA zero-shot eval')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = dict(ckpt.get('cfg') or {})
    release_cfg = build_pretrain_config(args.config)
    if not cfg:
        cfg = release_cfg
    else:
        for key in ('pkl_cache_root', 'cache_dir', 'zs_cache_dir', 'zs_test_specs'):
            cfg[key] = release_cfg[key]
    vocab = ckpt.get('vocab')
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    set_seed(ckpt.get('seed', 0), cudnn_benchmark=False)

    model_kwargs = model_kwargs_from_config(cfg)
    dummy = make_dummy_eeg(vocab, n_classes=len(cfg['train_classes']))
    model = GENTRAModel(dummy, channel_vocab=vocab, **model_kwargs).to(device)
    if any(k.startswith('recon_head.') for k in ckpt['model_state']):
        model.recon_head = build_reconstruction_head(
            cfg['recon_target'], emb_size=int(cfg['emb_size']),
            conv_dim=int(cfg['conv_filters']), channel_vocab_size=vocab_size_from_vocab(vocab),
            recon_hidden_dim=int(cfg.get('recon_hidden_dim', 128)),
            samples_per_token=int(cfg.get('recon_samples_per_token', 16)),
        ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    zs_pre = zs_pre_cfg_from_config(cfg)
    zs_bundles = build_all_zs_tests(
        cfg.get('zs_test_specs', []),
        zs_pre,
        vocab,
        cfg['zs_cache_dir'],
        data_workers=cfg.get('data_workers', 4),
    )
    loo_classes = {}
    for src in cfg.get('zs_loo_eval_sources', []):
        x, y, sid, ch, _allowed = load_loo_preprocessed_bundle(src, cfg)
        zs_bundles[src] = SourceBundle(src, x, y, sid, ch, 0)
        loo_classes[src] = tuple((cfg.get('zs_loo_eval_class_indices') or {}).get(src, _allowed))

    rows = {}
    for name, bundle in zs_bundles.items():
        entry = _test_entry_for_name(cfg, name)
        idx = resolve_zs_eval_class_indices(name, entry, model, cfg)
        if name in loo_classes:
            idx = loo_classes[name]
        metrics = evaluate_by_subject(
            model,
            bundle.x,
            bundle.y,
            bundle.subject_ids,
            bundle.channel_names,
            vocab,
            device,
            eval_class_indices=idx,
        )
        rows[name] = float(metrics['average_subject_balanced_accuracy'])

    mean4_datasets = cfg.get('mean4_datasets', list(rows.keys()))
    vals = [rows[d] for d in mean4_datasets if d in rows]
    out = {
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ckpt.get('epoch'),
        'pretrain_seed': ckpt.get('seed'),
        'class_filter_order': 'before_preprocessing_and_EA',
        'datasets': rows,
        'mean4_bac': sum(vals) / len(vals) if vals else None,
        'min4_bac': min(vals) if vals else None,
    }
    out_path = Path(args.output or output_root() / 'zs_eval.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

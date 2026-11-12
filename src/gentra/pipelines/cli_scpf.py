"""Frozen five-shot SCPF evaluation on downstream preprocessing scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from gentra.data.multi_source_mi import SourceBundle
from gentra.data.multi_source_dataset import MultiSourceChannelEEGDataset, multi_source_collate_fn
from gentra.evaluation.scpf import scpf_logits
from gentra.models.loader import GENTRAModel
from gentra.models.masked_reconstruction import build_reconstruction_head
from gentra.pipelines.cli_finetune import all_scenarios, cache_path, load_finetune_config
from gentra.pipelines.finetune_runner import build_training_args, load_and_preprocess, load_pretrained_bundle
from gentra.pipelines.pretrain_runner import make_dummy_eeg, model_kwargs_from_config
from gentra.training.channel_encoding import unpack_batch, vocab_size_from_vocab
from gentra.training.dataset_split import split_dataset_fewshot


def _model(bundle, device):
    cfg, vocab = bundle['cfg'], bundle['vocab']
    model = GENTRAModel(
        make_dummy_eeg(vocab, n_classes=len(cfg['train_classes'])),
        channel_vocab=vocab, **model_kwargs_from_config(cfg),
    ).to(device)
    if any(k.startswith('recon_head.') for k in bundle['state']):
        model.recon_head = build_reconstruction_head(
            cfg['recon_target'], emb_size=int(cfg['emb_size']),
            conv_dim=int(cfg['conv_filters']),
            channel_vocab_size=vocab_size_from_vocab(vocab),
            recon_hidden_dim=int(cfg.get('recon_hidden_dim', 128)),
            samples_per_token=int(cfg.get('recon_samples_per_token', 16)),
        ).to(device)
    model.load_state_dict(bundle['state'])
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def _embeddings(model, eeg, vocab, device, batch_size=64):
    bundle = SourceBundle(eeg.dataset_name, np.asarray(eeg.eeg_data, dtype=np.float32),
                          np.asarray(eeg.labels), np.asarray(eeg.subject_ids),
                          list(eeg.channel_names), 0)
    subject_ids = np.asarray(eeg.subject_ids)
    result = None
    head = model.prototype_head()
    for subject in sorted(np.unique(subject_ids), key=float):
        indices = np.flatnonzero(subject_ids == subject)
        ds = MultiSourceChannelEEGDataset([bundle], vocab, indices)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=multi_source_collate_fn)
        chunks = []
        for batch in loader:
            inputs, _, channel_ids, channel_mask = unpack_batch(batch, device)
            chunks.append(head.encode(model.encode_features(inputs, channel_ids, channel_mask)).cpu().numpy())
        values = np.vstack(chunks).astype(np.float32)
        if result is None:
            result = np.empty((len(subject_ids), values.shape[1]), dtype=np.float32)
        result[indices] = values
    return result


def evaluate_scpf_embeddings(embeddings, labels, subject_ids, head, scenario,
                             support_seeds=range(20), usage=None):
    """Evaluate disjoint support/query sets using the released SCPF fusion rule."""
    support_seeds = tuple(int(seed) for seed in support_seeds)
    if not support_seeds:
        raise ValueError('At least one support seed is required')
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError('Embeddings must be finite')
    labels = np.asarray(labels)
    subject_ids = np.asarray(subject_ids)
    classes = tuple(int(v) for v in np.unique(labels))
    heads = (1, 2) if scenario['dataset'] == 'BNCI2015001' else classes
    if len(classes) != scenario['num_classes'] or len(heads) != len(classes):
        raise ValueError('Class mapping does not match scenario')
    proto = head.prototypes.detach().cpu().numpy().astype(np.float32)
    checkpoint = {h: proto[h] for h in heads}
    novel = (3,) if scenario['dataset'] == 'BNCI2014001' and len(classes) == 4 else ()
    if scenario['task_mode'] == 'Fewshot':
        from gentra.data.eeg_data import EEGData
        # Only labels and subject IDs are used by split_dataset_fewshot.
        mock = EEGData(scenario['dataset'], np.empty((len(labels), 1, 1), dtype=np.float32),
                       subject_ids, ['C3'], 1, labels=labels, usage=usage)
        splits = split_dataset_fewshot(mock, train_percentage=scenario['train_percentage'])
        train = np.zeros(len(labels), dtype=bool)
        test = np.zeros(len(labels), dtype=bool)
        for split in splits:
            train |= np.asarray(split['train_mask'], dtype=bool)
            test |= np.asarray(split['test_mask'], dtype=bool)
        if np.any(train & test) or not np.all(train | test):
            raise ValueError('Fewshot split does not partition all trials')
    else:
        train = test = None
    rows = []
    for seed in support_seeds:
        rng = np.random.default_rng(int(seed) + 1000)
        for subject in sorted(np.unique(subject_ids), key=float):
            selected = {}
            for data_class, head_class in zip(classes, heads):
                eligible = (labels == data_class)
                if train is not None:
                    eligible &= train & (subject_ids == subject)
                else:
                    eligible &= subject_ids != subject
                pool = np.flatnonzero(eligible)
                if len(pool) < 5:
                    raise ValueError(f'{scenario["name"]}: fewer than five support trials for class {data_class}')
                selected[head_class] = np.sort(rng.choice(pool, size=5, replace=False))
            query = np.flatnonzero((subject_ids == subject) & (test if test is not None else True))
            if np.intersect1d(query, np.concatenate(list(selected.values()))).size:
                raise ValueError('Support/query overlap')
            logits = scpf_logits(embeddings[query], embeddings, selected, checkpoint,
                                 heads, float(head.temperature), alpha=0.75,
                                 novel_class_ids=novel)
            truth = np.array([classes.index(int(y)) for y in labels[query]])
            rows.append({'support_seed': int(seed), 'subject_id': float(subject),
                         'n_query': int(len(query)), 'balanced_accuracy': float(
                             balanced_accuracy_score(truth, logits.argmax(axis=1)))})
    by_seed = [{'support_seed': int(seed), 'balanced_accuracy': float(np.mean([
        r['balanced_accuracy'] for r in rows if r['support_seed'] == seed]))}
        for seed in support_seeds]
    return {'scenario': scenario['name'], 'support_seeds': list(support_seeds),
            'mean_balanced_accuracy': float(np.mean([r['balanced_accuracy'] for r in by_seed])),
            'per_support_seed': by_seed, 'per_subject': rows,
            'support_protocol': 'other_subjects' if train is None else 'within_subject_30pct_pool'}


def main():
    parser = argparse.ArgumentParser(description='Evaluate frozen SCPF on a downstream scenario')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--raw-root', default=None)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--support-seeds', nargs='+', type=int, default=list(range(20)))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scenarios = all_scenarios(load_finetune_config(args.config))
    selected = [s for s in scenarios if s['name'] == args.scenario]
    if len(selected) != 1:
        parser.error(f'Unknown scenario: {args.scenario}')
    sc = selected[0]
    if args.raw_root:
        sc['data_path'] = str(Path(args.raw_root) / f"{sc['dataset']}.pkl")
    else:
        from gentra.config.paths import downstream_raw_dir
        sc['data_path'] = str(downstream_raw_dir() / f"{sc['dataset']}.pkl")
    sc['preprocessed_cache'] = cache_path(sc)
    bundle = load_pretrained_bundle(args.checkpoint)
    eeg, _, _ = load_and_preprocess(build_training_args(bundle['cfg'], sc))
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    model = _model(bundle, device)
    z = _embeddings(model, eeg, bundle['vocab'], device)
    result = evaluate_scpf_embeddings(z, eeg.labels, eeg.subject_ids,
                                      model.prototype_head(), sc, tuple(args.support_seeds),
                                      usage=getattr(eeg, 'usage', None))
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_subject',)}, indent=2))


if __name__ == '__main__':
    main()

"""Load, preprocess, and merge MI training sources."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from gentra.training.channel_encoding import build_channel_vocab
from gentra.data.eeg_data import EEGData, load_eeg_data_from_pkl
from gentra.models.loader import GENTRAPreprocessor


@dataclass
class SourceBundle:
    name: str
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    channel_names: list[str]
    source_id: int


def load_source_pkl(path: str, dataset_name: str) -> EEGData:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_eeg_data_from_pkl(path)


def filter_train_classes(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray | None,
    classes=(0, 1),
):
    tc = np.asarray(classes, dtype=y.dtype)
    mask = np.isin(y, tc)
    x_out = x[mask]
    y_out = y[mask]
    sid_out = subject_ids[mask] if subject_ids is not None else None
    return x_out, y_out, sid_out


def filter_binary_classes(x, y, subject_ids, classes=(0, 1)):
    return filter_train_classes(x, y, subject_ids, classes=classes)


def _format_freq_tag(freq: float) -> str:
    value = float(freq)
    if value == int(value):
        return str(int(value))
    return str(value).replace('.', 'p')


def preprocess_band_tag(pre_cfg: dict) -> str:
    if pre_cfg.get('l_freq') is None and pre_cfg.get('h_freq') is None:
        return 'nofilter'
    return f"{_format_freq_tag(pre_cfg['l_freq'])}-{_format_freq_tag(pre_cfg['h_freq'])}"


def _channel_cache_tag(pre_cfg: dict) -> str:
    whitelist = pre_cfg.get('channel_whitelist')
    if not whitelist:
        return 'allch'
    encoded = json.dumps(list(whitelist), ensure_ascii=True, separators=(',', ':'))
    digest = hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:10]
    return f'ch{len(whitelist)}_{digest}'


def _time_length_cache_tag(pre_cfg: dict) -> str:
    tl = pre_cfg.get('time_length')
    if tl is None:
        return 'orig'
    return f'tl{_format_freq_tag(float(tl))}'


def preprocess_cache_suffix(pre_cfg: dict) -> str:
    band = preprocess_band_tag(pre_cfg)
    norm = pre_cfg.get('norm_method', 'car')
    ch_tag = _channel_cache_tag(pre_cfg)
    tl_tag = _time_length_cache_tag(pre_cfg)
    target_fs = pre_cfg.get('target_fs')
    fs_tag = 'fsorig' if target_fs is None else f'fs{_format_freq_tag(target_fs)}'
    ea_tag = f'ea{int(bool(pre_cfg.get("apply_EA", False)))}'
    return f'v3_{ea_tag}_{fs_tag}_{tl_tag}_{norm}_{band}_{ch_tag}'


def preprocess_cache_identity(pre_cfg: dict) -> dict:
    return {
        # Invalidate caches produced before channel/time validation.
        'preprocessing_version': 3,
        'apply_EA': bool(pre_cfg.get('apply_EA', False)),
        'target_fs': pre_cfg.get('target_fs'),
        'time_length': pre_cfg.get('time_length'),
        'l_freq': pre_cfg.get('l_freq'),
        'h_freq': pre_cfg.get('h_freq'),
        'norm_method': pre_cfg.get('norm_method', 'car'),
        'channel_whitelist': pre_cfg.get('channel_whitelist'),
    }


def mi_cache_path(cache_dir: str, source_name: str, pre_cfg: dict, suffix: str = '') -> str:
    os.makedirs(cache_dir, exist_ok=True)
    tag = preprocess_cache_suffix(pre_cfg)
    tag = f'{tag}{suffix}'
    return os.path.join(cache_dir, f'{source_name}_{tag}.pkl')


def preprocess_source(
    eeg_data: EEGData | None,
    pre_cfg: dict,
    cache_path: str | None = None,
    data_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if isinstance(cached, dict):
            cached_identity = cached.get('preprocess_config')
            if cached_identity is None:
                raise ValueError(f'Cache lacks preprocessing identity; remove and rebuild: {cache_path}')
            elif cached_identity != preprocess_cache_identity(pre_cfg):
                raise ValueError(f'Preprocessing cache identity mismatch: {cache_path}')
            x = np.asarray(cached['x'])
            y = np.asarray(cached['y'])
            sid = np.asarray(cached['subject_ids'])
            channel_names = list(cached.get('channel_names') or [])
            if x.ndim != 3 or y.shape[0] != x.shape[0] or sid.shape[0] != x.shape[0]:
                raise ValueError(f'Invalid preprocessing cache shapes: {cache_path}')
            if channel_names and len(channel_names) != x.shape[1]:
                raise ValueError(
                    f'Cache channel mismatch: data has {x.shape[1]} channels, '
                    f'names has {len(channel_names)}: {cache_path}'
                )
            target_fs = pre_cfg.get('target_fs')
            time_length = pre_cfg.get('time_length')
            if target_fs is not None and time_length is not None:
                expected_t = round(float(target_fs) * float(time_length))
                if x.shape[-1] != expected_t:
                    raise ValueError(
                        f'Cache time length mismatch: got {x.shape[-1]}, '
                        f'expected {expected_t}: {cache_path}'
                    )
            return x, y, sid
        raise TypeError(f'Unsupported preprocessing cache format; remove and rebuild: {cache_path}')

    if eeg_data is None:
        raise ValueError(f'preprocess_source requires eeg_data when cache is missing: {cache_path}')

    preprocessor = GENTRAPreprocessor(
        target_fs=pre_cfg['target_fs'],
        l_freq=pre_cfg['l_freq'],
        h_freq=pre_cfg['h_freq'],
        normalize_method=pre_cfg['norm_method'],
        time_length=pre_cfg.get('time_length'),
        apply_EA=pre_cfg.get('apply_EA', False),
        channel_whitelist=pre_cfg.get('channel_whitelist'),
        n_workers=data_workers,
    )
    data_copy = copy.deepcopy(eeg_data)
    x = preprocessor.preprocess(data_copy, task_mode='Cross', train_percentage=0.8)
    y = np.asarray(data_copy.labels).reshape(-1)
    sid = np.asarray(data_copy.subject_ids).reshape(-1)
    channel_names = list(data_copy.channel_names or eeg_data.channel_names or [])

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(
                {
                    'x': x,
                    'y': y,
                    'subject_ids': sid,
                    'channel_names': channel_names,
                    'preprocess_config': preprocess_cache_identity(pre_cfg),
                },
                f,
            )
    return x, y, sid


def _bundle_from_arrays(
    source_name: str,
    x: np.ndarray,
    y: np.ndarray,
    sid: np.ndarray | None,
    channel_names: list[str],
) -> SourceBundle:
    if sid is None:
        sid = np.zeros(len(y), dtype=np.int64)
    return SourceBundle(
        name=source_name,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=sid.astype(sid.dtype),
        channel_names=list(channel_names),
        source_id=-1,
    )


def load_and_preprocess_source(
    source_name: str,
    pkl_path: str,
    channel_names: list[str],
    pre_cfg: dict,
    cache_dir: str,
    train_classes=(0, 1),
    data_workers: int = 1,
) -> SourceBundle:
    cache_path = mi_cache_path(cache_dir, source_name, pre_cfg)
    if os.path.isfile(cache_path):
        x, y, sid = preprocess_source(None, pre_cfg, cache_path=cache_path, data_workers=data_workers)
        if pre_cfg.get('channel_whitelist'):
            channel_names = list(pre_cfg['channel_whitelist'])
        else:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('channel_names'):
                channel_names = list(cached['channel_names'])
    else:
        eeg = load_source_pkl(pkl_path, source_name)
        if not eeg.channel_names:
            eeg.channel_names = channel_names
        channel_names = list(eeg.channel_names)
        x, y, sid = preprocess_source(eeg, pre_cfg, cache_path=cache_path, data_workers=data_workers)
        if pre_cfg.get('channel_whitelist'):
            channel_names = list(pre_cfg['channel_whitelist'])
    x, y, sid = filter_train_classes(x, y, sid, classes=train_classes)
    return _bundle_from_arrays(source_name, x, y, sid, channel_names)


def resolve_preprocess_source_workers(n_sources: int, requested: int) -> int:
    if requested > 0:
        return min(requested, n_sources)
    cpu = os.cpu_count() or 4
    return max(1, min(n_sources, cpu // 4))


def preprocess_one_source_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker for ProcessPoolExecutor."""
    source_name = payload['source_name']
    t0 = time.perf_counter()
    bundle = load_and_preprocess_source(
        source_name=source_name,
        pkl_path=payload['pkl_path'],
        channel_names=payload['channel_names'],
        pre_cfg=payload['pre_cfg'],
        cache_dir=payload['cache_dir'],
        train_classes=tuple(payload.get('train_classes', (0, 1))),
        data_workers=payload.get('data_workers', 1),
    )
    elapsed = time.perf_counter() - t0
    return {
        'name': source_name,
        'x': bundle.x,
        'y': bundle.y,
        'subject_ids': bundle.subject_ids,
        'channel_names': bundle.channel_names,
        'raw_trials': int(len(bundle.y)),
        'preprocess_seconds': elapsed,
    }


def parallel_preprocess_sources(
    sources: list[str],
    source_pkl_fn,
    channel_names_fn,
    pre_cfg: dict,
    cache_dir: str,
    *,
    train_classes=(0, 1),
    data_workers: int = 1,
    preprocess_source_workers: int = 0,
    source_class_overrides: dict | None = None,
) -> list[SourceBundle]:
    n_workers = resolve_preprocess_source_workers(len(sources), preprocess_source_workers)
    print(
        f'[preprocess] source_workers={n_workers} data_workers={data_workers} '
        f'n_sources={len(sources)}'
    )
    payloads = [
        {
            'source_name': name,
            'pkl_path': source_pkl_fn(name),
            'channel_names': channel_names_fn(name),
            'pre_cfg': pre_cfg,
            'cache_dir': cache_dir,
            'train_classes': list(
                (source_class_overrides or {}).get(name, {}).get('train_classes', train_classes)
            ),
            'data_workers': data_workers,
        }
        for name in sources
    ]
    if n_workers <= 1 or len(payloads) <= 1:
        results = [preprocess_one_source_worker(p) for p in payloads]
    else:
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(preprocess_one_source_worker, p): p['source_name'] for p in payloads}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    out = fut.result()
                    results.append(out)
                    print(
                        f'[preprocess] done source={name} trials={out["raw_trials"]} '
                        f'elapsed={out["preprocess_seconds"]:.1f}s'
                    )
                except Exception as exc:
                    raise RuntimeError(f'preprocess failed for source={name}: {exc}') from exc

    name_to_result = {r['name']: r for r in results}
    bundles = []
    for name in sources:
        r = name_to_result[name]
        bundles.append(
            _bundle_from_arrays(name, r['x'], r['y'], r['subject_ids'], r['channel_names'])
        )
    return bundles


def merge_sources(bundles: list[SourceBundle]) -> tuple[list[SourceBundle], dict[str, int]]:
    all_channel_lists = []
    for i, b in enumerate(bundles):
        b.source_id = i
        all_channel_lists.append(b.channel_names)
    vocab = build_channel_vocab(all_channel_lists)
    return bundles, vocab


def build_multi_source_pool(
    sources: list[str],
    source_pkl_fn,
    channel_names_fn,
    pre_cfg: dict,
    cache_dir: str,
    train_classes=(0, 1),
    data_workers: int = 1,
    preprocess_source_workers: int = 0,
    source_class_overrides: dict | None = None,
) -> tuple[list[SourceBundle], dict[str, int]]:
    bundles = parallel_preprocess_sources(
        sources,
        source_pkl_fn,
        channel_names_fn,
        pre_cfg,
        cache_dir,
        train_classes=train_classes,
        data_workers=data_workers,
        preprocess_source_workers=preprocess_source_workers,
        source_class_overrides=source_class_overrides,
    )
    bundles, vocab = merge_sources(bundles)
    return bundles, vocab

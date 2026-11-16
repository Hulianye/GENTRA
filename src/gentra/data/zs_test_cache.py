"""Disk cache for zero-shot BNCI test arrays used in multi-source MI pretrain."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np

from gentra.training.channel_encoding import channels_to_ids
from gentra.data.multi_source_mi import (
    filter_binary_classes,
    load_source_pkl,
    preprocess_cache_identity,
    preprocess_cache_suffix,
    preprocess_source,
)


@dataclass(frozen=True)
class ZSTestSpec:
    name: str
    pkl_path: str
    channel_names: list[str]
    filter_binary: bool = False
    allowed_classes: tuple[int, ...] | None = None


@dataclass
class ZSTestBundle:
    name: str
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    channel_names: list[str]
    channel_ids: np.ndarray
    filter_binary: bool = False
    allowed_classes: tuple[int, ...] | None = None


def _allowed_classes_tag(allowed_classes: tuple[int, ...] | None) -> str:
    if not allowed_classes:
        return 'allcls'
    return 'cls' + '_'.join(str(c) for c in allowed_classes)


def zs_cache_path(
    cache_dir: str,
    dataset_name: str,
    pre_cfg: dict,
    filter_binary: bool,
    allowed_classes: tuple[int, ...] | None = None,
) -> str:
    suffix = '2cls' if filter_binary else _allowed_classes_tag(allowed_classes)
    tag = preprocess_cache_suffix(pre_cfg)
    return os.path.join(cache_dir, f'{dataset_name}_{tag}_classfirst_v1_{suffix}.pkl')


def _filter_allowed_classes(x, y, sid, allowed_classes: tuple[int, ...] | None):
    if not allowed_classes:
        return x, y, sid
    mask = np.isin(y, list(allowed_classes))
    return x[mask], y[mask], sid[mask]


def load_or_build_zs_test(
    spec: ZSTestSpec,
    pre_cfg: dict,
    vocab: dict[str, int],
    cache_dir: str,
    *,
    data_workers: int = 1,
) -> ZSTestBundle:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = zs_cache_path(
        cache_dir, spec.name, pre_cfg, spec.filter_binary, spec.allowed_classes,
    )
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict):
            raise TypeError(f'Unsupported zero-shot cache format; remove and rebuild: {cache_path}')
        cached_identity = cached.get('preprocess_config')
        if cached_identity is None:
            raise ValueError(f'Zero-shot cache lacks preprocessing identity; remove and rebuild: {cache_path}')
        elif cached_identity != preprocess_cache_identity(pre_cfg):
            raise ValueError(f'ZS preprocessing cache identity mismatch: {cache_path}')
        x = np.asarray(cached['x'], dtype=np.float32)
        y = np.asarray(cached['y'], dtype=np.int64)
        sid = np.asarray(cached['subject_ids'])
        channel_names = list(cached.get('channel_names') or spec.channel_names)
        if x.ndim != 3 or y.shape[0] != x.shape[0] or sid.shape[0] != x.shape[0]:
            raise ValueError(f'Invalid zero-shot cache shapes: {cache_path}')
        if len(channel_names) != x.shape[1]:
            raise ValueError(
                f'Zero-shot cache channel mismatch: data has {x.shape[1]} channels, '
                f'names has {len(channel_names)}: {cache_path}'
            )
        target_fs = pre_cfg.get('target_fs')
        time_length = pre_cfg.get('time_length')
        if target_fs is not None and time_length is not None:
            expected_t = round(float(target_fs) * float(time_length))
            if x.shape[-1] != expected_t:
                raise ValueError(
                    f'Zero-shot cache time length mismatch: got {x.shape[-1]}, '
                    f'expected {expected_t}: {cache_path}'
                )
    else:
        eeg = load_source_pkl(spec.pkl_path, spec.name)
        if not eeg.channel_names:
            eeg.channel_names = spec.channel_names
        # Define the evaluated task BEFORE estimating subject-level EA.
        mask = np.ones(len(eeg.labels), dtype=bool)
        if spec.filter_binary:
            mask &= np.isin(eeg.labels, (0, 1))
        if spec.allowed_classes:
            mask &= np.isin(eeg.labels, spec.allowed_classes)
        eeg.eeg_data = np.asarray(eeg.eeg_data)[mask]
        eeg.labels = np.asarray(eeg.labels)[mask]
        eeg.subject_ids = np.asarray(eeg.subject_ids)[mask]
        for key in ('usage',):
            value = getattr(eeg, key, None)
            if value is not None:
                setattr(eeg, key, np.asarray(value)[mask])
        x, y, sid = preprocess_source(
            eeg, pre_cfg, cache_path=None, data_workers=data_workers,
        )
        if spec.filter_binary:
            x, y, sid = filter_binary_classes(x, y, sid, classes=(0, 1))
        x, y, sid = _filter_allowed_classes(x, y, sid, spec.allowed_classes)
        if pre_cfg.get('channel_whitelist'):
            channel_names = list(pre_cfg['channel_whitelist'])
        else:
            channel_names = list(eeg.channel_names or spec.channel_names)
        with open(cache_path, 'wb') as f:
            pickle.dump(
                {
                    'x': x.astype(np.float32),
                    'y': y.astype(np.int64),
                    'subject_ids': sid,
                    'channel_names': channel_names,
                    'preprocess_config': preprocess_cache_identity(pre_cfg),
                    'class_filter_order': 'before_preprocessing_and_EA',
                },
                f,
            )
    channel_ids = channels_to_ids(channel_names, vocab, strict=True)
    return ZSTestBundle(
        name=spec.name,
        x=x.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=sid,
        channel_names=channel_names,
        channel_ids=channel_ids,
        filter_binary=spec.filter_binary,
        allowed_classes=spec.allowed_classes,
    )


def _parse_test_spec(entry) -> ZSTestSpec:
    if len(entry) >= 5:
        name, pkl_path, channel_names, filter_binary, allowed_classes = entry[:5]
    else:
        name, pkl_path, channel_names, filter_binary = entry
        allowed_classes = None
    return ZSTestSpec(
        name=name,
        pkl_path=pkl_path,
        channel_names=channel_names,
        filter_binary=filter_binary,
        allowed_classes=allowed_classes,
    )


def build_all_zs_tests(
    test_specs: list[tuple],
    pre_cfg: dict,
    vocab: dict[str, int],
    cache_dir: str,
    *,
    data_workers: int = 1,
) -> dict[str, ZSTestBundle]:
    bundles: dict[str, ZSTestBundle] = {}
    for entry in test_specs:
        spec = _parse_test_spec(entry)
        bundles[spec.name] = load_or_build_zs_test(
            spec, pre_cfg, vocab, cache_dir, data_workers=data_workers,
        )
    return bundles

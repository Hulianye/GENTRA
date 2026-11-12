"""Load LOO zero-shot bundles for held-out training sources (e.g. Zhou2016)."""

from __future__ import annotations

import pickle

import numpy as np

from gentra.config.build import source_preprocessed_cache_path
from gentra.data.multi_source_mi import preprocess_cache_identity
from gentra.data.registry import SOURCE_CLASS_SPECS
from gentra.data.source_channels import channel_names_for_source


def load_loo_preprocessed_bundle(source_name: str, cfg: dict | None = None):
    path = source_preprocessed_cache_path(source_name, cfg)
    if source_name not in SOURCE_CLASS_SPECS:
        raise KeyError(f'Unknown LOO source {source_name!r}')
    with open(path, 'rb') as f:
        cached = pickle.load(f)
    expected_identity = preprocess_cache_identity(
        {
            key: cfg[key]
            for key in ('l_freq', 'h_freq', 'target_fs', 'time_length', 'apply_EA', 'norm_method')
            if cfg is not None and key in cfg
        }
    ) if cfg is not None else None
    cached_identity = cached.get('preprocess_config') if isinstance(cached, dict) else None
    if expected_identity is not None and cached_identity != expected_identity:
        raise ValueError(f'LOO preprocessing cache identity mismatch: {path}')
    x = np.asarray(cached['x'], dtype=np.float32)
    y = np.asarray(cached['y'], dtype=np.int64)
    sid = np.asarray(cached['subject_ids'])
    channel_names = list(cached.get('channel_names') or channel_names_for_source(source_name))
    if x.ndim != 3 or y.shape[0] != x.shape[0] or sid.shape[0] != x.shape[0]:
        raise ValueError(f'Invalid LOO cache shapes: {path}')
    if len(channel_names) != x.shape[1]:
        raise ValueError(f'LOO cache channel mismatch: {path}')
    if cfg is not None and cfg.get('target_fs') is not None and cfg.get('time_length') is not None:
        expected_t = round(float(cfg['target_fs']) * float(cfg['time_length']))
        if x.shape[-1] != expected_t:
            raise ValueError(f'LOO cache time length mismatch: {path}')
    allowed = tuple(SOURCE_CLASS_SPECS[source_name]['allowed_classes'])
    mask = np.isin(y, list(allowed))
    return x[mask], y[mask], sid[mask], channel_names, allowed

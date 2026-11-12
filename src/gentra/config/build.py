"""Build flat training config dict from YAML."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from gentra.config.paths import (
    bnci_data_dir,
    pkl_root,
    train_cache_dir,
    zs_cache_dir,
)
from gentra.data.registry import (
    SOURCE_CLASS_SPECS,
    build_zs_test_specs,
    source_pkl_path,
)
from gentra.data.source_channels import channel_names_for_source


def _project_root() -> Path:
    from gentra.config.paths import project_root
    return project_root()


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_preprocess_cache_dirs(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg['cache_dir'] = str(train_cache_dir())
    cfg['zs_cache_dir'] = str(zs_cache_dir())
    cfg['pkl_cache_root'] = str(pkl_root())
    return cfg


def preprocess_cfg_from_config(cfg: dict) -> dict:
    keys = ('l_freq', 'h_freq', 'target_fs', 'time_length', 'apply_EA', 'norm_method')
    pre = {k: cfg[k] for k in keys if k in cfg}
    if 'channel_whitelist' in cfg:
        pre['channel_whitelist'] = cfg['channel_whitelist']
    return pre


def train_pre_cfg_from_config(cfg: dict) -> dict:
    return preprocess_cfg_from_config(cfg)


def zs_pre_cfg_from_config(cfg: dict) -> dict:
    pre = preprocess_cfg_from_config(cfg)
    pre.pop('channel_whitelist', None)
    if cfg.get('source_preprocessed'):
        return pre
    if cfg.get('zs_time_length') is not None:
        pre['time_length'] = float(cfg['zs_time_length'])
    elif pre.get('time_length') is None:
        pre['time_length'] = 4.0
    return pre


def _project_data_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else _project_root() / path)


def build_pretrain_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path or _project_root() / 'configs' / 'pretrain.yaml')
    raw = load_yaml(path)
    cfg: dict[str, Any] = {}
    cfg.update(raw.get('preprocess', {}))
    cfg.update(raw.get('model', {}))
    cfg.update(raw.get('loss', {}))
    cfg.update(raw.get('training', {}))
    cfg['sources'] = list(raw['data']['train_sources'])
    cfg['zs_loo_eval_sources'] = list(raw['data'].get('loo_eval_sources', []))
    cfg['mean4_datasets'] = list(raw['data'].get('mean4_datasets', []))
    cfg['zs_eval_class_indices'] = dict(raw['data'].get('zs_eval_class_indices', {}))
    cfg['source_class_overrides'] = {
        src: {'train_classes': classes}
        for src, classes in raw['data'].get('source_train_classes', {}).items()
    }
    cfg['train_classes'] = raw['data'].get('source_train_classes', {}).get('default', [0, 1, 2, 3, 4])
    cfg['run_id'] = f"gentra_{raw['experiment']['name']}_epoch{cfg['epochs']}"
    cfg['experiment_name'] = raw['experiment']['name']
    cfg['seed'] = int(raw['experiment'].get('seed', 1))
    if 'time_length' in raw.get('preprocess', {}):
        tl = raw['preprocess']['time_length']
        cfg['time_length'] = None if tl is None else float(tl)
    cfg['class_balanced_sampling'] = bool(cfg.get('class_balanced_sampling', True))
    cfg = apply_preprocess_cache_dirs(cfg)
    data_cfg = raw['data']
    cfg['pkl_cache_root'] = _project_data_path(os.environ.get('GENTRA_SOURCE_ROOT', data_cfg.get('pkl_root', cfg['pkl_cache_root'])))
    cfg['cache_dir'] = _project_data_path(os.environ.get('GENTRA_TRAIN_CACHE_DIR', data_cfg.get('train_cache_dir', cfg['cache_dir'])))
    cfg['zs_cache_dir'] = _project_data_path(os.environ.get('GENTRA_ZS_CACHE_DIR', data_cfg.get('eval_cache_dir', cfg['zs_cache_dir'])))
    cfg['zs_test_specs'] = build_zs_test_specs(
        bnci_data_dir(), _project_data_path(os.environ.get('GENTRA_SOURCE_ROOT', data_cfg['eval_pkl_root'])) if data_cfg.get('eval_pkl_root') else None
    )
    cfg['min_lr'] = float(cfg['lr']) * float(cfg.get('min_lr_ratio', 0.1))
    cfg['min_weight_decay'] = float(cfg.get('final_weight_decay', cfg['weight_decay']))
    cfg['warmup_epochs'] = int(cfg.get('lr_warmup_epochs', 0))
    return cfg


def save_config_snapshot(cfg: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config_snapshot.json'
    path.write_text(json.dumps(cfg, indent=2, default=str), encoding='utf-8')
    return path


def source_pkl_path_from_cfg(cfg: dict):
    root = cfg.get('pkl_cache_root')
    return lambda name: source_pkl_path(name, cache_root=root)


def source_preprocessed_cache_path(source_name: str, cfg: dict | None = None) -> str:
    from gentra.data.multi_source_mi import preprocess_cache_suffix

    if source_name not in SOURCE_CLASS_SPECS:
        raise KeyError(f'Unknown LOO source {source_name!r}')
    pre = train_pre_cfg_from_config(cfg) if cfg is not None else {
        'l_freq': 4.0,
        'h_freq': 30.0,
        'target_fs': 250,
        'time_length': 4.0,
        'apply_EA': True,
        'norm_method': 'none',
    }
    cache_root = Path(cfg['cache_dir']) if cfg is not None else train_cache_dir()
    path = cache_root / f'{source_name}_{preprocess_cache_suffix(pre)}.pkl'
    return str(path)

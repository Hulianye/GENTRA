"""Portable data and output locations for the GENTRA release."""
from __future__ import annotations
import os
from pathlib import Path

def project_root() -> Path:
    if os.environ.get('GENTRA_PROJECT_ROOT'):
        return Path(os.environ['GENTRA_PROJECT_ROOT']).expanduser().resolve()
    cwd = Path.cwd()
    if (cwd / 'configs' / 'pretrain.yaml').is_file():
        return cwd
    return Path(__file__).resolve().parents[3]

def pkl_root() -> Path:
    return Path(os.environ.get('GENTRA_PKL_ROOT', os.environ.get('GENTRA_SOURCE_ROOT', project_root() / 'data/source')))

def train_cache_dir() -> Path:
    return Path(os.environ.get('GENTRA_TRAIN_CACHE_DIR', project_root() / 'data/cache/train'))

def zs_cache_dir() -> Path:
    return Path(os.environ.get('GENTRA_ZS_CACHE_DIR', project_root() / 'data/cache/eval'))

def downstream_cache_dir() -> Path:
    return Path(os.environ.get('GENTRA_DOWNSTREAM_CACHE_DIR', project_root() / 'data/cache/downstream'))

def bnci_data_dir() -> Path:
    return Path(os.environ.get('GENTRA_BNCI_DIR', pkl_root()))

def output_root() -> Path:
    return Path(os.environ.get('GENTRA_OUTPUT_ROOT', project_root() / 'outputs'))

def downstream_raw_dir() -> Path:
    return Path(os.environ.get('GENTRA_DOWNSTREAM_RAW_ROOT', project_root() / 'data/downstream_raw'))

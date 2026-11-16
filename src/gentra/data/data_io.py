"""Locate a user-provided EEG dataset bundle."""
from __future__ import annotations
import os
from pathlib import Path
from gentra.config.paths import pkl_root

def resolve_data_path(custom_path=None, dataset_name='BNCI2014001'):
    if custom_path:
        return str(Path(custom_path).expanduser().resolve())
    return str(Path(os.environ.get('GENTRA_DATA_ROOT', pkl_root())) / f'{dataset_name}.pkl')

"""Dataset registry and zero-shot evaluation specifications."""

from __future__ import annotations

import os
from pathlib import Path

from gentra.data.source_channels import channel_names_for_source


SOURCE_CLASS_SPECS = {
    'Brandl2020': {'allowed_classes': (0, 1), 'n_cls': 2},
    'Lee2019': {'allowed_classes': (0, 1), 'n_cls': 2},
    'Cho2017': {'allowed_classes': (0, 1), 'n_cls': 2},
    'Zhou2020': {'allowed_classes': (0, 1, 2), 'n_cls': 3},
    'Zhou2016': {'allowed_classes': (0, 1, 2), 'n_cls': 3},
    'PhysionetMotorImagery': {'allowed_classes': (0, 1, 2, 3, 4), 'n_cls': 5},
    'Weibo2014': {'allowed_classes': (0, 1, 2, 3, 4), 'n_cls': 5},
    'Liu2024': {'allowed_classes': (0, 1), 'n_cls': 2},
}

def source_pkl_path(source_name: str, cache_root: str | Path | None = None) -> str:
    root = Path(cache_root) if cache_root else Path('.')
    return str(root / f'{source_name}.pkl')


def test_dataset_paths(bnci_dir: Path, pkl_root_override: str | Path | None = None) -> dict[str, str]:
    if pkl_root_override is not None:
        root = Path(pkl_root_override)
        return {name: str(root / f'{name}.pkl') for name in (
            'BNCI2014004', 'BNCI2014001', 'BNCI2015001', 'Zhou2016'
        )}
    return {
        'BNCI2014004': str(bnci_dir / 'BNCI2014004.pkl'),
        'BNCI2014001': str(bnci_dir / 'BNCI2014001.pkl'),
        'BNCI2015001': str(bnci_dir / 'BNCI2015001.pkl'),
        'Zhou2016': str(Path(os.environ.get('GENTRA_PKL_ROOT', bnci_dir)) / 'Zhou2016.pkl'),
    }


def build_zs_test_specs(
    bnci_dir: Path, pkl_root_override: str | Path | None = None
) -> list[tuple]:
    paths = test_dataset_paths(bnci_dir, pkl_root_override)
    return [
        ('BNCI2014004', paths['BNCI2014004'], channel_names_for_source('BNCI2014004'), True, None),
        ('BNCI2014001', paths['BNCI2014001'], channel_names_for_source('BNCI2014001'), False, (0, 1, 2)),
        ('BNCI2015001', paths['BNCI2015001'], channel_names_for_source('BNCI2015001'), False, None),
    ]

#!/usr/bin/env python3
"""Export the pre-fine-tuning EEGData bundles used by the released scenarios.

This deliberately preserves each dataset's original provider/MOABB processing
stage. The fine-tuning runner applies its own scenario-specific transforms.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gentra.data.eeg_data import EEGData


@dataclass(frozen=True)
class DownstreamSpec:
    moabb_class: str
    fmin: float
    fmax: float
    sessions: tuple[str, ...] | None
    labels: dict[str, int]
    channels: tuple[str, ...]
    shape: tuple[int, int, int]
    subjects: int
    sampling_rate: int
    dtype: str
    crop_samples: int | None = None
    enumerate_subjects: bool = False


SPECS = {
    "BNCI2014001": DownstreamSpec(
        "BNCI2014_001", 0.1, 75.0, ("0train",),
        {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3},
        ("Fz", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"),
        (2592, 22, 1000), 9, 250, "float64", crop_samples=1000,
    ),
    "BNCI2014004": DownstreamSpec(
        "BNCI2014_004", 0.0, 120.0, ("3test",),
        {"left_hand": 0, "right_hand": 1},
        ("C3", "CZ", "C4"),
        (1400, 3, 1126), 9, 250, "float64",
    ),
    "BNCI2015001": DownstreamSpec(
        "BNCI2015_001", 0.1, 75.0, ("0train", "0A"),
        {"right_hand": 0, "feet": 1},
        ("FC3", "FCZ", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CPZ", "CP4"),
        (2400, 13, 2561), 12, 512, "float64",
    ),
    "Zhou2016": DownstreamSpec(
        "Zhou2016", 8.0, 32.0, None,
        {"left_hand": 0, "right_hand": 1, "feet": 2},
        ("FP1", "FP2", "FC3", "FCZ", "FC4", "C3", "CZ", "C4", "CP3", "CPZ", "CP4", "O1", "OZ", "O2"),
        (1800, 14, 1251), 4, 250, "float32", enumerate_subjects=True,
    ),
}


def configure_raw_root(raw_root: Path) -> None:
    """Point MOABB/MNE at user-owned raw files before importing datasets."""
    for key in (
        "MNE_DATA", "MNE_DATASETS_BNCI_PATH",
        "MNE_DATASETS_BNCI2014-001_PATH", "MNE_DATASETS_BNCI2014-004_PATH",
        "MNE_DATASETS_BNCI2015-001_PATH", "MNE_DATASETS_ZHOU2016_PATH",
    ):
        os.environ[key] = str(raw_root)


def validate_bundle(bundle: EEGData, spec: DownstreamSpec) -> dict:
    x = np.asarray(bundle.eeg_data)
    y = np.asarray(bundle.labels)
    sid = np.asarray(bundle.subject_ids)
    if x.shape != spec.shape or x.dtype != np.dtype(spec.dtype):
        raise ValueError(f"{bundle.dataset_name}: unexpected signal shape/dtype {x.shape}/{x.dtype}")
    if y.shape != (len(x),) or sid.shape != (len(x),):
        raise ValueError(f"{bundle.dataset_name}: label or subject count mismatch")
    if not np.isfinite(x).all():
        raise ValueError(f"{bundle.dataset_name}: signal contains NaN or Inf")
    if len(np.unique(sid)) != spec.subjects:
        raise ValueError(f"{bundle.dataset_name}: unexpected number of subjects")
    if not np.array_equal(np.unique(y), np.arange(len(spec.labels))):
        raise ValueError(f"{bundle.dataset_name}: unexpected labels {np.unique(y)}")
    if tuple(str(ch).upper() for ch in bundle.channel_names) != tuple(ch.upper() for ch in spec.channels):
        raise ValueError(f"{bundle.dataset_name}: channel order mismatch")
    if bundle.sampling_rate != spec.sampling_rate or bundle.split_method != "LOSO":
        raise ValueError(f"{bundle.dataset_name}: sampling rate or split method mismatch")
    return {
        "shape": list(x.shape), "dtype": str(x.dtype),
        "subjects": spec.subjects, "sampling_rate": spec.sampling_rate,
        "labels": {str(int(v)): int(n) for v, n in zip(*np.unique(y, return_counts=True))},
    }


def export_dataset(name: str, spec: DownstreamSpec, output_dir: Path) -> dict:
    from moabb import datasets
    from moabb.datasets.base import CacheConfig
    from moabb.paradigms import MotorImagery

    dataset = getattr(datasets, spec.moabb_class)()
    paradigm = MotorImagery(fmin=spec.fmin, fmax=spec.fmax)
    xs, ys, sids = [], [], []
    for index, subject in enumerate(dataset.subject_list):
        x, labels, metadata = paradigm.get_data(
            dataset, subjects=[subject], cache_config=CacheConfig(use=False)
        )
        mask = np.isin(labels, tuple(spec.labels))
        if spec.sessions is not None:
            mask &= np.isin(metadata["session"].astype(str).to_numpy(), spec.sessions)
        if not mask.any():
            raise ValueError(f"{name}, subject {subject}: no selected trials")
        x = np.asarray(x[mask], dtype=spec.dtype)
        if spec.crop_samples is not None:
            if x.shape[-1] < spec.crop_samples:
                raise ValueError(f"{name}, subject {subject}: trial shorter than crop")
            x = x[..., :spec.crop_samples]
        xs.append(x)
        ys.append(np.asarray([spec.labels[str(v)] for v in np.asarray(labels)[mask]], dtype=np.int64))
        output_subject = index if spec.enumerate_subjects else subject
        sids.append(np.full(len(x), output_subject, dtype=np.int64))
        print(f"{name}: subject {subject} -> {len(x)} trials", flush=True)

    bundle = EEGData(
        dataset_name=name,
        eeg_data=np.concatenate(xs),
        labels=np.concatenate(ys),
        subject_ids=np.concatenate(sids),
        channel_names=list(spec.channels),
        sampling_rate=spec.sampling_rate,
        split_method="LOSO",
    )
    summary = validate_bundle(bundle, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    temp = path.with_suffix(".pkl.tmp")
    with temp.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)
    summary["path"] = str(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--raw-root", type=Path, default=Path(os.environ.get("GENTRA_RAW_ROOT", ROOT / "data/raw")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("GENTRA_DOWNSTREAM_RAW_ROOT", ROOT / "data/downstream_raw")))
    args = parser.parse_args()
    configure_raw_root(args.raw_root)
    report = {name: export_dataset(name, SPECS[name], args.output_dir) for name in args.datasets}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

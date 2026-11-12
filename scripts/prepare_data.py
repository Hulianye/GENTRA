#!/usr/bin/env python3
"""Rebuild the versioned 8-32 Hz, 250 Hz, 4 s MI sources and EA caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SOURCE_ROOT = Path(os.environ.get('GENTRA_SOURCE_ROOT', PROJECT_ROOT / 'data/source'))
TRAIN_CACHE_ROOT = Path(os.environ.get('GENTRA_TRAIN_CACHE_DIR', PROJECT_ROOT / 'data/cache/train'))
EVAL_CACHE_ROOT = Path(os.environ.get('GENTRA_ZS_CACHE_DIR', PROJECT_ROOT / 'data/cache/eval'))
RAW_ROOT = Path(os.environ.get('GENTRA_RAW_ROOT', PROJECT_ROOT / 'data/raw'))
os.environ.setdefault("MNE_DATA", str(RAW_ROOT))
for _key in (
    "MNE_DATASETS_PHYSIONETMOTORIMAGERY_PATH", "MNE_DATASETS_EEGBCI_PATH",
    "MNE_DATASETS_LEE2019-MI_PATH", "MNE_DATASETS_WEIBO2014_PATH",
    "MNE_DATASETS_WEIBO_PATH", "MNE_DATASETS_CHO2017_PATH",
    "MNE_DATASETS_GIGADB_PATH", "MNE_DATASETS_BNCI_PATH",
    "MNE_DATASETS_BNCI2014-001_PATH", "MNE_DATASETS_BNCI2014-004_PATH",
    "MNE_DATASETS_BNCI2015-001_PATH", "MNE_DATASETS_ZHOU2016_PATH",
):
    os.environ.setdefault(_key, str(RAW_ROOT))
os.environ.setdefault("MNE_DATASETS_BRANDL2020_PATH", str(RAW_ROOT / "mne_data"))
os.environ.setdefault("MNE_DATASETS_ZHOU2020_PATH", str(RAW_ROOT / "mne_data"))
for _key in ("MNE_DATASETS_LIU2024_PATH",):
    os.environ.setdefault(_key, str(RAW_ROOT / "mne_data"))
FS = 250
N_TIMES = 1000
@dataclass(frozen=True)
class Spec:
    name: str
    moabb_class: str
    labels: dict[str, int]
    expected_samples: int
    expected_subjects: int
    expected_channels: int
    role: str
    sessions: tuple[str, ...] | None = None
    runs: tuple[str, ...] | None = None
    subjects: tuple[int, ...] | None = None
    enumerate_subjects: bool = False
    tmax: float | None = None
    cho_loop_fill: bool = False
    literature_window: str = ""
    source_state: str = "unprocessed raw recording"
    exclude_channels: tuple[str, ...] = ()


SPECS = {
    "Brandl2020": Spec("Brandl2020", "Brandl2020", {"left_hand": 0, "right_hand": 1}, 8058, 16, 63, "train"),
    "PhysionetMotorImagery": Spec(
        "PhysionetMotorImagery", "PhysionetMI",
        {"left_hand": 0, "right_hand": 1, "feet": 2, "hands": 3, "rest": 4},
        38039, 109, 64, "train", enumerate_subjects=True, tmax=3.992,
    ),
    "Lee2019": Spec(
        "Lee2019", "Lee2019_MI", {"left_hand": 0, "right_hand": 1},
        10800, 54, 62, "train", sessions=("0", "1"), runs=("1train",),
    ),
    "Weibo2014": Spec(
        "Weibo2014", "Weibo2014",
        {"left_hand": 0, "right_hand": 1, "hands": 3, "feet": 2, "rest": 4},
        3960, 10, 60, "train",
    ),
    "Cho2017": Spec(
        "Cho2017", "Cho2017", {"left_hand": 0, "right_hand": 1},
        10520, 52, 64, "train", cho_loop_fill=True,
    ),
    "Zhou2020": Spec(
        "Zhou2020", "Zhou2020", {"left_hand": 0, "right_hand": 1, "feet": 2},
        10002, 8, 26, "train", subjects=tuple(range(13, 21)), enumerate_subjects=True,
    ),
    "Liu2024": Spec(
        "Liu2024", "Liu2024", {"left_hand": 0, "right_hand": 1},
        2000, 50, 29, "train", literature_window="exact 4 s MI window",
        source_state="publisher-preprocessed 0.5-40 Hz recording",
    ),
    "BNCI2014001": Spec(
        "BNCI2014001", "BNCI2014_001",
        {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3},
        2592, 9, 22, "eval", sessions=("0train",),
    ),
    "BNCI2014004": Spec(
        "BNCI2014004", "BNCI2014_004", {"left_hand": 0, "right_hand": 1},
        1400, 9, 3, "eval", sessions=("3test",),
    ),
    "BNCI2015001": Spec(
        "BNCI2015001", "BNCI2015_001", {"right_hand": 0, "feet": 1},
        2400, 12, 13, "eval", sessions=("0A",),
    ),
    "Zhou2016": Spec(
        "Zhou2016", "Zhou2016", {"left_hand": 0, "right_hand": 1, "feet": 2},
        1800, 4, 14, "eval",
    ),
}


def _versions() -> dict[str, str]:
    import mne
    import moabb
    return {"mne": mne.__version__, "moabb": moabb.__version__, "numpy": np.__version__}


def _dataset(spec: Spec):
    from moabb import datasets
    cls = getattr(datasets, spec.moabb_class)
    if spec.moabb_class == "PhysionetMI":
        return cls(imagined=True, executed=True)
    if spec.moabb_class == "Liu2024":
        return cls(break_events=False, instr_events=False)
    dataset = cls()
    if spec.moabb_class == "Lee2019_MI":
        # MOABB 1.5 emits zero-based keys but filters with one-based keys.
        dataset._selected_sessions = ("0", "1")
    return dataset


def _paradigm(spec: Spec):
    from moabb.paradigms import MotorImagery
    kwargs = {
        "events": list(spec.labels),
        "n_classes": len(spec.labels),
        "fmin": 8.0,
        "fmax": 32.0,
        "resample": FS,
    }
    if spec.tmax is not None:
        kwargs["tmin"] = 0.0
        kwargs["tmax"] = spec.tmax
    return MotorImagery(**kwargs)


def _fit_time_axis(x: np.ndarray, spec: Spec) -> tuple[np.ndarray, str]:
    n = x.shape[-1]
    if n >= N_TIMES:
        return x[..., :N_TIMES], "continuous_first_1000_samples"
    if not spec.cho_loop_fill:
        raise ValueError(f"{spec.name}: only {n} samples after resampling")
    need = N_TIMES - n
    if n != 750 or need != 250:
        raise ValueError(f"Cho2017: expected 750 samples before fill, got {n}")
    return np.concatenate((x, x[..., :need]), axis=-1), "append_first_250_samples"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _validate_source(bundle: dict, spec: Spec) -> dict:
    x = np.asarray(bundle["eeg_data"])
    y = np.asarray(bundle["labels"])
    sid = np.asarray(bundle["subject_ids"])
    errors = []
    if x.shape != (spec.expected_samples, spec.expected_channels, N_TIMES):
        errors.append(f"shape={x.shape}")
    if len(y) != len(x) or len(sid) != len(x):
        errors.append("sample/label/subject lengths differ")
    if len(np.unique(sid)) != spec.expected_subjects:
        errors.append(f"subjects={len(np.unique(sid))}")
    if bundle["sampling_rate"] != FS:
        errors.append(f"sampling_rate={bundle['sampling_rate']}")
    if len(bundle["channel_names"]) != spec.expected_channels:
        errors.append(f"channels={len(bundle['channel_names'])}")
    if not np.isfinite(x).all():
        errors.append("NaN or Inf present")
    if spec.cho_loop_fill:
        if not np.array_equal(x[..., 750:1000], x[..., :250]):
            errors.append("Cho tail does not equal first 250 samples")
    elif bundle["export_metadata"]["time_policy"] != "continuous_first_1000_samples":
        errors.append("non-Cho data are not marked continuous")
    if errors:
        raise ValueError(f"{spec.name} validation failed: {', '.join(errors)}")
    values, counts = np.unique(y, return_counts=True)
    return {
        "shape": list(x.shape),
        "subjects": int(len(np.unique(sid))),
        "labels": {str(int(k)): int(v) for k, v in zip(values, counts)},
        "finite": True,
    }


def export_source(spec: Spec) -> dict:
    from moabb.datasets.base import CacheConfig

    dataset = _dataset(spec)
    subjects = list(spec.subjects or dataset.subject_list)
    paradigm = _paradigm(spec)
    chunks_x, chunks_y, chunks_sid = [], [], []
    channel_names = None
    time_policy = None
    sessions_seen: set[str] = set()
    runs_seen: set[str] = set()
    for subject_index, subject in enumerate(subjects):
        epochs, labels, metadata = paradigm.get_data(
            dataset=dataset, subjects=[subject], return_epochs=True,
            cache_config=CacheConfig(use=False),
        )
        mask = np.isin(labels, list(spec.labels))
        if spec.sessions is not None:
            session_values = metadata["session"].astype(str).to_numpy()
            mask &= np.isin(session_values, spec.sessions)
        if spec.runs is not None:
            run_values = metadata["run"].astype(str).to_numpy()
            mask &= np.isin(run_values, spec.runs)
        if not mask.any():
            raise ValueError(f"{spec.name} subject {subject}: no selected trials")
        keep_channels = [
            index for index, name in enumerate(epochs.ch_names)
            if name.upper() not in spec.exclude_channels
        ]
        x = epochs.get_data(copy=False)[mask][:, keep_channels].astype(np.float32)
        selected_labels = np.asarray(labels)[mask]
        x, policy = _fit_time_axis(x, spec)
        if time_policy is None:
            time_policy = policy
        elif time_policy != policy:
            raise ValueError(f"{spec.name}: inconsistent time policy")
        current_channels = [str(epochs.ch_names[index]).upper() for index in keep_channels]
        if channel_names is None:
            channel_names = current_channels
        elif current_channels != channel_names:
            raise ValueError(f"{spec.name}: channel order changed for subject {subject}")
        chunks_x.append(x)
        chunks_y.append(np.asarray([spec.labels[v] for v in selected_labels], dtype=np.int64))
        output_subject = subject_index if spec.enumerate_subjects else subject
        chunks_sid.append(np.full(len(x), output_subject, dtype=np.int64))
        sessions_seen.update(metadata.loc[mask, "session"].astype(str).tolist())
        runs_seen.update(metadata.loc[mask, "run"].astype(str).tolist())
        print(f"{spec.name}: subject {subject} -> {len(x)} trials", flush=True)
    bundle = {
        "dataset_name": spec.name,
        "eeg_data": np.concatenate(chunks_x).astype(np.float32, copy=False),
        "labels": np.concatenate(chunks_y),
        "subject_ids": np.concatenate(chunks_sid),
        "channel_names": channel_names,
        "sampling_rate": FS,
        "export_metadata": {
            "version": "mi_8_32_fs250_tl4_v1",
            "moabb_class": spec.moabb_class,
            "moabb_dataset_code": dataset.code,
            "moabb_subjects": subjects,
            "output_subject_id_policy": "zero_based_enumeration" if spec.enumerate_subjects else "moabb_subject_id",
            "selected_sessions": list(spec.sessions) if spec.sessions else "all",
            "selected_runs": list(spec.runs) if spec.runs else "all",
            "sessions_seen": sorted(sessions_seen),
            "runs_seen": sorted(runs_seen),
            "event_label_mapping": spec.labels,
            "bandpass_hz": [8.0, 32.0],
            "target_sampling_rate_hz": FS,
            "target_samples": N_TIMES,
            "effective_duration_seconds": N_TIMES / FS,
            "moabb_tmax": spec.tmax,
            "time_policy": time_policy,
            "literature_window": spec.literature_window,
            "source_state": spec.source_state,
            "excluded_non_eeg_channels": list(spec.exclude_channels),
            "raw_data_root": str(RAW_ROOT),
            "software_versions": _versions(),
        },
    }
    validation = _validate_source(bundle, spec)
    path = SOURCE_ROOT / f"{spec.name}.pkl"
    _atomic_pickle(bundle, path)
    validation["path"] = str(path)
    validation["sha256"] = _sha256(path)
    return validation


def _ea(x: np.ndarray) -> np.ndarray:
    from gentra.data.preprocessing import Preprocessor
    preprocessor = Preprocessor(
        target_fs=None, l_freq=None, h_freq=None, normalize_method="none",
        time_length=None, apply_EA=True,
    )
    return preprocessor.EA(x).astype(np.float32)


def _cache_identity() -> dict:
    from gentra.data.multi_source_mi import preprocess_cache_identity
    return preprocess_cache_identity({
        "l_freq": None, "h_freq": None, "target_fs": None, "time_length": None,
        "apply_EA": True, "norm_method": "none",
    })


def _cache_filename(name: str, role: str, subset: str | None = None) -> Path:
    from gentra.data.multi_source_mi import preprocess_cache_suffix
    pre = {
        "l_freq": None, "h_freq": None, "target_fs": None, "time_length": None,
        "apply_EA": True, "norm_method": "none",
    }
    tag = preprocess_cache_suffix(pre)
    root = TRAIN_CACHE_ROOT if role == "train" else EVAL_CACHE_ROOT
    if role == "train":
        return root / f"{name}_{tag}.pkl"
    return root / f"{name}_{tag}_{subset or 'allcls'}.pkl"


def build_cache(spec: Spec, *, role_override: str | None = None, subset: str | None = None) -> dict:
    source_path = SOURCE_ROOT / f"{spec.name}.pkl"
    with source_path.open("rb") as f:
        source = pickle.load(f)
    x = np.asarray(source["eeg_data"], dtype=np.float32)
    y = np.asarray(source["labels"], dtype=np.int64)
    sid = np.asarray(source["subject_ids"])
    if subset == "2cls":
        keep = np.isin(y, (0, 1))
    elif subset == "cls0_1_2":
        keep = np.isin(y, (0, 1, 2))
    else:
        keep = np.ones(len(y), dtype=bool)
    x, y, sid = x[keep], y[keep], sid[keep]
    out = np.empty_like(x)
    for subject in np.unique(sid):
        mask = sid == subject
        out[mask] = _ea(x[mask])
    role = role_override or spec.role
    path = _cache_filename(spec.name, role, subset)
    cache = {
        "x": out,
        "y": y,
        "subject_ids": sid,
        "channel_names": source["channel_names"],
        "preprocess_config": _cache_identity(),
        "ea_config": {"enabled": True, "scope": "per_subject", "other_transforms": []},
        "source_file": str(source_path),
        "source_sha256": _sha256(source_path),
    }
    if not np.isfinite(out).all():
        raise ValueError(f"{spec.name}: cache contains NaN or Inf")
    if not np.array_equal(y, np.asarray(source["labels"])[keep]):
        raise ValueError(f"{spec.name}: cache labels changed")
    if not np.array_equal(sid, np.asarray(source["subject_ids"])[keep]):
        raise ValueError(f"{spec.name}: cache subject order changed")
    if out.shape[-1] != N_TIMES:
        raise ValueError(f"{spec.name}: cache time axis changed")
    _atomic_pickle(cache, path)
    return {"path": str(path), "shape": list(out.shape), "sha256": _sha256(path)}


def spectral_check() -> dict:
    from mne.filter import filter_data
    from scipy.signal import resample
    native_fs = 1000
    t = np.arange(native_fs * 8) / native_fs
    freqs = (6.0, 16.0, 40.0)
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    filtered = filter_data(signal, native_fs, 8.0, 32.0, verbose=False)
    filtered = resample(filtered, int(len(filtered) * FS / native_fs))
    ratios = {}
    for f in freqs:
        before = abs(np.vdot(signal, np.exp(-2j * np.pi * f * t)))
        t2 = np.arange(len(filtered)) / FS
        after = abs(np.vdot(filtered, np.exp(-2j * np.pi * f * t2)))
        ratios[str(int(f))] = float((after / len(filtered)) / (before / len(signal)))
    if ratios["6"] >= 0.25 or ratios["16"] <= 0.75 or ratios["40"] >= 0.25:
        raise ValueError(f"spectral check failed: {ratios}")
    return {"frequencies_hz": list(freqs), "amplitude_ratios_after_before": ratios}


def selected_specs(names: list[str] | None) -> list[Spec]:
    if not names:
        names = ['Brandl2020', 'PhysionetMotorImagery', 'Lee2019', 'Weibo2014', 'Cho2017', 'Zhou2020', 'Liu2024', 'BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'Zhou2016']
    unknown = sorted(set(names) - set(SPECS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return [SPECS[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("export", "cache", "all", "validate"), default="all")
    parser.add_argument("--datasets", nargs="*", choices=("Brandl2020", "PhysionetMotorImagery", "Lee2019", "Weibo2014", "Cho2017", "Zhou2020", "Liu2024", "BNCI2014001", "BNCI2014004", "BNCI2015001", "Zhou2016"))
    args = parser.parse_args()
    specs = selected_specs(args.datasets)
    report = {"version": "mi_8_32_fs250_tl4_v1", "versions": _versions()}
    if args.stage in {"export", "all"}:
        report["spectral_check"] = spectral_check()
        report["sources"] = {spec.name: export_source(spec) for spec in specs}
    if args.stage in {"cache", "all"}:
        caches = {}
        for spec in specs:
            if spec.name == "BNCI2014001":
                caches[f"{spec.name}_4class"] = build_cache(spec)
                caches[f"{spec.name}_3class"] = build_cache(spec, subset="cls0_1_2")
            elif spec.name == "BNCI2014004":
                caches[spec.name] = build_cache(spec, subset="2cls")
            elif spec.name == "Zhou2016":
                caches[f"{spec.name}_loo"] = build_cache(spec, role_override="train")
                caches[f"{spec.name}_eval"] = build_cache(spec)
            else:
                caches[spec.name] = build_cache(spec)
        report["caches"] = caches
    if args.stage == "validate":
        report["spectral_check"] = spectral_check()
        report["sources"] = {}
        for spec in specs:
            path = SOURCE_ROOT / f"{spec.name}.pkl"
            with path.open("rb") as f:
                bundle = pickle.load(f)
            item = _validate_source(bundle, spec)
            item.update(path=str(path), sha256=_sha256(path))
            report["sources"][spec.name] = item
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = SOURCE_ROOT / f"rebuild_report_{args.stage}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

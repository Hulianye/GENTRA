#!/usr/bin/env python3
"""Build and verify a GENTRA source archive."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_TOP = {"build", "dist", "data", "outputs", ".pytest_cache", ".pytest_tmp", ".git"}
REQUIRED = {
    "README.md", "LICENSE", "pyproject.toml", "configs/pretrain.yaml",
    "configs/finetune.yaml", "src/gentra/data/eeg_data.py",
    "src/gentra/models/loader.py", "scripts/pretrain.py",
    "scripts/finetune.py", "scripts/prepare_downstream_data.py", "scripts/evaluate_zero_shot.py",
    "scripts/evaluate_scpf.py", "checkpoints/gentra_pretrained.pt",
}


def included_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in EXCLUDE_TOP:
            continue
        if any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix == ".pyc":
            continue
        yield path, rel.as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (ROOT / "LICENSE").is_file():
        parser.error("Release archive requires LICENSE")
    files = list(included_files(ROOT))
    names = {name for _, name in files}
    missing = REQUIRED - names
    if missing:
        parser.error(f"Release files missing: {sorted(missing)}")
    checkpoints = [name for name in names if name.endswith(".pt")]
    if len(checkpoints) != 1:
        parser.error(f"Expected one checkpoint, got {checkpoints}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in files:
            archive.write(path, name)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != names:
            raise RuntimeError("Archive integrity check failed")
    print(f"Wrote {args.output} with {len(names)} files")


if __name__ == "__main__":
    main()

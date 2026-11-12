# GENTRA

Research code for multi-source motor imagery EEG pretraining, zero-shot evaluation, downstream full fine-tuning, and support-conditioned prototype fusion.

## Contents

- `src/gentra/`: model, preprocessing, training, evaluation and SCPF adaptation code.
- `configs/`: seven-source pretraining and Cross/Fewshot30 full fine-tuning configurations.
- `scripts/`: data preparation and command-line entry points.
- `checkpoints/gentra_pretrained.pt`: pretrained model checkpoint.
- `docs/`: data format and run instructions.
- `LICENSE`: MIT License for the released code and checkpoint.

## Install

Python 3.10+ and PyTorch 2.1+ are required. From the repository root:

```bash
pip install -e .
pip install -e '.[data]'  # only when preparing datasets with MOABB
```

## Quick smoke check

```bash
python examples/load_checkpoint.py
```

## Data

Obtain the EEG datasets from their original providers. Set `GENTRA_RAW_ROOT` to the raw-data directory if needed, then run:

```bash
python scripts/prepare_data.py --stage all
python scripts/prepare_downstream_data.py
```

Pretraining and zero-shot bundles are single-pass 8-32 Hz, 250 Hz, 4-second trials. The seven source datasets are Brandl2020, PhysionetMotorImagery, Lee2019, Weibo2014, Cho2017, Zhou2020, and Liu2024. The second command exports the four pre-fine-tuning EEGData bundles from MOABB datasets. See `docs/data.md` for the dataset-specific selection and expected bundle paths.

## Run

```bash
python scripts/pretrain.py --config configs/pretrain.yaml --seeds 3 --gpuid 0 --output_dir outputs/pretrain
python scripts/evaluate_zero_shot.py --config configs/pretrain.yaml --checkpoint checkpoints/gentra_pretrained.pt --gpuid 0 --output outputs/zero_shot.json
python scripts/finetune.py --config configs/finetune.yaml --checkpoint checkpoints/gentra_pretrained.pt --protocol cross --gpuid 0 --output_dir outputs/finetune
python scripts/evaluate_scpf.py --config configs/finetune.yaml --checkpoint checkpoints/gentra_pretrained.pt --scenario BNCI2014004_Cross_noCAR --output outputs/scpf.json
```

All commands run from the repository root. `--gpuid` falls back to CPU when CUDA is unavailable. Downstream fine-tuning reads the EEGData bundles exported to `data/downstream_raw/` (or `GENTRA_DOWNSTREAM_RAW_ROOT`) and builds its scenario-specific cache on first use.

The checkpoint contains the pretrained GENTRA model, channel vocabulary, and model configuration. SCPF uses `alpha=0.75` and five support trials per class. Cross scenarios draw support from other subjects; Fewshot30 scenarios draw support from each target subject’s 30% training pool and score the disjoint 70% query pool. The evaluation command implements this protocol on the scenario-specific preprocessing cache. The original EEG datasets are not redistributed.

The released code and checkpoint are licensed under the MIT License in `LICENSE`.

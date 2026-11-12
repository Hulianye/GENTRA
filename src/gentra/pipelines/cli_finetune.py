"""Run the configured GENTRA fine-tuning scenarios."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from gentra.config.paths import output_root, downstream_cache_dir, downstream_raw_dir


def _project_root() -> Path:
    from gentra.config.paths import project_root
    return project_root()


def load_finetune_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path or _project_root() / 'configs' / 'finetune.yaml')
    return yaml.safe_load(cfg_path.read_text(encoding='utf-8'))


def all_scenarios(cfg: dict) -> list[dict]:
    out = []
    for group in ('cross', 'fewshot'):
        for item in cfg.get(group, []):
            sc = dict(cfg.get('defaults', {}))
            sc.update(item)
            sc['protocol'] = group
            sc['task_mode'] = 'Cross' if group == 'cross' else 'Fewshot'
            out.append(sc)
    return out


def cache_path(scenario: dict) -> str:
    if scenario.get('preprocessed_cache'):
        return scenario['preprocessed_cache']
    if scenario.get('cache_filename'):
        return str(downstream_cache_dir() / scenario['cache_filename'])
    raise ValueError(f"Scenario {scenario.get('name', '<unnamed>')} needs cache_filename or preprocessed_cache")


def main():
    parser = argparse.ArgumentParser(description='GENTRA downstream finetune')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--protocol', choices=['cross', 'fewshot', 'all'], default='all')
    parser.add_argument('--scenario', type=str, default='')
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()

    cfg = load_finetune_config(args.config)
    scenarios = all_scenarios(cfg)
    if args.protocol != 'all':
        scenarios = [s for s in scenarios if s['protocol'] == args.protocol]
    if args.scenario:
        scenarios = [s for s in scenarios if s['name'] == args.scenario]

    out_root = Path(args.output_dir or output_root() / 'finetune')
    runner = Path(__file__).resolve().parent / 'finetune_runner.py'

    if not scenarios:
        raise ValueError('No matching scenarios')
    for scenario in scenarios:
        out_dir = out_root / scenario['protocol'] / scenario['name']
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(runner),
            '--pretrained_checkpoint', args.checkpoint,
            '--dataset', scenario['dataset'],
            '--task_mode', scenario['task_mode'],
            '--train_percentage', str(scenario['train_percentage']),
            '--batch_size', str(scenario['batch_size']),
            '--epochs', str(scenario['epochs']),
            '--dropout_rate', str(scenario['dropout_rate']),
            '--lr', str(scenario['lr']),
            '--weight_decay', str(scenario['weight_decay']),
            '--finetune_strategy', scenario.get('finetune_strategy', 'full'),
            '--gpuid', str(args.gpuid),
            '--preprocessed_cache', cache_path(scenario),
            '--output_dir', str(out_dir),
            '--seeds', str(scenario.get('seed', 0)),
            '--data_path', str(downstream_raw_dir() / f"{scenario['dataset']}.pkl"),
        ]
        if scenario.get('num_classes'):
            cmd.extend(['--num_classes', str(scenario['num_classes'])])
        if scenario.get('train_classes'):
            cmd.extend(['--train_classes', *[str(c) for c in scenario['train_classes']]])
        if scenario.get('data_path'):
            cmd[cmd.index('--data_path') + 1] = scenario['data_path']
        for key in ('l_freq', 'h_freq', 'norm_method', 'target_fs', 'time_length',
                    'optimizer_type', 'label_smoothing', 'min_lr', 'min_lr_ratio',
                    'final_weight_decay', 'lr_warmup_epochs', 'lr_decay_start_epoch',
                    'wd_decay_start_epoch'):
            if scenario.get(key) is not None:
                cmd.extend([f'--{key}', str(scenario[key])])
        for key in ('use_lr_scheduler', 'use_wd_scheduler', 'use_amp', 'cudnn_benchmark'):
            if key in scenario:
                cmd.append('--' + key if scenario[key] else '--no-' + key)
        print('Running', scenario['name'], flush=True)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

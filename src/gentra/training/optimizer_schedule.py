"""Delayed cosine LR / WD schedules for multi-source training."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _cosine_segment(n_steps: int, base: float, final: float) -> np.ndarray:
    if n_steps <= 0:
        return np.array([])
    if n_steps == 1:
        return np.array([final], dtype=float)
    return np.array([
        final + 0.5 * (base - final) * (1 + math.cos(math.pi * i / (n_steps - 1)))
        for i in range(n_steps)
    ], dtype=float)


def build_value_schedule(
    *,
    epochs: int,
    niter_per_ep: int,
    base_value: float,
    final_value: float,
    warmup_epochs: int = 0,
    decay_start_epoch: int = 1,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """Per-step schedule with optional warmup, flat hold, then cosine decay.

    decay_start_epoch is 1-indexed (epoch 1 = first epoch).
    """
    total_steps = int(epochs) * int(niter_per_ep)
    warmup_steps = max(0, int(warmup_epochs)) * int(niter_per_ep)
    decay_start_steps = max(warmup_steps, (max(1, int(decay_start_epoch)) - 1) * int(niter_per_ep))
    decay_steps = max(0, total_steps - decay_start_steps)

    parts = []
    if warmup_steps > 0:
        parts.append(np.linspace(start_warmup_value, base_value, warmup_steps, dtype=float))
    hold_steps = decay_start_steps - warmup_steps
    if hold_steps > 0:
        parts.append(np.full(hold_steps, base_value, dtype=float))
    parts.append(_cosine_segment(decay_steps, base_value, final_value))

    schedule = np.concatenate(parts) if parts else np.array([], dtype=float)
    if schedule.size < total_steps:
        pad = np.full(total_steps - schedule.size, final_value, dtype=float)
        schedule = np.concatenate((schedule, pad))
    elif schedule.size > total_steps:
        schedule = schedule[:total_steps]
    return schedule


def optimizer_lr_scales(optimizer, base_lr: float) -> List[float]:
    if base_lr <= 0:
        return [1.0 for _ in optimizer.param_groups]
    return [pg['lr'] / base_lr for pg in optimizer.param_groups]


def setup_optimizer_schedules(optimizer, args, train_loader, epochs: int) -> None:
    use_lr = bool(getattr(args, 'use_lr_scheduler', False))
    use_wd = bool(getattr(args, 'use_wd_scheduler', False))
    if not use_lr and not use_wd:
        args.lr_schedule_values = None
        args.wd_schedule_values = None
        args.optimizer_lr_scales = None
        args.optimizer_wd_mask = None
        return

    niter = len(train_loader)
    base_lr = float(args.lr)
    args.optimizer_lr_scales = optimizer_lr_scales(optimizer, base_lr)
    args.optimizer_wd_mask = [pg['weight_decay'] > 0 for pg in optimizer.param_groups]

    lr_warmup = int(getattr(args, 'lr_warmup_epochs', getattr(args, 'warmup_epochs', 0)))
    lr_decay_start = int(getattr(args, 'lr_decay_start_epoch', 1))
    min_lr_ratio = float(getattr(args, 'min_lr_ratio', 0.0))
    min_lr = getattr(args, 'min_lr', None)
    if min_lr is None:
        min_lr = base_lr * (min_lr_ratio if min_lr_ratio > 0 else 0.01)
    else:
        min_lr = float(min_lr)

    args.lr_schedule_values = None
    if use_lr:
        args.lr_schedule_values = build_value_schedule(
            epochs=int(epochs),
            niter_per_ep=niter,
            base_value=base_lr,
            final_value=min_lr,
            warmup_epochs=min(lr_warmup, int(epochs)),
            decay_start_epoch=lr_decay_start,
            start_warmup_value=0.0,
        )

    args.wd_schedule_values = None
    if use_wd:
        initial_wd = float(args.weight_decay)
        final_wd = float(getattr(args, 'final_weight_decay', getattr(args, 'min_weight_decay', initial_wd)))
        wd_decay_start = int(getattr(args, 'wd_decay_start_epoch', lr_decay_start))
        args.wd_schedule_values = build_value_schedule(
            epochs=int(epochs),
            niter_per_ep=niter,
            base_value=initial_wd,
            final_value=final_wd,
            warmup_epochs=0,
            decay_start_epoch=wd_decay_start,
            start_warmup_value=initial_wd,
        )


def apply_optimizer_schedule(optimizer, args) -> None:
    lr_schedule = getattr(args, 'lr_schedule_values', None)
    wd_schedule = getattr(args, 'wd_schedule_values', None)
    if lr_schedule is None and wd_schedule is None:
        return

    step = int(getattr(args, 'global_step', 0))
    scales = getattr(args, 'optimizer_lr_scales', None) or [1.0] * len(optimizer.param_groups)
    wd_mask = getattr(args, 'optimizer_wd_mask', None) or [pg['weight_decay'] > 0 for pg in optimizer.param_groups]

    if lr_schedule is not None:
        idx = min(step, len(lr_schedule) - 1)
        base_lr = float(lr_schedule[idx])
        for pg, scale in zip(optimizer.param_groups, scales):
            pg['lr'] = base_lr * scale

    if wd_schedule is not None:
        idx = min(step, len(wd_schedule) - 1)
        wd = float(wd_schedule[idx])
        for pg, apply_wd in zip(optimizer.param_groups, wd_mask):
            if apply_wd:
                pg['weight_decay'] = wd

    args.global_step = step + 1


def current_schedule_values(optimizer) -> Tuple[float, float, float]:
    lrs = [float(pg['lr']) for pg in optimizer.param_groups]
    wds = [float(pg['weight_decay']) for pg in optimizer.param_groups if pg['weight_decay'] > 0]
    return min(lrs), max(lrs), (wds[0] if wds else 0.0)

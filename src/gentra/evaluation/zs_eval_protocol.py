"""Shared zero-shot evaluation protocol helpers (train/eval consistency)."""

from __future__ import annotations


def resolve_zs_eval_class_indices(
    test_name: str,
    entry: tuple,
    model,
    cfg: dict,
) -> tuple[int, ...] | None:
    """Resolve the class indices used by the released zero-shot protocol."""
    if len(entry) >= 5 and entry[4] is not None:
        return tuple(entry[4])
    eval_idx_map = cfg.get('zs_eval_class_indices') or {}
    eval_class_indices = eval_idx_map.get(test_name)
    filt = entry[3] if len(entry) > 3 else False
    if eval_class_indices is None and filt and model.nb_classes > 2:
        eval_class_indices = (0, 1)
    return eval_class_indices

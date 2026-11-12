"""Support-conditioned prototype fusion used for frozen GENTRA adaptation."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def scpf_logits(
    query_embeddings: np.ndarray,
    support_embeddings: np.ndarray,
    support_indices: Mapping[int, np.ndarray],
    checkpoint_prototypes: Mapping[int, np.ndarray],
    class_ids: tuple[int, ...],
    temperature: float,
    *,
    alpha: float = 0.75,
    novel_class_ids: tuple[int, ...] = (),
) -> np.ndarray:
    """Return class logits from five-shot support and checkpoint prototypes.

    Embeddings are the normalized output of the checkpoint prototype head.
    Support indices refer to rows of support_embeddings. For classes absent from
    pretraining, pass their IDs in novel_class_ids to use support prototypes only.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must lie in [0, 1]')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    query = np.asarray(query_embeddings, dtype=np.float32)
    support = np.asarray(support_embeddings, dtype=np.float32)
    if query.ndim != 2 or support.ndim != 2 or query.shape[1] != support.shape[1]:
        raise ValueError('query and support embeddings must be 2D with the same feature size')
    columns = []
    for class_id in class_ids:
        indices = np.asarray(support_indices[class_id], dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f'No support examples for class {class_id}')
        support_mean = _unit(support[indices].mean(axis=0))
        if class_id in novel_class_ids:
            fused = support_mean
        else:
            checkpoint_mean = _unit(np.asarray(checkpoint_prototypes[class_id], dtype=np.float32).mean(axis=0))
            fused = _unit((1.0 - alpha) * checkpoint_mean + alpha * support_mean)
        columns.append((query @ fused) / temperature)
    return np.stack(columns, axis=1)

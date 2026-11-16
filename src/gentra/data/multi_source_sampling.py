"""Train/val splitting for multi-source MI pretraining."""

from __future__ import annotations

import numpy as np


def stratified_train_val_indices(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0:
        all_idx = np.arange(len(labels))
        return all_idx, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        val_size = max(1, int(round(len(label_indices) * val_ratio)))
        val_parts.append(label_indices[:val_size])
        train_parts.append(label_indices[val_size:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_train_val_indices_by_source(
    labels: np.ndarray,
    source_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0:
        all_idx = np.arange(len(labels))
        return all_idx, np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for source in np.unique(source_ids):
        src_mask = source_ids == source
        src_idx = np.where(src_mask)[0]
        src_labels = labels[src_idx]
        train_indices: list[np.ndarray] = []
        val_indices: list[np.ndarray] = []
        for label in np.unique(src_labels):
            label_indices = src_idx[src_labels == label]
            rng.shuffle(label_indices)
            val_size = max(1, int(round(len(label_indices) * val_ratio)))
            val_indices.append(label_indices[:val_size])
            train_indices.append(label_indices[val_size:])
        train_parts.append(np.concatenate(train_indices))
        val_parts.append(np.concatenate(val_indices))
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class ClassBalancedBatchSampler:
    """Each batch draws equally from every class (batch_size % n_classes == 0)."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        class_ids: tuple[int, ...] | list[int] = (0, 1, 2),
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.class_ids = tuple(int(c) for c in class_ids)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.n_classes = len(self.class_ids)
        if self.batch_size % self.n_classes != 0:
            raise ValueError(
                f'batch_size={batch_size} must be divisible by n_classes={self.n_classes}'
            )
        self.per_class = self.batch_size // self.n_classes
        self._by_class = {}
        for c in self.class_ids:
            idx = np.where(self.labels == c)[0]
            if idx.size == 0:
                raise ValueError(f'No training samples for class {c}')
            self._by_class[c] = idx

    def __len__(self) -> int:
        counts = [len(v) for v in self._by_class.values()]
        n_batches = max(c // self.per_class for c in counts)
        return n_batches if self.drop_last else max(1, n_batches)

    def __iter__(self):
        pools = {}
        for cid, idx in self._by_class.items():
            shuffled = idx.copy()
            self.rng.shuffle(shuffled)
            pools[cid] = shuffled
        n_batches = len(self)
        for _ in range(n_batches):
            batch_parts = []
            for cid in self.class_ids:
                if len(pools[cid]) < self.per_class:
                    extra = self._by_class[cid].copy()
                    self.rng.shuffle(extra)
                    pools[cid] = np.concatenate([pools[cid], extra])
                batch_parts.append(pools[cid][: self.per_class])
                pools[cid] = pools[cid][self.per_class :]
            batch_idx = np.concatenate(batch_parts)
            self.rng.shuffle(batch_idx)
            yield batch_idx.tolist()



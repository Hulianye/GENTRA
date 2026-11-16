"""Multi-source channel-aware EEG dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from gentra.training.channel_encoding import channels_to_ids


def build_global_index_map(bundles) -> list[tuple[int, int]]:
    mapping = []
    for b_id, bundle in enumerate(bundles):
        for t_id in range(len(bundle.y)):
            mapping.append((b_id, t_id))
    return mapping


def labels_from_bundles(bundles) -> np.ndarray:
    return np.concatenate([b.y for b in bundles], axis=0)


def source_ids_from_bundles(bundles) -> np.ndarray:
    parts = []
    for b_id, bundle in enumerate(bundles):
        parts.append(np.full(len(bundle.y), b_id, dtype=np.int64))
    return np.concatenate(parts, axis=0)


def subject_ids_from_bundles(bundles) -> np.ndarray:
    parts = []
    for bundle in bundles:
        parts.append(np.asarray(bundle.subject_ids, dtype=np.float64))
    return np.concatenate(parts, axis=0)


class MultiSourceChannelEEGDataset(Dataset):
    def __init__(
        self,
        bundles,
        vocab: dict[str, int],
        global_indices: np.ndarray | list[int],
        *,
        return_subject_id: bool = False,
    ):
        self.bundles = bundles
        self.vocab = vocab
        self.return_subject_id = return_subject_id
        full_map = build_global_index_map(bundles)
        idx_arr = np.asarray(global_indices, dtype=np.int64)
        self.global_indices = [full_map[int(i)] for i in idx_arr]
        self._flat_global_indices = idx_arr
        self._subject_ids = subject_ids_from_bundles(bundles)
        self._channel_ids_cache = {
            b_id: channels_to_ids(bundles[b_id].channel_names, vocab, strict=True)
            for b_id in range(len(bundles))
        }

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, idx: int):
        b_id, t_id = self.global_indices[idx]
        bundle = self.bundles[b_id]
        x = torch.from_numpy(bundle.x[t_id]).float()
        y = torch.tensor(bundle.y[t_id], dtype=torch.long)
        cids = torch.from_numpy(self._channel_ids_cache[b_id]).long()
        domain_id = torch.tensor(bundle.source_id, dtype=torch.long)
        if self.return_subject_id:
            gi = int(self._flat_global_indices[idx])
            subject_id = torch.tensor(float(self._subject_ids[gi]), dtype=torch.float32)
            return x, y, cids, domain_id, subject_id
        return x, y, cids, domain_id


def multi_source_collate_fn(batch):
    xs, ys, channel_ids_list, domain_ids = zip(*batch)
    time_lengths = {int(x.shape[-1]) for x in xs}
    if len(time_lengths) != 1:
        raise ValueError(
            f'Mixed time lengths in one batch ({sorted(time_lengths)}); '
            'preprocessing must produce a fixed number of samples.'
        )
    batch_size = len(batch)
    max_channels = max(x.shape[0] for x in xs)
    time_len = max(x.shape[-1] for x in xs)
    x_pad = torch.zeros(batch_size, max_channels, time_len, dtype=xs[0].dtype)
    channel_ids_pad = torch.zeros(batch_size, max_channels, dtype=torch.long)
    channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)
    y = torch.stack(ys)
    domain_ids_t = torch.stack(domain_ids)
    for i, (x, cids) in enumerate(zip(xs, channel_ids_list)):
        n_channels = x.shape[0]
        n_times = x.shape[-1]
        x_pad[i, :n_channels, :n_times] = x
        channel_ids_pad[i, :n_channels] = cids
        channel_mask[i, :n_channels] = True
    return x_pad, y, channel_ids_pad, channel_mask, domain_ids_t

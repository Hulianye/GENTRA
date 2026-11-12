from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

UNK_CHANNEL_ID = 0


def normalize_channel_name(name: str) -> str:
    return str(name).strip().strip('.').upper()


def build_channel_vocab(channel_name_lists: list[list[str]]) -> dict[str, int]:
    names: set[str] = set()
    for channel_names in channel_name_lists:
        for ch in channel_names:
            names.add(normalize_channel_name(ch))
    return {name: idx + 1 for idx, name in enumerate(sorted(names))}


def channels_to_ids(
    channels: list[str],
    vocab: dict[str, int],
    *,
    strict: bool = False,
) -> np.ndarray:
    """Map physical channel names to IDs; ID 0 is reserved for padding."""
    normalized = [normalize_channel_name(ch) for ch in channels]
    unknown = sorted({ch for ch in normalized if ch not in vocab})
    if strict and unknown:
        raise ValueError(
            'Unknown physical EEG channel(s); refusing to map them to padding '
            f'ID 0: {unknown}'
        )
    return np.array(
        [vocab.get(ch, UNK_CHANNEL_ID) for ch in normalized],
        dtype=np.int64,
    )


def vocab_size_from_vocab(vocab: dict[str, int]) -> int:
    if not vocab:
        return 1
    return max(vocab.values()) + 1


class ChannelAwareEEGDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, channel_ids: np.ndarray, device=None):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.channel_ids = torch.from_numpy(channel_ids).long()
        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)
            self.channel_ids = self.channel_ids.to(device)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.channel_ids


def channel_collate_fn(batch):
    xs, ys, channel_ids_list = zip(*batch)
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
    for i, (x, cids) in enumerate(zip(xs, channel_ids_list)):
        n_channels = x.shape[0]
        n_times = x.shape[-1]
        x_pad[i, :n_channels, :n_times] = x
        channel_ids_pad[i, :n_channels] = cids
        channel_mask[i, :n_channels] = True
    return x_pad, y, channel_ids_pad, channel_mask


def is_channel_encoded_batch(batch) -> bool:
    return isinstance(batch, (list, tuple)) and len(batch) in (4, 5)


def unpack_batch(batch, device, with_domain=False):
    if is_channel_encoded_batch(batch):
        if len(batch) == 5:
            inputs, labels, channel_ids, channel_mask, domain_ids = batch
            out = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
                channel_ids.to(device, non_blocking=True),
                channel_mask.to(device, non_blocking=True),
            )
            if with_domain:
                return out + (domain_ids.to(device, non_blocking=True),)
            return out
        inputs, labels, channel_ids, channel_mask = batch
        out = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
            channel_ids.to(device, non_blocking=True),
            channel_mask.to(device, non_blocking=True),
        )
        if with_domain:
            return out + (None,)
        return out
    inputs, labels = batch
    out = (
        inputs.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        None,
        None,
    )
    if with_domain:
        return out + (None,)
    return out

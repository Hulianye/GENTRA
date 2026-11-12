"""MOE spatial filter weight normalization."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def normalize_filter_weights(
    filter_weight: Tensor,
    filter_scale: Tensor,
) -> Tensor:
    """Return scaled weights while keeping vocabulary index 0 as padding."""
    # Index 0 is the padding/unknown sentinel, not a physical channel.  Do
    # not let it change the norm of the real channel weights.
    out = filter_weight.new_zeros(filter_weight.shape)
    if filter_weight.shape[1] > 1:
        out[:, 1:] = F.normalize(filter_weight[:, 1:], dim=1) * filter_scale.unsqueeze(1)
    return out


def gather_mix_weights(
    vocab_weights: Tensor,
    channel_ids: Tensor,
    channel_mask: Tensor,
) -> Tensor:
    """Gather vocab weights onto batch channels: [F, B, C]."""
    mix_w = vocab_weights[:, channel_ids]
    return mix_w * channel_mask.unsqueeze(0).to(dtype=mix_w.dtype)

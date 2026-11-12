"""Domain-aware auxiliary losses on encoder embeddings."""

from __future__ import annotations

import torch
from torch import Tensor


def _supcon_core(
    embeddings: Tensor,
    weights: Tensor,
    temperature: float,
    denominator_mask: Tensor | None = None,
) -> Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    sim = torch.matmul(embeddings, embeddings.t()) / temperature
    eye = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    if denominator_mask is None:
        denominator_mask = ~eye
    exp_logits = torch.exp(logits) * denominator_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    pos_w_sum = weights.sum(dim=1)
    valid = pos_w_sum > 0
    if not torch.any(valid):
        return embeddings.new_tensor(0.0)
    per = -(weights * log_prob).sum(dim=1) / pos_w_sum.clamp_min(1e-12)
    return per[valid].mean()


def class_negative_supcon_loss(
    embeddings: Tensor,
    labels: Tensor,
    domain_ids: Tensor,
    temperature: float,
) -> Tensor:
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    cross_domain_mask = domain_ids[:, None] != domain_ids[None, :]
    weights = (pos_mask & cross_domain_mask).float()
    negative_mask = labels[:, None] != labels[None, :]
    denominator_mask = weights.bool() | negative_mask
    return _supcon_core(embeddings, weights, temperature, denominator_mask)


def compute_aux_loss(
    aux_method: str,
    embeddings: Tensor,
    labels: Tensor,
    domain_ids: Tensor,
    *,
    supcon_temperature: float = 0.1,
) -> Tensor:
    if not aux_method:
        return embeddings.new_tensor(0.0)
    if aux_method == 'cross_domain_supcon_class_negatives':
        return class_negative_supcon_loss(
            embeddings,
            labels,
            domain_ids,
            supcon_temperature,
        )
    raise ValueError(f'Unknown aux_method: {aux_method!r}')

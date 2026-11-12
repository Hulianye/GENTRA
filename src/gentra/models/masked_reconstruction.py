"""Block-masked reconstruction helpers for GENTRA pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def recon_warmup_scale(epoch: int, warmup_epochs: int, target_lambda: float) -> float:
    if target_lambda <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return target_lambda
    if epoch <= 0:
        return 0.0
    progress = min(1.0, float(epoch) / float(warmup_epochs))
    return target_lambda * progress


def generate_block_token_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    block_tokens: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError('num_tokens must be positive')
    if block_tokens <= 0:
        raise ValueError('block_tokens must be positive')
    if mask_ratio <= 0.0:
        return torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    target_masked = max(1, int(round(num_tokens * mask_ratio)))
    max_start = max(0, num_tokens - block_tokens)
    for b in range(batch_size):
        masked_count = 0
        attempts = 0
        while masked_count < target_masked and attempts < num_tokens * 8:
            attempts += 1
            if max_start == 0:
                start = 0
            else:
                start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            block = torch.zeros(num_tokens, dtype=torch.bool, device=device)
            end = min(num_tokens, start + block_tokens)
            block[start:end] = True
            overlap = (mask[b] & block).any().item()
            if overlap:
                continue
            new_count = int(block.sum().item())
            if masked_count + new_count > num_tokens - 1:
                continue
            mask[b] |= block
            masked_count += new_count
        if not mask[b].any():
            start = int(torch.randint(0, max(1, max_start + 1), (1,), device=device).item())
            end = min(num_tokens, start + block_tokens)
            mask[b, start:end] = True
        if mask[b].all():
            keep = int(torch.randint(0, num_tokens, (1,), device=device).item())
            mask[b, keep] = False
    return mask


def masked_token_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    channel_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if pred.shape != target.shape:
        raise ValueError(f'pred shape {tuple(pred.shape)} != target {tuple(target.shape)}')
    if pred.ndim == 4:
        mask = token_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred)
    else:
        mask = token_mask.unsqueeze(-1).expand_as(pred)
    if channel_mask is not None and pred.ndim == 4:
        ch_mask = channel_mask.unsqueeze(1).unsqueeze(-1).expand_as(pred)
        mask = mask & ch_mask
    if not mask.any():
        zero = pred.new_tensor(0.0)
        return zero, zero, 0.0
    diff = pred - target.detach()
    mse = (diff[mask] ** 2).mean()
    mae = diff[mask].abs().mean()
    return mse, mae, float(mask.sum().item())


class ConvFeatureReconstructionHead(nn.Module):
    def __init__(self, emb_size: int, conv_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, conv_dim),
        )

    def forward(self, masked_states: torch.Tensor) -> torch.Tensor:
        return self.net(masked_states)


def compute_conv_feature_reconstruction_loss(
    head: ConvFeatureReconstructionHead,
    masked_states: torch.Tensor,
    conv_target: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    conv_target = F.layer_norm(conv_target, (conv_target.shape[-1],))
    pred = head(masked_states)
    return masked_token_mse(pred, conv_target, token_mask)


def build_reconstruction_head(
    recon_target: str,
    *,
    emb_size: int,
    conv_dim: int,
    channel_vocab_size: int,
    recon_hidden_dim: int,
    samples_per_token: int,
) -> nn.Module:
    if recon_target == 'conv_feature':
        return ConvFeatureReconstructionHead(emb_size, conv_dim, hidden_dim=recon_hidden_dim)
    raise ValueError(f'Unknown recon_target: {recon_target}')

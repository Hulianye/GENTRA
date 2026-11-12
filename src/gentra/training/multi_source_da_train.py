"""Domain-adaptation training step for multi-source MI pretrain."""

from __future__ import annotations

import torch

from gentra.training.channel_encoding import unpack_batch
from gentra.training.domain_aux_loss import compute_aux_loss
from gentra.training.prototype_loss import prototype_cross_entropy
from gentra.training.trainer import _is_prototype_model, _model_forward
from gentra.models.masked_reconstruction import recon_warmup_scale
from gentra.training.optimizer_schedule import apply_optimizer_schedule


def apply_random_channel_mask(inputs, channel_mask, max_ratio: float):
    """Randomly hide complete valid channels independently for each sample."""
    if max_ratio <= 0.0:
        return inputs, channel_mask

    masked_inputs = inputs.clone()
    masked_channel_mask = channel_mask.clone()
    valid_counts = channel_mask.sum(dim=1).long()
    max_mask_counts = torch.minimum(
        torch.floor(valid_counts * max_ratio).long(),
        (valid_counts - 1).clamp_min(0),
    )
    mask_counts = torch.floor(
        torch.rand(inputs.shape[0], device=inputs.device) * (max_mask_counts + 1)
    ).long()
    random_scores = torch.rand(channel_mask.shape, device=inputs.device)
    random_scores.masked_fill_(~channel_mask.bool(), 2.0)
    channel_order = random_scores.argsort(dim=1)
    selected_in_order = (
        torch.arange(channel_mask.shape[1], device=inputs.device).unsqueeze(0)
        < mask_counts.unsqueeze(1)
    )
    selected = torch.zeros_like(channel_mask, dtype=torch.bool)
    selected.scatter_(1, channel_order, selected_in_order)
    masked_inputs.masked_fill_(selected.unsqueeze(-1), 0)
    masked_channel_mask[selected] = False
    return masked_inputs, masked_channel_mask


def prototype_forward(model, inputs, channel_ids, channel_mask, patch_tokens=None):
    h = model.encode_features(inputs, channel_ids, channel_mask, patch_tokens=patch_tokens)
    head = model.prototype_head()
    embeddings = head.encode(h)
    return head.logits_from_embeddings(embeddings), embeddings


def _compute_reconstruction_loss(
    model,
    recon_head,
    inputs,
    channel_ids,
    channel_mask,
    patch_tokens,
    conv_target,
    args,
    device,
    *,
    token_mask=None,
    masked_states=None,
):
    from gentra.models.masked_reconstruction import (
        compute_conv_feature_reconstruction_loss,
        generate_block_token_mask,
    )

    recon_target = getattr(args, 'recon_target', '') or ''
    if recon_target != 'conv_feature' or recon_head is None or conv_target is None:
        return None

    if token_mask is None or masked_states is None:
        num_tokens = patch_tokens.shape[1]
        token_mask = generate_block_token_mask(
            patch_tokens.shape[0],
            num_tokens,
            float(getattr(args, 'mask_ratio', 0.0)),
            int(getattr(args, 'mask_block_tokens', 1)),
            device=device,
        )
        masked_tokens = model.apply_token_mask(patch_tokens, token_mask)
        h_masked = model.encode_from_patch_tokens(masked_tokens)
        masked_states = h_masked[:, 1:, :]

    mse, mae, n_elem = compute_conv_feature_reconstruction_loss(
        recon_head, masked_states, conv_target, token_mask,
    )
    return {
        'loss': mse,
        'mae': mae,
        'mask_ratio': float(token_mask.float().mean().item()),
        'masked_elements': n_elem,
    }


def train_one_epoch_da(
    model,
    train_loader,
    optimizer,
    device,
    args,
    scaler=None,
    epoch: int = 1,
):
    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
    if scaler is None and use_amp:
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    filter_l1_w = float(getattr(args, 'filter_sparsity_l1_weight', 0.0))
    filter_l1_target = getattr(args, 'filter_sparsity_l1_target', 'normalized_only')
    aux_method = getattr(args, 'aux_method', '') or ''
    lambda_aux = float(getattr(args, 'lambda_aux', 0.0))
    use_prototype_forward = _is_prototype_model(model)
    recon_target = getattr(args, 'recon_target', '') or ''
    lambda_recon = float(getattr(args, 'lambda_recon', 0.0))
    recon_warmup_epochs = int(getattr(args, 'recon_warmup_epochs', 0))
    recon_head = getattr(model, 'recon_head', None)
    use_reconstruction = bool(recon_target and lambda_recon > 0 and recon_head is not None)
    recon_scale = recon_warmup_scale(epoch, recon_warmup_epochs, lambda_recon) if use_reconstruction else 0.0
    need_embeddings = use_prototype_forward and bool(
        aux_method or use_reconstruction
    )
    aux_kw = dict(
        supcon_temperature=getattr(args, 'supcon_temperature', 0.1),
    )

    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_recon_mae = 0.0
    running_recon_mask = 0.0
    running_aux = 0.0
    n_samples = 0

    for batch in train_loader:
        apply_optimizer_schedule(optimizer, args)
        inputs, labels, channel_ids, channel_mask, domain_ids = unpack_batch(
            batch, device, with_domain=True,
        )
        inputs, channel_mask = apply_random_channel_mask(
            inputs,
            channel_mask,
            float(getattr(args, 'channel_mask_max_ratio', 0.0)),
        )
        optimizer.zero_grad(set_to_none=True)
        batch_n = inputs.size(0)
        with torch.amp.autocast('cuda', enabled=use_amp):
            patch_tokens = conv_target = None
            main_patch_tokens = None
            if use_reconstruction:
                patch_tokens, conv_target = model.encode_patch_tokens_and_targets(
                    inputs, channel_ids, channel_mask,
                )
                main_patch_tokens = patch_tokens
            if need_embeddings:
                outputs, embeddings = prototype_forward(
                    model, inputs, channel_ids, channel_mask,
                    patch_tokens=main_patch_tokens,
                )
            else:
                outputs = _model_forward(model, inputs, channel_ids, channel_mask)
                embeddings = None

            per_sample_loss = prototype_cross_entropy(
                model, outputs, labels, args,
            )
            loss = per_sample_loss.mean()

            if use_reconstruction and patch_tokens is not None:
                recon_stats = _compute_reconstruction_loss(
                    model, recon_head, inputs, channel_ids, channel_mask,
                    patch_tokens, conv_target, args, device,
                )
                if recon_stats is not None:
                    loss = loss + recon_scale * recon_stats['loss']
                    running_recon += recon_stats['loss'].item() * batch_n
                    running_recon_mae += recon_stats['mae'].item() * batch_n
                    running_recon_mask += recon_stats['mask_ratio'] * batch_n

            if embeddings is not None and domain_ids is not None:
                if aux_method and lambda_aux > 0:
                    aux_loss = compute_aux_loss(
                        aux_method, embeddings, labels, domain_ids, **aux_kw,
                    )
                    loss = loss + lambda_aux * aux_loss
                    running_aux += aux_loss.item() * batch_n

            if filter_l1_w > 0 and hasattr(model, 'filter_sparsity_l1_loss'):
                sparsity_loss = model.filter_sparsity_l1_loss(target=filter_l1_target)
                if sparsity_loss is not None:
                    loss = loss + filter_l1_w * sparsity_loss

        running_loss += loss.item() * batch_n
        n_samples += batch_n

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

    out = {
        'train_loss': running_loss / max(1, n_samples),
    }
    if use_reconstruction and n_samples > 0:
        out['train_recon_loss'] = running_recon / n_samples
        out['train_recon_mae'] = running_recon_mae / n_samples
        out['train_recon_mask_ratio'] = running_recon_mask / n_samples
        out['train_recon_lambda'] = recon_scale
    if aux_method and lambda_aux > 0 and n_samples > 0:
        out['train_aux_loss'] = running_aux / n_samples
        out['train_aux_weighted_loss'] = lambda_aux * running_aux / n_samples
    if hasattr(model, 'collect_gate_stats'):
        out['gate_stats'] = model.collect_gate_stats()
    return out

"""The public pretraining objective updates the encoder and reconstruction head."""
import math
from types import SimpleNamespace

import numpy as np
import torch

from gentra.data.eeg_data import EEGData
from gentra.models.loader import GENTRAModel
from gentra.models.masked_reconstruction import build_reconstruction_head
from gentra.training.channel_encoding import build_channel_vocab
from gentra.training.multi_source_da_train import train_one_epoch_da


def test_pretrain_all_active_losses():
    torch.manual_seed(0)
    channels = ['C3', 'CZ', 'C4']
    labels = torch.tensor([0, 1, 0, 1])
    data = EEGData('synthetic', np.zeros((4, 3, 128), dtype='float32'),
                   np.array([1, 1, 2, 2]), channels, 128, labels=labels.numpy())
    vocab = build_channel_vocab([channels])
    model = GENTRAModel(
        data, channel_vocab=vocab, emb_size=8, depth=1, conv_filters=8,
        conv_temporal_kernel=3, conv_temporal_kernel2=3, pool_kernel=8,
        pool_stride=4, num_heads=2, dropout_rate=0.0, activation='gelu',
        moe_shared_filters=2, moe_private_filters=2, moe_top_k=1,
        moe_gate_hidden_dim=4, num_prototypes=1, proj_dim=8,
    )
    model.recon_head = build_reconstruction_head(
        'conv_feature', emb_size=8, conv_dim=8, channel_vocab_size=len(vocab),
        recon_hidden_dim=16, samples_per_token=4,
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    ids = torch.tensor([[vocab[c] for c in channels]]).repeat(4, 1)
    batch = (torch.randn(4, 3, 128), labels, ids, torch.ones_like(ids, dtype=torch.bool),
             torch.tensor([0, 0, 1, 1]))
    args = SimpleNamespace(
        clip_grad_norm=1.0, aux_method='cross_domain_supcon_class_negatives',
        lambda_aux=0.1, recon_target='conv_feature', lambda_recon=0.1,
        recon_warmup_epochs=5, mask_ratio=0.3, mask_block_tokens=2,
        filter_sparsity_l1_weight=0.03, filter_sparsity_l1_target='normalized_only',
        channel_mask_max_ratio=0.5,
    )
    stats = train_one_epoch_da(model, [batch], torch.optim.AdamW(model.parameters(), lr=1e-3),
                              torch.device('cpu'), args)
    for key in ('train_loss', 'train_recon_loss', 'train_aux_loss'):
        assert math.isfinite(stats[key]) and stats[key] > 0
    assert 0 < stats['train_recon_mask_ratio'] < 1
    assert abs(stats['train_recon_lambda'] - 0.02) < 1e-8
    updated = [name for name, p in model.named_parameters() if not torch.equal(before[name], p)]
    assert any(name.startswith('recon_head.') for name in updated)
    assert any(not name.startswith('recon_head.') for name in updated)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

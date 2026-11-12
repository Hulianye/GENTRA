"""The released classification path supports a complete optimization step."""

import numpy as np
import torch

from gentra.data.eeg_data import EEGData
from gentra.models.loader import GENTRAModel
from gentra.pipelines.finetune_runner import build_training_args
from gentra.training.channel_encoding import build_channel_vocab
from gentra.training.dataset_split import create_dataloaders
from gentra.training.trainer import train_model


def test_full_finetune_one_epoch():
    torch.manual_seed(0)
    channels = ['C3', 'CZ', 'C4']
    data = EEGData(
        'synthetic', np.random.default_rng(0).standard_normal((8, 3, 128)).astype('float32'),
        np.repeat([1, 2], 4), channels, 128, labels=np.tile([0, 1], 4),
    )
    vocab = build_channel_vocab([channels])
    model = GENTRAModel(
        data, channel_vocab=vocab, emb_size=8, depth=1, conv_filters=8,
        conv_temporal_kernel=3, conv_temporal_kernel2=3, pool_kernel=8,
        pool_stride=4, num_heads=2, dropout_rate=0.0, activation='gelu',
        moe_shared_filters=2, moe_private_filters=2, moe_top_k=1,
        moe_gate_hidden_dim=4, num_prototypes=1, proj_dim=8,
    )
    train_mask = np.arange(8) < 4
    test_mask = ~train_mask
    train_loader, test_loader = create_dataloaders(
        data, train_mask, test_mask, batch_size=4, device=torch.device('cpu'),
        num_workers=0, use_channel_encoding=True, channel_vocab=vocab,
    )
    args = build_training_args({'epochs': 1, 'lr': 1e-3, 'proto_init': 'mean'}, {'quiet': True})
    trained, _, log, metrics = train_model(
        model, train_loader, test_loader, torch.device('cpu'), args, logger=None,
    )
    assert trained is model
    assert len(log['epochs']) == 1
    assert np.isfinite(metrics['balanced_accuracy'])

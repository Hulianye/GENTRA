import torch

from gentra.models.gentra import GENTRA, PrototypeClassificationHead
from gentra.training.channel_encoding import build_channel_vocab, unpack_batch, vocab_size_from_vocab
from gentra.training.model_loader_base import ModelLoader
from gentra.data.preprocessing import Preprocessor


class GENTRAModel(ModelLoader):
    def __init__(self, eeg_dataset, finetune_strategy='full', channel_vocab=None, **kwargs):
        super().__init__(eeg_dataset, finetune_strategy)
        if channel_vocab is None:
            channel_vocab = build_channel_vocab([list(eeg_dataset.channel_names)])
        self.channel_vocab = channel_vocab
        channel_vocab_size = kwargs.get('channel_vocab_size', 0) or vocab_size_from_vocab(channel_vocab)
        emb_size = kwargs.get('emb_size', 40)
        proj_dim = kwargs.get('proj_dim', 0) or emb_size
        model_kwargs = {
            'emb_size': emb_size,
            'depth': kwargs.get('depth', 6),
            'conv_filters': kwargs.get('conv_filters', 40),
            'conv_temporal_kernel': kwargs.get('conv_temporal_kernel', 25),
            'conv_temporal_kernel2': kwargs.get('conv_temporal_kernel2', 15),
            'pool_kernel': kwargs.get('pool_kernel', 75),
            'pool_stride': kwargs.get('pool_stride', 15),
            'num_heads': kwargs.get('num_heads', None),
            'dropout_rate': kwargs.get('dropout_rate', 0.5),
            'shallow_dropout': kwargs.get('shallow_dropout', None),
            'transformer_dropout': kwargs.get('transformer_dropout', None),
            'head_type': kwargs.get('head_type', 'cls_prototype'),
            'activation': kwargs.get('activation', 'elu'),
            'proj_dim': proj_dim,
            'temperature': kwargs.get('temperature', 0.1),
            'num_prototypes': kwargs.get('num_prototypes', 1),
            'prototype_logit_aggregation': kwargs.get(
                'prototype_logit_aggregation', 'logmeanexp'
            ),
            'channel_vocab_size': channel_vocab_size,
            'moe_shared_filters': kwargs.get('moe_shared_filters', 2),
            'moe_private_filters': kwargs.get('moe_private_filters', 16),
            'moe_top_k': kwargs.get('moe_top_k', 4),
            'moe_gate_hidden_dim': kwargs.get('moe_gate_hidden_dim', 64),
            'moe_gate_temperature': kwargs.get('moe_gate_temperature', 1.0),
            'moe_apply_topk_weight': kwargs.get('moe_apply_topk_weight', True),
            'moe_gate_feature_mode': kwargs.get('moe_gate_feature_mode', 'layout'),
            'moe_mixer_mode': kwargs.get('moe_mixer_mode', 'stack_gated_topk'),
            'moe_stack_kernel': kwargs.get('moe_stack_kernel', 1),
            'moe_stack_merge_mode': kwargs.get('moe_stack_merge_mode', 'full_1x1'),
            'shallow_norm_type': kwargs.get('shallow_norm_type', 'batch'),
            'use_rope': kwargs.get('use_rope', True),
            'rope_mode': kwargs.get('rope_mode', 'attention_qk'),
            'rope_max_seq_len': kwargs.get('rope_max_seq_len', 256),
            'attention_scale': kwargs.get('attention_scale', 'emb_size'),
            'n_classes': self.nb_classes,
        }
        self.main_model = GENTRA(**model_kwargs)
        self.apply_finetune_strategy(finetune_strategy)

    def apply_finetune_strategy(self, strategy=None):
        return super().apply_finetune_strategy(strategy)

    def is_prototype_head(self):
        return isinstance(self.main_model.head, PrototypeClassificationHead)

    def prototype_head(self):
        head = self.main_model.head
        if not isinstance(head, PrototypeClassificationHead):
            raise TypeError('Model does not use PrototypeClassificationHead')
        return head


    def encode_patch_tokens(self, x, channel_ids=None, channel_mask=None):
        return self.main_model.encode_patch_tokens(x.unsqueeze(1), channel_ids, channel_mask)

    def encode_patch_tokens_and_targets(self, x, channel_ids=None, channel_mask=None):
        return self.main_model.encode_patch_tokens_and_targets(
            x.unsqueeze(1), channel_ids, channel_mask,
        )

    def encode_from_patch_tokens(self, patch_tokens):
        return self.main_model.encode_from_patch_tokens(patch_tokens)

    def apply_token_mask(self, patch_tokens, token_mask):
        return self.main_model.apply_token_mask(patch_tokens, token_mask)

    def encode_features(self, x, channel_ids=None, channel_mask=None, patch_tokens=None):
        return self.main_model.encode_features(
            x.unsqueeze(1), channel_ids, channel_mask, patch_tokens=patch_tokens,
        )

    def prototype_targets(self, labels):
        return self.prototype_head().targets(labels)

    @torch.no_grad()
    def init_prototypes_from_loader(
        self,
        train_loader,
        device,
        proto_init='mean',
        max_init_samples=8000,
    ):
        if not self.is_prototype_head():
            return
        head = self.prototype_head()
        was_training = self.training
        self.eval()
        all_embeddings = []
        all_labels = []
        for batch in train_loader:
            inputs, labels, channel_ids, channel_mask = unpack_batch(batch, device)
            h = self.encode_features(inputs, channel_ids, channel_mask)
            embeddings = head.encode(h)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            if max_init_samples and sum(t.shape[0] for t in all_embeddings) >= max_init_samples:
                break
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if max_init_samples and embeddings.shape[0] > max_init_samples:
            embeddings = embeddings[:max_init_samples]
            labels = labels[:max_init_samples]
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        head.init_prototypes_from_embeddings(embeddings, labels, proto_init=proto_init)
        if was_training:
            self.train()

    def forward(self, x, channel_ids=None, channel_mask=None):
        if channel_ids is None or channel_mask is None:
            raise ValueError('channel_ids and channel_mask are required')
        return self.main_model(x.unsqueeze(1), channel_ids, channel_mask)

    def filter_sparsity_l1_loss(self, target='normalized_only'):
        if hasattr(self.main_model, 'filter_sparsity_l1_loss'):
            return self.main_model.filter_sparsity_l1_loss(target=target)
        return None


    def collect_gate_stats(self, labels=None):
        if hasattr(self.main_model, 'collect_gate_stats'):
            return self.main_model.collect_gate_stats(labels=labels)
        return {}

class GENTRAPreprocessor(Preprocessor):
    """GENTRA defaults: 250 Hz, 8-32 Hz bandpass, CAR, 4 s, EA."""

    def __init__(
        self,
        target_fs=250,
        l_freq=8,
        h_freq=32,
        notch_freq=None,
        normalize_method='car',
        time_length=4,
        apply_EA=True,
        channel_whitelist=None,
        n_workers=1,
    ):
        super().__init__(
            target_fs=target_fs,
            l_freq=l_freq,
            h_freq=h_freq,
            notch_freq=notch_freq,
            normalize_method=normalize_method,
            time_length=time_length,
            apply_EA=apply_EA,
            channel_whitelist=channel_whitelist,
            n_workers=n_workers,
        )

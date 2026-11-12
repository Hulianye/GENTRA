"""GENTRA model for multi-source motor imagery EEG decoding."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor

from gentra.training.filter_norm import gather_mix_weights, normalize_filter_weights


def _act(name):
    return {'elu': nn.ELU, 'gelu': nn.GELU, 'relu': nn.ReLU}[name.lower()]


def _norm2d(channels, norm_type='batch'):
    if norm_type != 'batch':
        raise ValueError('The released model supports batch normalization only')
    return nn.BatchNorm2d(channels)


class MoESpatialMixer(nn.Module):
    """Channel-gated spatial mixing on joint-temporal features [B, D, C, T]."""

    GATE_FEATURE_MODES = ('layout',)

    def __init__(
        self,
        feature_dim,
        channel_vocab_size=128,
        num_shared_filters=2,
        num_private_filters=16,
        top_k_private=4,
        gate_hidden_dim=64,
        gate_temperature=1.0,
        apply_topk_weight=True,
        gate_feature_mode='layout',
        mixer_mode='stack_gated_topk',
        dropout_rate=0.1,
        activation='gelu',
        stack_kernel=1,
        stack_merge_mode='full_1x1',
        shallow_norm_type='batch',
    ):
        super().__init__()
        if mixer_mode != 'stack_gated_topk':
            raise ValueError('The released model supports stack_gated_topk only')
        if top_k_private > num_private_filters:
            raise ValueError(
                f'top_k_private={top_k_private} must be <= num_private_filters={num_private_filters}'
            )
        if gate_feature_mode not in self.GATE_FEATURE_MODES:
            raise ValueError(f'unsupported gate_feature_mode: {gate_feature_mode}')
        if not apply_topk_weight:
            raise ValueError('The released model requires weighted top-k routing')
        self.feature_dim = feature_dim
        self.channel_vocab_size = channel_vocab_size
        self.num_shared_filters = num_shared_filters
        self.num_private_filters = num_private_filters
        self.top_k_private = top_k_private
        self.gate_temperature = gate_temperature
        self.apply_topk_weight = apply_topk_weight
        self.gate_feature_mode = gate_feature_mode
        self.mixer_mode = mixer_mode
        if stack_merge_mode != 'full_1x1':
            raise ValueError('The released model supports full_1x1 merge only')
        self.stack_merge_mode = stack_merge_mode
        if stack_kernel < 1:
            raise ValueError(f'stack_kernel must be >= 1, got {stack_kernel}')
        self.stack_kernel = stack_kernel
        self.num_active_filters = num_shared_filters + top_k_private
        gate_in_dim = channel_vocab_size
        shared_filter_weight = torch.randn(num_shared_filters, channel_vocab_size) * 0.02
        shared_filter_weight[:, 0] = 0.0
        self.shared_filter_weight = nn.Parameter(shared_filter_weight)
        self.shared_filter_scale = nn.Parameter(torch.ones(num_shared_filters))
        private_filter_weight = torch.randn(num_private_filters, channel_vocab_size) * 0.02
        private_filter_weight[:, 0] = 0.0
        self.private_filter_weight = nn.Parameter(private_filter_weight)
        self.private_filter_scale = nn.Parameter(torch.ones(num_private_filters))
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, num_private_filters),
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)
        stack_in = self.num_active_filters * feature_dim
        stack_pad = (stack_kernel - 1) // 2
        self.stack_conv = nn.Conv2d(
            stack_in,
            feature_dim,
            kernel_size=(1, stack_kernel),
            padding=(0, stack_pad),
            groups=1,
        )
        self.post_stack = nn.Sequential(
            _norm2d(feature_dim, shallow_norm_type),
            _act(activation)(),
            nn.Dropout2d(dropout_rate),
        )
        self._last_gate_logits = None
        self._last_gate_probs = None
        self._last_topk_idx = None

    @staticmethod
    def _channel_histogram(channel_ids: Tensor, channel_mask: Tensor, vocab_size: int) -> Tensor:
        hist = torch.zeros(
            channel_ids.shape[0],
            vocab_size,
            device=channel_ids.device,
            dtype=torch.float32,
        )
        hist.scatter_add_(1, channel_ids, channel_mask.float())
        return hist / channel_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def _gate_features(self, x, channel_ids, channel_mask):
        return self._channel_histogram(
            channel_ids, channel_mask, self.channel_vocab_size
        )

    def _vocab_filter_weights(self, filter_weight: Tensor, filter_scale: Tensor) -> Tensor:
        return normalize_filter_weights(filter_weight, filter_scale)

    def _spatial_mix_weights(self, filter_weight, filter_scale, channel_ids, channel_mask):
        w = self._vocab_filter_weights(filter_weight, filter_scale)
        return gather_mix_weights(w, channel_ids, channel_mask)

    def filter_sparsity_l1_loss(self, target='normalized_only') -> Tensor:
        if target != 'normalized_only':
            raise ValueError('The released model uses normalized_only filter sparsity')
        w_shared = F.normalize(self.shared_filter_weight[:, 1:], dim=1)
        w_private = F.normalize(self.private_filter_weight[:, 1:], dim=1)
        return (w_shared.abs().mean() + w_private.abs().mean()) * 0.5


    def collect_gate_stats(self, labels=None):
        if self._last_gate_logits is None:
            return {}
        probs = self._last_gate_probs
        if probs is None:
            probs = F.softmax(self._last_gate_logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
        stats = {
            'mixer_mode': self.mixer_mode,
            'gate_entropy_mean': float(entropy.mean().item()),
            'gate_entropy_std': float(entropy.std().item()) if entropy.numel() > 1 else 0.0,
            'expert_load': [float(v) for v in probs.mean(dim=0).cpu()],
            'topk_usage': None,
        }
        if self._last_topk_idx is not None:
            flat_idx = self._last_topk_idx.reshape(-1).cpu()
            counts = torch.bincount(flat_idx, minlength=self.num_private_filters).float()
            stats['topk_usage'] = [float(v) for v in (counts / counts.sum().clamp(min=1)).tolist()]
        if labels is not None:
            by_class = {}
            labels_cpu = labels.detach().cpu().tolist()
            for i, label in enumerate(labels_cpu):
                by_class.setdefault(int(label), []).append(float(entropy[i].item()))
            stats['gate_entropy_by_class'] = {
                str(k): float(sum(v) / len(v)) for k, v in by_class.items()
            }
        return stats

    def _select_private(self, private_out, gate_logits, time_tokens, feature_dim):
        topk_idx = gate_logits.topk(self.top_k_private, dim=-1).indices
        topk_logits = gate_logits.gather(1, topk_idx)
        topk_weight = F.softmax(topk_logits / self.gate_temperature, dim=-1)
        self._last_topk_idx = topk_idx.detach()
        gather_idx = topk_idx.unsqueeze(1).unsqueeze(-1).expand(-1, time_tokens, -1, feature_dim)
        selected_private = torch.gather(private_out, 2, gather_idx)
        return selected_private * topk_weight.unsqueeze(1).unsqueeze(-1)

    def forward(self, feat, x_gate, channel_ids, channel_mask):
        batch_size, feature_dim, num_channels, time_tokens = feat.shape
        if channel_ids.shape != (batch_size, num_channels):
            raise ValueError(
                f'channel_ids shape {tuple(channel_ids.shape)} must match channels {num_channels}'
            )
        gate_features = self._gate_features(x_gate, channel_ids, channel_mask)
        gate_logits = self.gate_mlp(gate_features)
        gate_probs = F.softmax(gate_logits, dim=-1)
        self._last_gate_logits = gate_logits.detach()
        self._last_gate_probs = gate_probs.detach()

        tokens_bt = feat.permute(0, 3, 2, 1)
        mix_w_shared = self._spatial_mix_weights(
            self.shared_filter_weight, self.shared_filter_scale, channel_ids, channel_mask
        )
        shared_out = torch.einsum('sbc,btcd->sbtd', mix_w_shared, tokens_bt)
        shared_out = shared_out.permute(1, 2, 0, 3)

        mix_w_private = self._spatial_mix_weights(
            self.private_filter_weight, self.private_filter_scale, channel_ids, channel_mask
        )
        private_out = torch.einsum('pbc,btcd->pbtd', mix_w_private, tokens_bt)
        private_out = private_out.permute(1, 2, 0, 3)
        selected_private = self._select_private(
            private_out, gate_logits, time_tokens, feature_dim,
        )
        filt_out = torch.cat([shared_out, selected_private], dim=2)

        filt_out = self.spatial_norm(filt_out)
        stacked = filt_out.reshape(
            batch_size, time_tokens, self.num_active_filters * feature_dim
        )
        stacked = stacked.permute(0, 2, 1).unsqueeze(2)
        merged = self.stack_conv(stacked)
        merged = self.post_stack(merged)
        return merged


class HybridMoESpatialPatchEmbedding(nn.Module):
    """Shared temporal stem + joint BN, MoE replaces (chn, 1) spatial conv only."""

    def __init__(
        self,
        emb_size=40,
        conv_filters=40,
        conv_temporal_kernel=25,
        conv_temporal_kernel2=15,
        pool_kernel=75,
        pool_stride=15,
        dropout_rate=0.5,
        shallow_dropout=None,
        activation='elu',
        channel_vocab_size=128,
        num_shared_filters=2,
        num_private_filters=16,
        top_k_private=4,
        gate_hidden_dim=64,
        gate_temperature=1.0,
        apply_topk_weight=True,
        gate_feature_mode='layout',
        mixer_mode='stack_gated_topk',
        stack_kernel=1,
        stack_merge_mode='full_1x1',
        shallow_norm_type='batch',
    ):
        super().__init__()
        shallow_drop = dropout_rate if shallow_dropout is None else shallow_dropout
        act = _act(activation)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, conv_filters, (1, conv_temporal_kernel), (1, 1)),
            _norm2d(conv_filters, shallow_norm_type),
            act(),
            nn.Conv2d(conv_filters, conv_filters, (1, conv_temporal_kernel2), (1, 1)),
            _norm2d(conv_filters, shallow_norm_type),
            act(),
        )
        self.moe_mixer = MoESpatialMixer(
            feature_dim=conv_filters,
            channel_vocab_size=channel_vocab_size,
            num_shared_filters=num_shared_filters,
            num_private_filters=num_private_filters,
            top_k_private=top_k_private,
            gate_hidden_dim=gate_hidden_dim,
            gate_temperature=gate_temperature,
            apply_topk_weight=apply_topk_weight,
            gate_feature_mode=gate_feature_mode,
            mixer_mode=mixer_mode,
            dropout_rate=dropout_rate if shallow_dropout is None else shallow_dropout,
            activation=activation,
            stack_kernel=stack_kernel,
            stack_merge_mode=stack_merge_mode,
            shallow_norm_type=shallow_norm_type,
        )
        self.post_spatial = nn.Sequential(
            _norm2d(conv_filters, shallow_norm_type),
            act(),
        )
        self.pool = nn.AvgPool2d((1, pool_kernel), (1, pool_stride))
        self.shallow_dropout = nn.Dropout(shallow_drop)
        self.projection = nn.Sequential(
            nn.Conv2d(conv_filters, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.conv_filters = conv_filters


    def filter_sparsity_l1_loss(self, target='normalized_only') -> Tensor:
        return self.moe_mixer.filter_sparsity_l1_loss(target=target)

    def collect_gate_stats(self, labels=None):
        return self.moe_mixer.collect_gate_stats(labels=labels)

    def _encode_spatial(self, x: Tensor, channel_ids: Tensor, channel_mask: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if channel_mask is not None:
            mask_f = channel_mask.to(dtype=x.dtype).unsqueeze(1).unsqueeze(-1)
            x = x * mask_f
            valid = channel_mask.bool()
            valid_x = x.squeeze(1)[valid][:, None, None, :]
            valid_feat = self.temporal(valid_x).squeeze(2)
            feat = valid_feat.new_zeros((*valid.shape, *valid_feat.shape[1:]))
            feat[valid] = valid_feat
            feat = feat.permute(0, 2, 1, 3)
        else:
            feat = self.temporal(x)
        feat = self.moe_mixer(feat, x, channel_ids, channel_mask)
        return self.post_spatial(feat)

    def encode_tokens_and_targets(
        self,
        x: Tensor,
        channel_ids: Tensor,
        channel_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        feat = self._encode_spatial(x, channel_ids, channel_mask)
        pooled = self.pool(feat)
        conv_target = pooled.squeeze(2).permute(0, 2, 1)
        pooled = self.shallow_dropout(pooled)
        patch_tokens = self.projection(pooled)
        return patch_tokens, conv_target

    def forward(self, x: Tensor, channel_ids: Tensor, channel_mask: Tensor) -> Tensor:
        patch_tokens, _ = self.encode_tokens_and_targets(x, channel_ids, channel_mask)
        return patch_tokens


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryPositionEmbedding(nn.Module):
    """RoPE on token sequence [B, N, D]; D must be even."""

    def __init__(self, dim, max_seq_len=256, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f'RoPE dim must be even, got {dim}')
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, :], persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[1]
        cos = self.cos_cached[:, offset: offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, offset: offset + seq_len, :].to(dtype=x.dtype, device=x.device)
        return _apply_rotary_pos_emb(x, cos, sin)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
        dropout,
        attention_scale='emb_size',
        rope_mode='attention_qk',
        rope_max_seq_len=256,
    ):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(f'emb_size={emb_size} must be divisible by num_heads={num_heads}')
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.attention_scale = attention_scale
        if attention_scale != 'emb_size':
            raise ValueError('The released model uses emb_size attention scaling')
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        if rope_mode != 'attention_qk':
            raise ValueError('The released model supports attention_qk RoPE only')
        self.qk_rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=rope_max_seq_len)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        seq_len = queries.shape[2]
        cos = self.qk_rope.cos_cached[:, None, :seq_len, :].to(
            dtype=queries.dtype, device=queries.device
        )
        sin = self.qk_rope.sin_cached[:, None, :seq_len, :].to(
            dtype=queries.dtype, device=queries.device
        )
        queries = _apply_rotary_pos_emb(queries, cos, sin)
        keys = _apply_rotary_pos_emb(keys, cos, sin)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads=None,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
        attention_scale='emb_size',
        rope_mode='attention_qk',
        rope_max_seq_len=256,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = 10 if emb_size % 10 == 0 else 8 if emb_size % 8 == 0 else 4
        self.attn_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(
                emb_size,
                num_heads,
                drop_p,
                attention_scale=attention_scale,
                rope_mode=rope_mode,
                rope_max_seq_len=rope_max_seq_len,
            ),
            nn.Dropout(drop_p),
        ))
        self.ff_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
            nn.Dropout(drop_p),
        ))

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.attn_block(x, **kwargs)
        x = self.ff_block(x, **kwargs)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth,
        emb_size,
        num_heads=None,
        dropout_rate=0.5,
        attention_scale='emb_size',
        rope_mode='attention_qk',
        rope_max_seq_len=256,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                emb_size,
                num_heads=num_heads,
                drop_p=dropout_rate,
                forward_drop_p=dropout_rate,
                attention_scale=attention_scale,
                rope_mode=rope_mode,
                rope_max_seq_len=rope_max_seq_len,
            )
            for _ in range(depth)
        ])

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class AddCLSToken(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.size(0))
        return torch.cat([cls_tokens, x], dim=1)


class PrototypeClassificationHead(nn.Module):
    def __init__(
        self,
        emb_size,
        n_classes,
        proj_dim=None,
        temperature=0.1,
        num_prototypes=1,
        prototype_logit_aggregation='logmeanexp',
    ):
        super().__init__()
        proj_dim = proj_dim or emb_size
        self.emb_size = emb_size
        self.proj_dim = proj_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        if prototype_logit_aggregation != 'logmeanexp':
            raise ValueError(
                f'unsupported prototype_logit_aggregation: {prototype_logit_aggregation}'
            )
        self.prototype_logit_aggregation = prototype_logit_aggregation
        self.projector = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Linear(emb_size, proj_dim),
        )
        self.register_buffer('class_ids', torch.arange(n_classes, dtype=torch.long))
        self.prototypes = nn.Parameter(torch.empty(n_classes, num_prototypes, proj_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def encode(self, x):
        z = self.projector(x[:, 0])
        return F.normalize(z, dim=1)

    def logits_from_embeddings(self, embeddings):
        prototypes = F.normalize(self.prototypes, dim=2)
        proto_logits = torch.einsum('bd,ckd->bck', embeddings, prototypes) / self.temperature
        if self.num_prototypes == 1:
            return proto_logits.squeeze(2)
        return torch.logsumexp(proto_logits, dim=2) - math.log(proto_logits.size(2))

    def forward(self, x):
        return self.logits_from_embeddings(self.encode(x))

    def targets(self, labels):
        return torch.searchsorted(self.class_ids.to(labels.device), labels)

    @torch.no_grad()
    def init_prototypes_from_embeddings(self, embeddings, labels, proto_init='mean'):
        prototypes = []
        for idx, class_id in enumerate(self.class_ids.to(labels.device)):
            class_embeddings = embeddings[labels == class_id]
            if class_embeddings.numel() == 0:
                class_prototypes = self.prototypes[idx].clone()
            elif proto_init == 'kmeans' and class_embeddings.shape[0] >= self.num_prototypes:
                from sklearn.cluster import KMeans

                centers = KMeans(
                    n_clusters=self.num_prototypes,
                    n_init=3,
                    random_state=0,
                ).fit(class_embeddings.cpu().numpy()).cluster_centers_
                class_prototypes = torch.tensor(
                    centers,
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
            else:
                prototype = class_embeddings.mean(dim=0)
                class_prototypes = prototype.unsqueeze(0).repeat(self.num_prototypes, 1)
                if self.num_prototypes > 1:
                    class_prototypes = class_prototypes + torch.randn_like(class_prototypes) * 0.01
            prototypes.append(F.normalize(class_prototypes, dim=1))
        self.prototypes.copy_(torch.stack(prototypes, dim=0).to(self.prototypes.device))


class GENTRA(nn.Module):
    def __init__(
        self,
        emb_size=40,
        depth=6,
        n_classes=4,
        conv_filters=40,
        conv_temporal_kernel=25,
        conv_temporal_kernel2=15,
        pool_kernel=75,
        pool_stride=15,
        num_heads=None,
        dropout_rate=0.5,
        shallow_dropout=None,
        transformer_dropout=None,
        head_type='cls_prototype',
        activation='elu',
        proj_dim=None,
        temperature=0.1,
        num_prototypes=1,
        prototype_logit_aggregation='logmeanexp',
        channel_vocab_size=0,
        moe_shared_filters=2,
        moe_private_filters=16,
        moe_top_k=4,
        moe_gate_hidden_dim=64,
        moe_gate_temperature=1.0,
        moe_apply_topk_weight=True,
        moe_gate_feature_mode='layout',
        moe_mixer_mode='stack_gated_topk',
        moe_stack_kernel=1,
        moe_stack_merge_mode='full_1x1',
        shallow_norm_type='batch',
        use_rope=True,
        rope_max_seq_len=256,
        rope_mode='attention_qk',
        attention_scale='emb_size',
        **kwargs,
    ):
        super().__init__()
        if not use_rope:
            raise ValueError('The released model requires attention_qk RoPE')
        self.rope_mode = rope_mode
        if rope_mode != 'attention_qk':
            raise ValueError('The released model supports attention_qk RoPE only')
        shallow_drop = dropout_rate if shallow_dropout is None else shallow_dropout
        transformer_drop = dropout_rate if transformer_dropout is None else transformer_dropout
        resolved_heads = num_heads
        if resolved_heads is None:
            resolved_heads = 10 if emb_size % 10 == 0 else 8 if emb_size % 8 == 0 else 4
        if channel_vocab_size <= 0:
            raise ValueError('channel_vocab_size must be > 0')
        self.patch_embedding = HybridMoESpatialPatchEmbedding(
            emb_size=emb_size,
            conv_filters=conv_filters,
            conv_temporal_kernel=conv_temporal_kernel,
            conv_temporal_kernel2=conv_temporal_kernel2,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout_rate=dropout_rate,
            shallow_dropout=shallow_drop,
            activation=activation,
            channel_vocab_size=channel_vocab_size,
            num_shared_filters=moe_shared_filters,
            num_private_filters=moe_private_filters,
            top_k_private=moe_top_k,
            gate_hidden_dim=moe_gate_hidden_dim,
            gate_temperature=moe_gate_temperature,
            apply_topk_weight=moe_apply_topk_weight,
            gate_feature_mode=moe_gate_feature_mode,
            mixer_mode=moe_mixer_mode,
            stack_kernel=moe_stack_kernel,
            stack_merge_mode=moe_stack_merge_mode,
            shallow_norm_type=shallow_norm_type,
        )
        self.cls_token = AddCLSToken(emb_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.transformer = TransformerEncoder(
            depth,
            emb_size,
            num_heads=resolved_heads,
            dropout_rate=transformer_drop,
            attention_scale=attention_scale,
            rope_mode=rope_mode,
            rope_max_seq_len=rope_max_seq_len,
        )
        if head_type != 'cls_prototype':
            raise ValueError(f'Unsupported head_type={head_type!r}; only cls_prototype is supported')
        self.head = PrototypeClassificationHead(
            emb_size,
            n_classes,
            proj_dim=proj_dim or emb_size,
            temperature=temperature,
            num_prototypes=num_prototypes,
            prototype_logit_aggregation=prototype_logit_aggregation,
        )


    def encode_patch_tokens(self, x, channel_ids=None, channel_mask=None):
        """Patch embedding output [B, T, E] without AddCLSToken."""
        return self.patch_embedding(x, channel_ids, channel_mask)

    def encode_patch_tokens_and_targets(self, x, channel_ids=None, channel_mask=None):
        return self.patch_embedding.encode_tokens_and_targets(x, channel_ids, channel_mask)

    def encode_from_patch_tokens(self, patch_tokens):
        x = self.cls_token(patch_tokens)
        return self.transformer(x)

    def apply_token_mask(self, patch_tokens, token_mask):
        mask_val = self.mask_token.expand(patch_tokens.size(0), patch_tokens.size(1), -1)
        expand_mask = token_mask.unsqueeze(-1).expand_as(patch_tokens)
        return torch.where(expand_mask, mask_val, patch_tokens)

    def encode_features(self, x, channel_ids=None, channel_mask=None, patch_tokens=None):
        if patch_tokens is None:
            patch_tokens = self.patch_embedding(x, channel_ids, channel_mask)
        return self.encode_from_patch_tokens(patch_tokens)

    def forward(self, x, channel_ids=None, channel_mask=None):
        x = self.encode_features(x, channel_ids=channel_ids, channel_mask=channel_mask)
        return self.head(x)


    def filter_sparsity_l1_loss(self, target='normalized_only'):
        pe = self.patch_embedding
        if hasattr(pe, 'filter_sparsity_l1_loss'):
            return pe.filter_sparsity_l1_loss(target=target)
        return None

    def collect_gate_stats(self, labels=None):
        pe = self.patch_embedding
        if hasattr(pe, 'collect_gate_stats'):
            return pe.collect_gate_stats(labels=labels)
        return {}

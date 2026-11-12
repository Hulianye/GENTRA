"""Per-sample cross-entropy for the GENTRA prototype classifier."""

import torch.nn.functional as F


def prototype_cross_entropy(model, outputs, labels, args):
    return F.cross_entropy(
        outputs,
        model.prototype_targets(labels),
        reduction='none',
        label_smoothing=getattr(args, 'label_smoothing', 0.0),
        weight=getattr(args, 'class_weight_tensor', None),
    )

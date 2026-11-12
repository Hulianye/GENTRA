import numpy as np
from gentra.evaluation.scpf import scpf_logits


def test_novel_class_uses_only_support():
    query = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    support = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    selected = {0: np.array([0]), 1: np.array([1])}
    checkpoints_a = {0: np.array([[1., 0.]]), 1: np.array([[1., 0.]])}
    checkpoints_b = {0: np.array([[1., 0.]]), 1: np.array([[-1., 0.]])}
    a = scpf_logits(query, support, selected, checkpoints_a, (0, 1), 1.0, novel_class_ids=(1,))
    b = scpf_logits(query, support, selected, checkpoints_b, (0, 1), 1.0, novel_class_ids=(1,))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, np.eye(2, dtype=np.float32))

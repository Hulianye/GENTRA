"""Old preprocessing caches must not silently enter the public pipeline."""

import pickle

import numpy as np
import pytest

from gentra.data.multi_source_mi import preprocess_source
from gentra.data.zs_test_cache import ZSTestSpec, load_or_build_zs_test, zs_cache_path


def test_fresh_and_cached_source_match(tmp_path):
    from gentra.data.multi_source_mi import load_and_preprocess_source

    source = tmp_path / 'input.pkl'
    channels = ['C3', 'CZ', 'C4']
    with source.open('wb') as handle:
        pickle.dump({
            'dataset_name': 'synthetic',
            'eeg_data': np.random.default_rng(0).standard_normal((4, 3, 128)).astype('float32'),
            'labels': np.array([0, 1, 0, 1]), 'subject_ids': np.array([1, 1, 2, 2]),
            'channel_names': channels, 'sampling_rate': 128,
        }, handle)
    pre = dict(target_fs=128, l_freq=None, h_freq=None, norm_method='none',
               time_length=1.0, apply_EA=False)
    args = ('synthetic', str(source), list(reversed(channels)), pre, str(tmp_path / 'cache'))
    fresh = load_and_preprocess_source(*args)
    source.unlink()  # A cache hit must be independent of the raw source file.
    cached = load_and_preprocess_source(*args)
    for field in ('x', 'y', 'subject_ids'):
        np.testing.assert_array_equal(getattr(fresh, field), getattr(cached, field))
    assert fresh.channel_names == cached.channel_names == channels


def test_identityless_preprocessing_cache_is_rejected(tmp_path):
    cache = tmp_path / 'source.pkl'
    with cache.open('wb') as handle:
        pickle.dump({
            'x': np.zeros((2, 2, 10), dtype=np.float32),
            'y': np.array([0, 1]),
            'subject_ids': np.array([1, 1]),
        }, handle)
    with pytest.raises(ValueError, match='lacks preprocessing identity'):
        preprocess_source(None, {}, cache_path=str(cache))


def test_identityless_zero_shot_cache_is_rejected(tmp_path):
    spec = ZSTestSpec('synthetic', str(tmp_path / 'missing.pkl'), ['C3'])
    cache = zs_cache_path(str(tmp_path), spec.name, {}, False)
    with open(cache, 'wb') as handle:
        pickle.dump({'x': np.zeros((1, 1, 10)), 'y': np.array([0]), 'subject_ids': np.array([1])}, handle)
    with pytest.raises(ValueError, match='lacks preprocessing identity'):
        load_or_build_zs_test(spec, {}, {'C3': 1}, str(tmp_path))

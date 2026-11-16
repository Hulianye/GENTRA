"""Scenario-specific downstream preprocessing cache I/O."""

import os
import pickle

from gentra.data.eeg_data import EEGData


def save_preprocessed(eeg_data, cache_path):
    if not isinstance(eeg_data, EEGData):
        raise TypeError('Expected EEGData instance')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(eeg_data, f)
    return cache_path


def load_preprocessed(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'Preprocessed cache not found: {cache_path}')
    with open(cache_path, 'rb') as f:
        eeg_data = pickle.load(f)
    if not isinstance(eeg_data, EEGData):
        raise TypeError(f'Cache does not contain EEGData: {type(eeg_data)}')
    return eeg_data

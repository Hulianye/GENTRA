"""The exporter and downstream loader agree on the public bundle format."""
import pickle
import numpy as np
from gentra.data.eeg_data import EEGData, load_eeg_data_from_pkl

def test_exported_dict_loads_for_finetuning(tmp_path):
    path = tmp_path / "example.pkl"
    bundle = {"dataset_name": "example", "eeg_data": np.zeros((2, 4, 1000), dtype=np.float32), "labels": np.array([0, 1]), "subject_ids": np.array([1, 1]), "channel_names": ["C3", "C4", "CZ", "FZ"], "sampling_rate": 250}
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    loaded = load_eeg_data_from_pkl(path)
    assert isinstance(loaded, EEGData)
    assert loaded.eeg_data.shape == (2, 4, 1000)
    assert loaded.get_label_count() == 2

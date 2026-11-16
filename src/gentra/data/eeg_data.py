import pickle
import os
from pathlib import Path

import numpy as np

class EEGData:
    """Container for EEG dataset: data array, subject/trial ids, labels, and optional usage (train/test)."""

    def __init__(self, dataset_name, eeg_data, subject_ids, channel_names, sampling_rate,
                 labels=None, dataset_type='classification', split_method='LOSO', is_binary=False, usage=None):

        self.dataset_name = dataset_name
        self.eeg_data = np.array(eeg_data)
        self.subject_ids = np.array(subject_ids)
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate

        if labels is not None:
            if not isinstance(labels, np.ndarray):
                self.labels = np.array(labels)
            else:
                self.labels = labels
        else:
            self.labels = None

        if dataset_type != 'classification':
            raise ValueError("GENTRA supports classification datasets only")
        self.dataset_type = dataset_type

        self.split_method = split_method
        self.is_binary = is_binary

        if usage is not None:
            if not isinstance(usage,np.ndarray):
               self.usage = np.array(usage)
            else:
                self.usage = usage
        else:
            self.usage = None
    
    def get_sample_count(self):
        return self.eeg_data.shape[0]

    def get_channel_count(self):
        return self.eeg_data.shape[1]

    def get_time_point_count(self):
        return self.eeg_data.shape[2]

    def get_duration(self):
        return self.get_time_point_count() / self.sampling_rate

    def get_subject_ids(self):
        return self.subject_ids

    def get_channel_names(self):
        return self.channel_names

    def get_subject_unique_count(self):
        return len(np.unique(self.subject_ids))

    def get_label_count(self):
        if self.labels is None:
            return 0
        return len(np.unique(self.labels))

    def get_data_by_label(self, label):
        """Return EEG data for samples with the given label."""
        if self.labels is None:
            raise ValueError("Dataset has no labels")
        mask = self.labels == label
        return self.eeg_data[mask]
    
    def __str__(self):
        result = (
            f"Dataset: {self.dataset_name}\n"
            f"Samples: {self.get_sample_count()}, Channels: {self.get_channel_count()}, Time points: {self.get_time_point_count()}\n"
            f"Sampling rate: {self.sampling_rate} Hz, Duration per sample: {self.get_duration():.2f} s\n"
            f"Subjects: {self.get_subject_unique_count()}, Task: {self.dataset_type}\n"
        )
        if self.labels is not None:
            result += f"Labels: {self.get_label_count()} classes, {list(np.unique(self.labels))}\n"
        return result


def load_eeg_data_from_pkl(file_path):
    """Load an EEGData instance from a pkl file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        eeg_data_obj = pickle.load(f)
    if isinstance(eeg_data_obj, dict):
        eeg_data_obj = EEGData(
            dataset_name=eeg_data_obj.get("dataset_name", Path(file_path).stem),
            eeg_data=eeg_data_obj["eeg_data"],
            subject_ids=eeg_data_obj["subject_ids"],
            channel_names=eeg_data_obj["channel_names"],
            sampling_rate=eeg_data_obj["sampling_rate"],
            labels=eeg_data_obj["labels"],
        )
    if not isinstance(eeg_data_obj, EEGData):
        raise TypeError(f"Loaded object is not EEGData: {type(eeg_data_obj)}")
    print(f"Loaded EEG data from {file_path}")
    return eeg_data_obj

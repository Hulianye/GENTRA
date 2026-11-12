"""Shared metadata and full fine-tuning controls for GENTRA models."""

import torch.nn as nn

from gentra.data.eeg_data import EEGData


class ModelLoader(nn.Module):
    def __init__(self, eeg_dataset, finetune_strategy='full'):
        super().__init__()
        if not isinstance(eeg_dataset, EEGData):
            raise TypeError('eeg_dataset must be an EEGData instance')
        if eeg_dataset.dataset_type != 'classification':
            raise ValueError('GENTRA supports classification datasets only')
        self.ch_names = eeg_dataset.channel_names
        self.num_channels = len(self.ch_names)
        self.num_time_points = eeg_dataset.get_time_point_count()
        self.num_time = eeg_dataset.get_duration()
        self.sampling_rate = eeg_dataset.sampling_rate
        self.dataset_name = getattr(eeg_dataset, 'dataset_name', 'unknown')
        self.dataset_type = 'classification'
        self.nb_classes = eeg_dataset.get_label_count() if eeg_dataset.labels is not None else 1
        self.task_type = 'classification'
        self.finetune_strategy = finetune_strategy
        self.main_model = None

    def apply_finetune_strategy(self, strategy=None):
        """Enable full fine-tuning for the released model."""
        strategy = strategy or self.finetune_strategy
        if strategy not in ('full', None):
            raise ValueError(f"Unsupported finetune strategy: {strategy}. Use 'full'")
        for param in self.parameters():
            param.requires_grad = True
        return self

    def forward(self, x):
        raise NotImplementedError('Subclasses must implement forward')

    def get_model_info(self):
        return {
            'model_name': self.__class__.__name__,
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'num_channels': self.num_channels,
            'num_time_points': self.num_time_points,
            'sampling_rate': self.sampling_rate,
            'nb_classes': self.nb_classes,
            'finetune_strategy': self.finetune_strategy,
        }

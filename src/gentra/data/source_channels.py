"""Channel name maps for MI training and evaluation sources."""

from __future__ import annotations

from gentra.data.channel_constants import CHANNEL_NAMES

SOURCE_CHANNEL_NAMES = dict(CHANNEL_NAMES)

BNCI2014001_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',
]
BNCI2014004_CHANNELS = ['C3', 'Cz', 'C4']
BNCI2015001_CHANNELS = [
    'FC3', 'FCZ', 'FC4', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP3', 'CPZ', 'CP4',
]

LEE2019_CHANNELS = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'CZ', 'C4', 'T8',
    'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10',
    'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPZ', 'CP4', 'P1', 'P2', 'POZ', 'FT9', 'FTT9H', 'TPP7H', 'TP7',
    'TPP9H', 'FT10', 'FTT10H', 'TPP8H', 'TP8', 'TPP10H', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4',
]

PHYSIONET_MOTOR_IMAGERY_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ',
]

ZHOU2016_CHANNELS = [
    'FP1', 'FP2', 'FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4',
    'CP3', 'CPZ', 'CP4', 'O1', 'OZ', 'O2',
]

SOURCE_CHANNEL_NAMES.setdefault('Lee2019', LEE2019_CHANNELS)
SOURCE_CHANNEL_NAMES.setdefault('PhysionetMotorImagery', PHYSIONET_MOTOR_IMAGERY_CHANNELS)
SOURCE_CHANNEL_NAMES.setdefault('Zhou2016', ZHOU2016_CHANNELS)

def channel_names_for_source(source_name: str) -> list[str]:
    if source_name in SOURCE_CHANNEL_NAMES:
        return list(SOURCE_CHANNEL_NAMES[source_name])
    if source_name == 'BNCI2014001':
        return list(BNCI2014001_CHANNELS)
    if source_name == 'BNCI2014004':
        return list(BNCI2014004_CHANNELS)
    if source_name == 'BNCI2015001':
        return list(BNCI2015001_CHANNELS)
    raise KeyError(f'Unknown source channel list: {source_name}')

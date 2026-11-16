"""Channel lists for the released GENTRA training sources."""

CHO_ORIGIN_CHN_NAMES = [
    "FP1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2",
    "AF8", "AF4", "AFZ", "FZ", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCZ", "CZ", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

BRANDL2020_CHN_NAMES = [
    "FP1", "FP2", "AF3", "AF4", "F9", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F10", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "CFC7",
    "CFC5", "CFC3", "CFC1", "CFC2", "CFC4", "CFC6", "CFC8", "T7", "C5",
    "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "CCP7", "CCP5", "CCP3", "CCP1",
    "CCP2", "CCP4", "CCP6", "CCP8", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "CP6", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "PO3", "PO4", "O1", "O2",
]

LIU2024_CHN_NAMES = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FCZ", "FC3", "FC4",
    "FT7", "FT8", "CZ", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7",
    "TP8", "PZ", "P3", "P4", "T5", "T6", "OZ", "O1", "O2",
]

WEIBO_CHN_NAMES = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6",
    "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
    "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2",
    "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5",
    "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
]

ZHOU2020_CHN_NAMES = [
    "F3", "F1", "FZ", "F2", "F4", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3",
    "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]

CHANNEL_NAMES = {
    "Cho2017": CHO_ORIGIN_CHN_NAMES,
    "Brandl2020": BRANDL2020_CHN_NAMES,
    "Liu2024": LIU2024_CHN_NAMES,
    "Weibo2014": WEIBO_CHN_NAMES,
    "Zhou2020": ZHOU2020_CHN_NAMES,
}

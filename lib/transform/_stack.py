import numpy as np
from lib.logging import get_logger

logger = get_logger()


class SpectrogramStacker:
    def __init__(self, labels):
        self.labels = labels

    def stack(self, spectrograms_per_epoch):
        if len(spectrograms_per_epoch) != len(self.labels):
            raise ValueError("Mismatch between spectrogram epochs and labels.")
        
        stacked_spectrograms = np.array(spectrograms_per_epoch) 
        labels = np.array(self.labels) 
        return stacked_spectrograms, labels

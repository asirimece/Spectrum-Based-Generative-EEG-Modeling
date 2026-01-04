import numpy as np
import random
from typing import Dict, Any

class SpectrogramAugmenter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.array([self.augment(spectrogram) for spectrogram in X])
    
    def apply_time_warping(self, spectrogram: np.ndarray, max_warp: int) -> np.ndarray:
        warp_amount = random.randint(-max_warp, max_warp)
        warped = np.roll(spectrogram, warp_amount, axis=1)
        return warped

    def apply_noise_injection(self, spectrogram: np.ndarray, noise_level: float) -> np.ndarray:
        noise = np.random.normal(0, noise_level, spectrogram.shape)
        return spectrogram + noise

    def apply_frequency_shifting(self, spectrogram: np.ndarray, shift: int) -> np.ndarray:
        shifted = np.roll(spectrogram, shift, axis=0)
        return shifted

    def apply_amplitude_scaling(self, spectrogram: np.ndarray, scale_range: list) -> np.ndarray:
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        return spectrogram * scale_factor

    def augment(self, spectrogram: np.ndarray) -> np.ndarray:
        if self.config["time_warping"]["enabled"]:
            spectrogram = self.apply_time_warping(
                spectrogram, self.config["time_warping"]["max_warp"]
            )

        if self.config["noise_injection"]["enabled"]:
            spectrogram = self.apply_noise_injection(
                spectrogram, self.config["noise_injection"]["noise_level"]
            )

        if self.config["frequency_shifting"]["enabled"]:
            spectrogram = self.apply_frequency_shifting(
                spectrogram, self.config["frequency_shifting"]["shift"]
            )

        if self.config["amplitude_scaling"]["enabled"]:
            spectrogram = self.apply_amplitude_scaling(
                spectrogram, self.config["amplitude_scaling"]["scale_range"]
            )

        return spectrogram

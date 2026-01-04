import numpy as np
from scipy.signal import istft
from sklearn.base import BaseEstimator, TransformerMixin

class TimeSignalReconstructor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 fs: int = 128,
                 nperseg: int = 64,
                 noverlap: int = 48,
                 n_fft: int = 256,
                 target_signal_length: int = None,
                 original_data_min: float = None,
                 original_data_max: float = None,
                 feature_range: tuple = (-1.0, 1.0)):

        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.n_fft = n_fft
        self.target_signal_length = target_signal_length
        
        self.original_data_min = original_data_min
        self.original_data_max = original_data_max
        self.feature_range = feature_range

    def fit(self, X, y=None):
        return self

    def transform(self, log_mag, phase):
        if log_mag.shape != phase.shape:
            raise ValueError("log_mag and phase must have the same shape.")

        n_epochs   = log_mag.shape[0]
        n_channels = log_mag.shape[1]

        reconstructed_epochs = []

        for e in range(n_epochs):
            epoch_signals = []
            for ch in range(n_channels):
                log_mag_ch = log_mag[e, ch]
                ph_ch = phase[e, ch]

                # Revert normalization
                log_mag_ch = self._unscale_min_max(log_mag_ch)
                
                # Exponentiate to get the magnitude back
                magnitude = np.exp(log_mag_ch)

                # Reconstruct the complex STFT, reintegrate phase
                Zxx_approx = magnitude * np.exp(1j * ph_ch)

                _, time_signal = istft(
                    Zxx_approx,
                    fs=self.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    nfft=self.n_fft
                )

                # Pad/trim to match a target length
                if self.target_signal_length is not None:
                    length = time_signal.shape[0]
                    if length > self.target_signal_length:
                        time_signal = time_signal[:self.target_signal_length]
                    elif length < self.target_signal_length:
                        diff = self.target_signal_length - length
                        time_signal = np.pad(time_signal, (0, diff), mode='constant')

                epoch_signals.append(time_signal)

            epoch_signals = np.array(epoch_signals)
            reconstructed_epochs.append(epoch_signals)

        time_signals = np.array(reconstructed_epochs)
        return time_signals

    def _unscale_min_max(self, scaled_log_mag):
        if self.original_data_min is None or self.original_data_max is None:
            raise ValueError("original_data_min / original_data_max must be provided for min–max unscale.")
        if self.original_data_min == self.original_data_max:
            raise ValueError("original_data_min equals original_data_max. Invalid for min–max unscale.")

        min_r, max_r = self.feature_range

        scaled_01 = (scaled_log_mag - min_r) / (max_r - min_r)

        unscaled = scaled_01 * (self.original_data_max - self.original_data_min) + self.original_data_min
        return unscaled
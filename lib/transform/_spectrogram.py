import numpy as np
from joblib import Parallel, delayed
from scipy.signal import stft


class STFTComputer:
    def __init__(self, sfreq: int, params: dict):
        self.sfreq = sfreq
        self.params = params

        # Store phase for iSTFT
        self.phase_data = None
        self.original_signal_length = None
        
        self.spectrogram_min  = None
        self.spectrogram_max  = None
        self.feature_range    = None
        
        self.nperseg = int(self.sfreq * self.params['window_size'])
        self.noverlap = int(self.nperseg * self.params['overlap'])
        self.n_fft = self.params['n_fft']
        
        self.fmin = self.params.get('fmin', 0)
        self.fmax = self.params.get('fmax', sfreq / 2.0)

        self.remove_border_bins = self.params.get('remove_border_bins', False)

        # Epsilon floor to avoid log(0)
        self.log_power_floor = self.params.get('log_power_floor', 1e-12)


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (n_epochs, n_channels, n_samples), got {X.shape}")
        
        self.original_signal_length = X.shape[-1]

        results = Parallel(n_jobs=-1)(
            delayed(self._compute_multichannel_stft)(epoch_data) for epoch_data in X
        )

        log_power_list = [r[0] for r in results]
        phase_list     = [r[1] for r in results]

        log_power_out = np.stack(log_power_list, axis=0)
        phase_out     = np.stack(phase_list, axis=0)

        # For model compatibility
        self.phase_data = phase_out.astype(np.float32)
        log_power_out = log_power_out.astype(np.float32)

        return log_power_out

    def _compute_multichannel_stft(self, epoch_data):
        channel_log_power = []
        channel_phases    = []

        for ch_data in epoch_data:
            freqs, times, Zxx = stft(
                ch_data,
                fs=self.sfreq,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.n_fft,
                return_onesided=True
            )
            
            mask = (freqs < self.fmin) | (freqs > self.fmax)
            Zxx[mask, :] = 0.0

            # Magnitude spectrogram
            power = np.abs(Zxx)   
            phase = np.angle(Zxx)

            power = np.maximum(power, self.log_power_floor)
            log_power = np.log(power)

            channel_log_power.append(log_power)
            channel_phases.append(phase)

        # (n_channels, n_freqs, n_times)
        log_power_3d = np.array(channel_log_power, dtype=np.float32)
        phase_3d     = np.array(channel_phases,   dtype=np.float32)

        return log_power_3d, phase_3d

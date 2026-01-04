import torch
from torch import Tensor, as_tensor
from sklearn.utils import check_random_state
from numbers import Real
from torch.fft import fft, ifft
from torch.nn.functional import pad
import numpy as np
from numpy import ndarray


def _pick_channels_randomly(X: Tensor, p_pick: float, random_state: int | None = None) -> Tensor:
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    # allows to use the same RNG
    uniform_samples = torch.as_tensor(
        rng.uniform(0, 1, size=(batch_size, n_channels)),
        dtype=torch.float,
        device=X.device,
    )
    # equivalent to a 0s and 1s mask
    return torch.sigmoid(1000 * (uniform_samples - p_pick))


def channels_dropout(X: Tensor, y: Tensor, p_drop: float, random_state: int | None = None) -> (Tensor, Tensor):
    """Randomly set channels to flat signal.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    p_drop : float
        Float between 0 and 1 setting the probability of dropping each channel.
    random_state : int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """
    mask = _pick_channels_randomly(X, p_drop, random_state=random_state)
    return X * mask.unsqueeze(-1), y


def _new_random_fft_phase_odd(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
    ).to(device)
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        -torch.flip(random_phase, [-1])
    ], dim=-1)


def _new_random_fft_phase_even(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
    ).to(device)
    return torch.cat([
        torch.zeros((batch_size, c, 1), device=device),
        random_phase,
        torch.zeros((batch_size, c, 1), device=device),
        -torch.flip(random_phase, [-1])
    ], dim=-1)


_new_random_fft_phase = {
    0: _new_random_fft_phase_even,
    1: _new_random_fft_phase_odd
}


def ft_surrogate(
        X: Tensor,
        y: Tensor,
        phase_noise_magnitude: float,
        channel_independence: bool = False,
        random_state: int | None = None
) -> (Tensor, Tensor):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

    Function copied from https://github.com/cliffordlab/sleep-convolutions-tf
    and modified.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    phase_noise_magnitude: float
        Float between 0 and 1 setting the range over which the phase
        perturbation is uniformly sampled:
        [0, `phase_noise_magnitude` * 2 * `pi`].
    channel_independence : bool
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI.
    random_state: int | numpy.random.Generator, optional
        Used to draw the phase perturbation. Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    assert isinstance(
        phase_noise_magnitude,
        (Real, torch.FloatTensor, torch.cuda.FloatTensor)
    ) and 0 <= phase_noise_magnitude <= 1, (
        f"eps must be a float between 0 and 1. Got {phase_noise_magnitude}.")

    f = fft(X.double(), dim=-1)
    device = X.device

    n = f.shape[-1]
    random_phase = _new_random_fft_phase[n % 2](
        f.shape[0],
        f.shape[-2] if channel_independence else 1,
        n,
        device=device,
        random_state=random_state
    )
    if not channel_independence:
        random_phase = torch.tile(random_phase, (1, f.shape[-2], 1))
    if isinstance(phase_noise_magnitude, torch.Tensor):
        phase_noise_magnitude = phase_noise_magnitude.to(device)
    f_shifted = f * torch.exp(phase_noise_magnitude * random_phase)
    shifted = ifft(f_shifted, dim=-1)
    transformed_X = shifted.real.type(X.dtype)

    return transformed_X, y


def _get_mask_start_per_sample(X: Tensor, mask_len_samples: int, random_state: int | None = None) -> Tensor:
    seq_length = torch.as_tensor(X.shape[-1], device=X.device)
    if isinstance(mask_len_samples, torch.Tensor):
        mask_len_samples = mask_len_samples.to(X.device)
    rng = check_random_state(random_state)
    return torch.as_tensor(rng.uniform(
        low=0, high=1, size=X.shape[0],
    ), device=X.device) * (seq_length - mask_len_samples)


def smooth_time_mask(X: Tensor, y: Tensor,
                     mask_len_samples: int, mask_start_per_sample: Tensor | None = None,
                     random_state: int | None = None) -> (Tensor, Tensor):
    """Smoothly replace a contiguous part of all channels by zeros.

    Originally proposed in [1]_ and [2]_

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    mask_start_per_sample : torch.tensor
        Tensor of integers containing the position (in last dimension) where to
        start masking the signal. Should have the same size as the first
        dimension of X (i.e. one start position per example in the batch).
    mask_len_samples : int
        Number of consecutive samples to zero out.

    random_state: int | numpy.random.Generator, optional

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """
    batch_size, n_channels, seq_len = X.shape
    if mask_start_per_sample is None or not isinstance(mask_start_per_sample, Tensor):
        mask_start_per_sample = _get_mask_start_per_sample(X, mask_len_samples, random_state=random_state)
    t = torch.arange(seq_len, device=X.device).float()
    t = t.repeat(batch_size, n_channels, 1)
    mask_start_per_sample = mask_start_per_sample.view(-1, 1, 1)
    s = 1000 / seq_len
    mask = (torch.sigmoid(s * -(t - mask_start_per_sample)) +
            torch.sigmoid(s * (t - mask_start_per_sample - mask_len_samples))
            ).float().to(X.device)
    return X * mask, y


def _analytic_transform(x):
    if torch.is_complex(x):
        raise ValueError("x must be real.")

    N = x.shape[-1]
    f = fft(x, N, dim=-1)
    h = torch.zeros_like(f)
    if N % 2 == 0:
        h[..., 0] = h[..., N // 2] = 1
        h[..., 1:N // 2] = 2
    else:
        h[..., 0] = 1
        h[..., 1:(N + 1) // 2] = 2

    return ifft(f * h, dim=-1)


def _next_pow2(n):
    """Return the first integer N such that 2**N >= abs(n)."""
    return int(np.ceil(np.log2(np.abs(n))))


def _frequency_shift(X: Tensor, fs: float, f_shift: float | Tensor):
    """
    Shift the specified signal by the specified frequency.

    See https://gist.github.com/lebedov/4428122
    """
    # Pad the signal with zeros to prevent the FFT invoked by the transform
    # from slowing down the computation:
    n_channels, N_orig = X.shape[-2:]
    N_padded = 2 ** _next_pow2(N_orig)
    t = torch.arange(N_padded, device=X.device) / fs
    padded = pad(X, (0, N_padded - N_orig))
    analytical = _analytic_transform(padded)
    if isinstance(f_shift, (float, int, np.ndarray, list)):
        f_shift = torch.as_tensor(f_shift).float()
    f_shift_stack = f_shift.repeat(N_padded, n_channels, 1)
    reshaped_f_shift = f_shift_stack.permute(
        *torch.arange(f_shift_stack.ndim - 1, -1, -1))
    shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
    return shifted[..., :N_orig].real.float()


def frequency_shift(X: Tensor, y: Tensor, max_delta_freq: float,
                    sfreq: float, random_state: int | None = None) -> (Tensor, Tensor):
    """Adds a shift in the frequency domain to all channels.

    Note that here, the shift is the same for all channels of a single example.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    max_delta_freq : float
        The max amplitude of the frequency shift (in Hz).
    sfreq : float
        Sampling frequency of the signals to be transformed.

    random_state: int | numpy.random.Generator, optional

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    rng = check_random_state(random_state)
    u = as_tensor(
        rng.uniform(size=X.shape[0]),
        device=X.device
    )
    if isinstance(max_delta_freq, Tensor):
        max_delta_freq = max_delta_freq.to(X.device)
    delta_freq = u * 2 * max_delta_freq - max_delta_freq

    transformed_X = _frequency_shift(
        X=X,
        fs=sfreq,
        f_shift=delta_freq,
    )
    transformed_X = transformed_X.real.type(X.dtype)
    return transformed_X, y


def crop_to_seconds(X: Tensor | ndarray,
                    sfreq: float,
                    in_t_min: float, in_t_max: float,
                    out_t_min: float, out_t_max: float,
                    n_times: int) -> Tensor:
    start_index = int(abs(out_t_min - in_t_min) * sfreq)
    end_index = X.shape[-1] - int((in_t_max - out_t_max) * sfreq)
    new_n_times = end_index - start_index
    if new_n_times > n_times:
        start_index = start_index + (new_n_times - n_times) // 2
        end_index = start_index + n_times
    elif new_n_times < n_times:
        pad_len = n_times - new_n_times
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        start_index = start_index - pad_left
        end_index = end_index + pad_right
    # Cut the signal
    if len(X.shape) == 2:
        return X[:, start_index:end_index]
    else:
        return X[:, :, start_index:end_index]


def time_shift(X: Tensor,
               y: Tensor,
               max_delta_time: float,
               sfreq: float,
               delta_times: Tensor | ndarray | None = None,
               random_state: int | None = None,
               target_t_min: float = -0.2, target_t_max: float = 1.0,
               is_t_min: float = -1.5, is_t_max: float = 1.5,
               n_times: int = 155) -> (Tensor, Tensor):
    rng = check_random_state(random_state)
    max_delta_time = max_delta_time

    if delta_times is None:
        delta_times = as_tensor(
            rng.uniform(low=-max_delta_time, high=max_delta_time, size=X.shape[0]),
            device=X.device
        )

    new_min = target_t_min + delta_times
    new_max = target_t_max + delta_times

    transformed = []
    for i in range(X.shape[0]):
        transformed.append(crop_to_seconds(X[i], sfreq, is_t_min, is_t_max, new_min[i], new_max[i], n_times))

    assert all([t.shape[-1] == n_times for t in transformed]), "The transformed signals must have the same length."
    transformed_X = torch.stack(transformed, dim=0)
    return transformed_X, y

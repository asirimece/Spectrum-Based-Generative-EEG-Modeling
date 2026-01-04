from ._model import Transform, TransformType, TransformStep
from ._functional import channels_dropout, ft_surrogate, smooth_time_mask, frequency_shift
from enum import Enum
from typing import List, Tuple
from numbers import Real
from lib.config import DictInit
from torch import Tensor, as_tensor


class Transforms(Enum):
    """Enum class for transform types"""
    CHANNELS_DROPOUT = "channels_dropout"
    FT_SURROGATE = "ft_surrogate"
    SMOOTH_TIME_MASK = "smooth_time_mask"
    FREQUENCY_SHIFT = "frequency_shift"
    TIME_SHIFT = "time_shift"
    GEN_GAN = "gen_gan"
    GEN_DIFFUSION = "gen_diffusion"
    TIME_CROP = "time_crop"


class ChannelsDropout(Transform):
    """Randomly set channels to flat signal.

    Part of the CMSAugment policy proposed in [1]_

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    p_drop: float | None, optional
        Float between 0 and 1 setting the probability of dropping each channel.
        Defaults to 0.2.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether to transform given the probability
        argument and to sample channels to erase. Defaults to None.

    References
    ----------
    .. [1] Saeed, A., Grangier, D., Pietquin, O., & Zeghidour, N. (2020).
       Learning from Heterogeneous EEG Signals with Differentiable Channel
       Reordering. arXiv preprint arXiv:2010.13694.
    """

    def __init__(
            self,
            probability: float = 0.5,
            p_drop: float = 0.2,
            transform_type: TransformType | str = TransformType.IN_PLACE,
            random_state: int | None = None,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.CHANNELS_DROPOUT.value,
            operation=staticmethod(channels_dropout),
            probability=probability,
            transform_type=transform_type,
            transform_step=TransformStep.AUGMENT,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )
        self.p_drop = p_drop

    def get_augmentation_params(self, *batch):
        """Return transform parameters.

        Parameters
        ----------
        X : tensor.Tensor
            The data.
        y : tensor.Tensor
            The labels.

        Returns
        -------
        params : dict
            Contains

            * p_drop : float
                Float between 0 and 1 setting the probability of dropping each
                channel.
            * random_state : numpy.random.Generator
                The generator to use.
        """
        return {
            "p_drop": self.p_drop,
            "random_state": self.rng,
        }


class FTSurrogate(Transform):
    """FT surrogate augmentation of a single EEG channel, as proposed in [1]_.

    Parameters
    ----------
    probability: float
        Float setting the probability of applying the operation.
    phase_noise_magnitude : float | torch.Tensor, optional
        Float between 0 and 1 setting the range over which the phase
        perturbation is uniformly sampled:
        ``[0, phase_noise_magnitude * 2 * pi]``. Defaults to 1.
    channel_independence : bool, optional
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI. Default False.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether to transform given the probability
        argument. Defaults to None.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """

    def __init__(
            self,
            probability: float = 0.5,
            phase_noise_magnitude: float | Tensor = 1,
            channel_independence: bool = False,
            transform_type: TransformType | str = TransformType.IN_PLACE,
            random_state: int | None = None,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.FT_SURROGATE.value,
            operation=staticmethod(ft_surrogate),
            probability=probability,
            transform_type=transform_type,
            transform_step=TransformStep.AUGMENT,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )
        assert isinstance(phase_noise_magnitude, (float, int, Tensor)), \
            "phase_noise_magnitude should be a float."
        assert 0 <= phase_noise_magnitude <= 1, \
            "phase_noise_magnitude should be between 0 and 1."
        assert isinstance(channel_independence, bool), (
            "channel_independence is expected to be a boolean")
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_independence = channel_independence

    def get_augmentation_params(self, *batch):
        """Return transform parameters."""
        return {
            "phase_noise_magnitude": self.phase_noise_magnitude,
            "channel_independence": self.channel_independence,
            "random_state": self.rng,
        }


class SmoothTimeMask(Transform):
    """Smoothly replace a randomly chosen contiguous part of all channels by
    zeros.

    Suggested e.g. in [1]_ and [2]_

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    mask_len_samples : int | torch.Tensor, optional
        Number of consecutive samples to zero out. Will be ignored if
        magnitude is not set to None. Defaults to 100.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.

    References
    ----------
    .. [1] Cheng, J. Y., Goh, H., Dogrusoz, K., Tuzel, O., & Azemi, E. (2020).
       Subject-aware contrastive learning for biosignals. arXiv preprint
       arXiv:2007.04871.
    .. [2] Mohsenvand, M. N., Izadi, M. R., & Maes, P. (2020). Contrastive
       Representation Learning for Electroencephalogram Classification. In
       Machine Learning for Health (pp. 238-253). PMLR.
    """

    def __init__(
            self,
            probability: float = 0.5,
            mask_len_samples: int = 100,
            transform_type: TransformType | str = TransformType.IN_PLACE,
            random_state: int | None = None,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.SMOOTH_TIME_MASK.value,
            operation=staticmethod(smooth_time_mask),
            probability=probability,
            transform_type=transform_type,
            transform_step=TransformStep.AUGMENT,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )

        assert (
                isinstance(mask_len_samples, (int, Tensor)) and
                mask_len_samples > 0
        ), "mask_len_samples has to be a positive integer"
        self.mask_len_samples = mask_len_samples

    def get_augmentation_params(self, *batch):
        """Return transform parameters."""

        return {
            "mask_len_samples": self.mask_len_samples
        }


class FrequencyShift(Transform):
    """Add a random shift in the frequency domain to all channels.

    Note that here, the shift is the same for all channels of a single example.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    sfreq : float
        Sampling frequency of the signals to be transformed.
    max_delta_freq : float | torch.Tensor, optional
        Maximum shift in Hz that can be sampled (in absolute value).
        Defaults to 2 (shift sampled between -2 and 2 Hz).
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Defaults to None.
    """

    def __init__(
            self,
            probability: float = 0.5,
            sfreq: float = 128,
            max_delta_freq: float | Tensor = 2.,
            random_state: int | None = None,
            transform_type: TransformType | str = TransformType.IN_PLACE,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.FREQUENCY_SHIFT.value,
            probability=probability,
            operation=staticmethod(frequency_shift),
            transform_type=transform_type,
            transform_step=TransformStep.AUGMENT,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )
        assert isinstance(sfreq, Real) and sfreq > 0, \
            "sfreq should be a positive float."
        self.sfreq = sfreq

        self.max_delta_freq = max_delta_freq

    def get_augmentation_params(self, *batch):
        if len(batch) == 0:
            return super().get_augmentation_params(*batch)
        X = batch[0]

        u = as_tensor(
            self.rng.uniform(size=X.shape[0]),
            device=X.device
        )
        max_delta_freq = self.max_delta_freq
        if isinstance(max_delta_freq, Tensor):
            max_delta_freq = max_delta_freq.to(X.device)
        delta_freq = u * 2 * max_delta_freq - max_delta_freq
        return {
            "delta_freq": delta_freq,
            "sfreq": self.sfreq,
        }


def get_transform_by_name(name: str, *args, **kwargs) -> Transform:
    match name:
        case Transforms.CHANNELS_DROPOUT.value:
            return ChannelsDropout(*args, **kwargs)
        case Transforms.FT_SURROGATE.value:
            return FTSurrogate(*args, **kwargs)
        case Transforms.SMOOTH_TIME_MASK.value:
            return SmoothTimeMask(*args, **kwargs)
        case Transforms.FREQUENCY_SHIFT.value:
            return FrequencyShift(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown transform name: {name}")


def get_transforms(configs: List[DictInit] | None) -> List[Transform]:
    if configs is None:
        print("DEBUG: configs is None before processing in get_transforms.")
        return []
    elif not configs:
        print("DEBUG: configs is an empty list before processing in get_transforms.")
        return []
    else:
        print(f"DEBUG: configs has {len(configs)} items before processing in get_transforms.")
    return [get_transform_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]

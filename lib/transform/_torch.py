from ._model import TorchTransform, TransformType
from ._functional import channels_dropout, ft_surrogate, smooth_time_mask
from typing import List
from lib.config import DictInit
from torch import Tensor
from ._transforms import Transforms


class TorchChannelsDropout(TorchTransform):
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
            random_state: int | None = None,
            classes: List[int] | None = None
    ):
        super().__init__(
            name=Transforms.CHANNELS_DROPOUT.value,
            operation=staticmethod(channels_dropout),
            probability=probability,
            random_state=random_state,
            classes=classes
        )
        self.p_drop = p_drop

    def get_augmentation_params(self, *batch):
        """Return transform parameters."""

        return {
            "p_drop": self.p_drop,
            "random_state": self.rng,
        }


class TorchFTSurrogate(TorchTransform):
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
            random_state: int | None = None,
            classes: List[int] | None = None
    ):
        super().__init__(
            name=Transforms.FT_SURROGATE.value,
            operation=staticmethod(ft_surrogate),
            probability=probability,
            random_state=random_state,
            classes=classes
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


class TorchSmoothTimeMask(TorchTransform):
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
            random_state: int | None = None,
            classes: List[int] | None = None
    ):
        super().__init__(
            name=Transforms.SMOOTH_TIME_MASK.value,
            operation=staticmethod(smooth_time_mask),
            probability=probability,
            random_state=random_state,
            classes=classes
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


def get_torch_transform_by_name(name: str, *args, **kwargs) -> TorchTransform:
    match name:
        case Transforms.CHANNELS_DROPOUT.value:
            return TorchChannelsDropout(*args, **kwargs)
        case Transforms.FT_SURROGATE.value:
            return TorchFTSurrogate(*args, **kwargs)
        case Transforms.SMOOTH_TIME_MASK.value:
            return TorchSmoothTimeMask(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown epochs transform name: {name}")


def get_torch_transforms(configs: List[DictInit]) -> List[TorchTransform]:
    return [get_torch_transform_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]

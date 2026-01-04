from mne import Epochs, EpochsArray
import numpy as np
from numpy import ndarray
from ._model import (MneEpochsTransform, TransformType, TransformStep, MneGenEpochsTransform,
                     MneAugmentationEpochsTransform, TransformResult, Transform)
from ._functional import channels_dropout, ft_surrogate, smooth_time_mask, frequency_shift, time_shift, crop_to_seconds
from typing import List, Dict
from numbers import Real
from lib.config import DictInit
from torch import Tensor, as_tensor
from ._transforms import Transforms
from torch import Tensor
from lib.preprocess import RawPreprocessor, get_raw_preprocessors
import torch


class EpochsChannelsDropout(MneAugmentationEpochsTransform):
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
        """Return transform parameters."""

        return {
            "p_drop": self.p_drop,
            "random_state": self.rng,
        }


class EpochsFTSurrogate(MneAugmentationEpochsTransform):
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


class EpochsSmoothTimeMask(MneAugmentationEpochsTransform):
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
            "mask_len_samples": self.mask_len_samples,
            "random_state": self._random_state,
        }


class EpochsFrequencyShift(MneAugmentationEpochsTransform):
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
            transform_type: TransformType | str = TransformType.IN_PLACE,
            max_delta_freq: float | Tensor = 2.,
            random_state: int | None = None,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.FREQUENCY_SHIFT.value,
            probability=probability,
            transform_type=transform_type,
            operation=staticmethod(frequency_shift),
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
            "max_delta_freq": self.max_delta_freq,
            "sfreq": self.sfreq,
            "random_state": self._random_state,
        }


class EpochsTimeShift(MneAugmentationEpochsTransform):
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
            transform_type: TransformType | str = TransformType.IN_PLACE,
            max_delta_time: float | Tensor = 0.5,
            target_t_min: float = -0.2,
            target_t_max: float = 1.0,
            is_t_min: float = -1.5,
            is_t_max: float = 1.5,
            n_times: int = 155,
            sfreq: float = 128,
            random_state: int | None = None,
            classes: List[int] | None = None,
            shuffle: bool = False
    ):
        super().__init__(
            name=Transforms.TIME_SHIFT.value,
            probability=probability,
            transform_type=transform_type,
            operation=staticmethod(time_shift),
            transform_step=TransformStep.AUGMENT,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )
        self.max_delta_time = max_delta_time
        self.sfreq = sfreq
        self.target_t_min = target_t_min
        self.target_t_max = target_t_max
        self.is_t_min = is_t_min
        self.is_t_max = is_t_max
        self.n_times = n_times

        def time_shift_remainder(X: ndarray | Tensor):
            X, y, input_type = Transform.check_type(X)
            X_out = crop_to_seconds(X=X, sfreq=self.sfreq, in_t_min=self.is_t_min, in_t_max=self.is_t_max,
                                    out_t_min=self.target_t_min, out_t_max=self.target_t_max, n_times=self.n_times)
            X_out, y, _ = Transform.check_type(X_out, input_type=input_type)
            return X_out

        self.remainder_operation = time_shift_remainder

    def transform(self, X: Tensor | ndarray, y: Tensor | ndarray | None = None) -> TransformResult:
        X, y, input_type = Transform.check_type(X, y)
        params = self.get_augmentation_params(X, y)
        transformed, y = self._operation(X, y, **params)
        transformed, y, _ = Transform.check_type(transformed, y, input_type)
        return transformed

    def get_augmentation_params(self, *batch):

        if len(batch) == 0:
            return super().get_augmentation_params(*batch)
        X = batch[0]
        max_delta_time = self.max_delta_time
        delta_times = as_tensor(
            self.rng.uniform(low=-max_delta_time, high=max_delta_time, size=X.shape[0]),
            device=X.device
        )

        return {
            "max_delta_time": self.max_delta_time,
            "delta_times": delta_times,
            "sfreq": self.sfreq,
            "target_t_min": self.target_t_min,
            "target_t_max": self.target_t_max,
            "is_t_min": self.is_t_min,
            "is_t_max": self.is_t_max,
            "n_times": self.n_times,
            "random_state": self._random_state,
        }


class EpochsGanTransform(MneGenEpochsTransform):
    z_dim: int

    def __init__(self,
                 model_path: str,
                 fraction: float | str = 1,
                 probability: float = 1,
                 z_dim: int = 100,
                 preprocessors: List[RawPreprocessor] | List[DictInit] | None = None,
                 transform_step: TransformStep | str = TransformStep.AUGMENT,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 shuffle: bool = False):
        super().__init__(
            name=Transforms.GEN_GAN.value,
            fraction=fraction,
            probability=probability,
            model_path=model_path,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle,
            preprocessors=preprocessors
        )
        self.z_dim = z_dim

    def gen_epochs(self, epochs: Epochs | EpochsArray, gen_labels: ndarray) -> Epochs | EpochsArray:
        self.model.eval()
        self.model.to(self.device)
        generated = self.model.sample(as_tensor(gen_labels, device=self.device, dtype=torch.int)).detach().cpu()
        generated = generated.squeeze().numpy()
        n_times = epochs[0].get_data(copy=True).shape[-1]

        if generated.shape[0] > len(gen_labels):
            idx = np.random.choice(generated.shape[0], len(gen_labels), replace=False)
            generated = generated[idx]

        events = np.column_stack((np.arange(len(generated)) * n_times + int(np.abs(epochs.tmin) * n_times),
                                  np.zeros(len(generated), dtype=int), gen_labels))
        epochs = EpochsArray(generated, epochs.info, events=events, tmin=epochs.tmin)
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                epochs = preprocessor.transform(epochs)
        return epochs


class EpochsDiffusionTransform(MneGenEpochsTransform):
    z_dim: int

    def __init__(self,
                 model_path: str,
                 fraction: float | str = 1,
                 probability: float = 1,
                 preprocessors: List[RawPreprocessor] | List[DictInit] | None = None,
                 transform_step: TransformStep | str = TransformStep.AUGMENT,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 shuffle: bool = False):
        super().__init__(
            name=Transforms.GEN_DIFFUSION.value,
            fraction=fraction,
            probability=probability,
            model_path=model_path,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle,
            preprocessors=preprocessors
        )

    def gen_epochs(self, epochs: Epochs | EpochsArray, gen_labels: ndarray) -> Epochs | EpochsArray:
        self.model.eval_mode()
        self.model.to(self.device)
        generated = self.model(as_tensor(gen_labels, device=self.device, dtype=torch.int)).detach().cpu()
        generated = generated.squeeze().numpy()
        n_times = epochs[0].get_data(copy=True).shape[-1]

        if generated.shape[0] > len(gen_labels):
            idx = np.random.choice(generated.shape[0], len(gen_labels), replace=False)
            generated = generated[idx]

        events = np.column_stack((np.arange(len(generated)) * n_times + int(np.abs(epochs.tmin) * n_times),
                                  np.zeros(len(generated), dtype=int), gen_labels))
        epochs = EpochsArray(generated, epochs.info, events=events, tmin=epochs.tmin)
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                epochs = preprocessor.transform(epochs)
        return epochs


class TimeCropTransform(MneEpochsTransform):

    def __init__(self,
                 transform_step: TransformStep | str = TransformStep.POST_AUGMENT,
                 in_t_min: float = -1.5,
                 in_t_max: float = 1.5,
                 out_t_min: float = -0.2,
                 out_t_max: float = 1.0,
                 n_times: int = 155,
                 sfreq: float = 128,
                 random_state: int | None = None,
                 classes: List[int] | None = None):
        super().__init__(
            name=Transforms.TIME_CROP.value,
            transform_type=TransformType.IN_PLACE,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
        )

        self.target_t_min = out_t_min
        self.target_t_max = out_t_max
        self.is_t_min = in_t_min
        self.is_t_max = in_t_max
        self.n_times = n_times
        self.sfreq = sfreq

    def transform_epochs(self, epochs: Epochs | EpochsArray) -> Epochs | EpochsArray:
        data = epochs.get_data(copy=True)

        if data.shape[-1] != self.n_times:
            X, y, input_type = Transform.check_type(data)
            X_out = crop_to_seconds(X=X, sfreq=self.sfreq, in_t_min=self.is_t_min, in_t_max=self.is_t_max,
                                    out_t_min=self.target_t_min, out_t_max=self.target_t_max, n_times=self.n_times)
            X_out, y, _ = Transform.check_type(X_out, input_type=input_type)
            epochs = EpochsArray(X_out, info=epochs.info, tmin=epochs.tmin, events=epochs.events)

        return epochs


def get_epochs_transform_by_name(name: str, *args, **kwargs) -> MneEpochsTransform:
    match name:
        case Transforms.CHANNELS_DROPOUT.value:
            return EpochsChannelsDropout(*args, **kwargs)
        case Transforms.FT_SURROGATE.value:
            return EpochsFTSurrogate(*args, **kwargs)
        case Transforms.SMOOTH_TIME_MASK.value:
            return EpochsSmoothTimeMask(*args, **kwargs)
        case Transforms.FREQUENCY_SHIFT.value:
            return EpochsFrequencyShift(*args, **kwargs)
        case Transforms.TIME_SHIFT.value:
            return EpochsTimeShift(*args, **kwargs)
        case Transforms.GEN_GAN.value:
            return EpochsGanTransform(*args, **kwargs)
        case Transforms.GEN_DIFFUSION.value:
            return EpochsDiffusionTransform(*args, **kwargs)
        case Transforms.TIME_CROP.value:
            return TimeCropTransform(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown epochs transform name: {name}")


def get_epochs_transforms(configs: List[DictInit]) -> List[MneEpochsTransform]:
    return [get_epochs_transform_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]

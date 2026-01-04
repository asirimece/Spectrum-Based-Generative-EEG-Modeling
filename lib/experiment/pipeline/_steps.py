from enum import Enum
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyriemann.classification import MDM, SVC as R_SVC, FgMDM
from pyriemann.estimation import XdawnCovariances
from sklearn.decomposition import PCA
from omegaconf import OmegaConf, ListConfig
from lib.dataset import BaseDataset
from lib.config import DictInit
from typing import List, Callable, Dict, Tuple, Union
from braindecode import EEGClassifier
from skorch.dataset import ValidSplit
from lib.callback.skorch import get_callbacks
from numpy import ndarray
import numpy as np
import torchvision
import torch
import copy
from torch.nn import functional as F
from mne.decoding import (
    Scaler,
    Vectorizer
)

from lib.transform._augment import SpectrogramAugmenter
from lib.transform._spectrogram import STFTComputer
from lib.transform._stack import SpectrogramStacker


class PipelineSteps(Enum):
    MULTIPLIER = "multiplier"
    SCALER = "scaler"
    CUSTOM_SCALER = "custom_scaler"
    STANDARD_SCALER = "standard_scaler"
    MIN_MAX_SCALER = "min_max_scaler"
    VECTORIZER = "vectorizer"
    NOISER = "noiser"
    EPOCH_SPLITTER = "epoch_splitter"
    CHANNELS_REPLICATOR = "channels_replicator"
    PADDING = "padding"
    
    SPECTROGRAM_COMPUTER = "spectrogram_computer"
    SPECTROGRAM_AUGMENTER = "spectrogram_augmenter"
    SPECTROGRAM_STACKER = "spectrogram_stacker"
        
    PCA = "pca"
    XDAWN = "xdawn"

    LDA = "lda"
    SVM = "svm"
    R_SVM = "r_svm"
    SHALLOW_FBCSP_NET = "shallow_fbcsp_net"
    EEG_NET = "eeg_net"
    INCEPTION = "inception"
    MDM = "mdm" 
    FG_MDM = "fg_mdm"


def get_named_optimizer(optimizer: str) -> Callable:
    match optimizer.lower():
        case 'adamw':
            return torch.optim.AdamW
        case 'adam':
            return torch.optim.Adam
        case 'rmsprop':
            return torch.optim.RMSprop
        case 'sgd':
            return torch.optim.SGD
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer}")


def get_optimizer(config: Dict | str | None = None) -> Tuple[Union[Callable, None], Dict]:
    kwargs = {}
    if isinstance(config, str):
        optimizer = get_named_optimizer(config)
    elif isinstance(config, Dict) and 'name' in config:
        optimizer = get_named_optimizer(config['name'])
        if 'kwargs' in config:
            old_kwargs = config['kwargs']
            for key, value in old_kwargs.items():
                if not key.startswith('optimizer__'):
                    kwargs[f'optimizer__{key}'] = value
                else:
                    kwargs[key] = value
    else:
        return get_named_optimizer('sgd'), kwargs

    return optimizer, kwargs


def get_optimizer_from_kwargs(kwargs: Dict) -> Tuple[Callable, Dict]:
    optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else 'sgd'
    return get_optimizer(optimizer)


class StepBuilder:

    @staticmethod
    def build(*args, **kwargs):
        raise NotImplementedError


class ShallowFBCSPNetBuilder(StepBuilder):

    @staticmethod
    def __get_train_split(kwargs: Dict | None = None):
        if kwargs is None:
            kwargs = {}
        if 'experiment_type' in kwargs and kwargs['experiment_type'] == 'cross_subject':
            return None
        return ValidSplit(kwargs['valid_split'] if 'valid_split' in kwargs else 0.2,
                          random_state=kwargs['random_state'] if 'random_state' in kwargs else 1)

    @staticmethod
    def __get_nonlin_by_str(nonlin: str) -> Callable:
        if nonlin == 'relu':
            return lambda x: F.relu(x, inplace=False)
        if nonlin == 'square':
            return lambda x: x * x
        if nonlin == 'identity':
            return lambda x: x
        raise ValueError(f"Unknown nonlin: {nonlin}")

    @staticmethod
    def __get_criteria_by_str(criterion: str) -> Callable:
        if criterion == 'cross_entropy':
            return torch.nn.CrossEntropyLoss
        raise ValueError(f"Unknown criterion: {criterion}")

    @staticmethod
    def __get_optimizer(kwargs: Dict | None = None):
        if kwargs is None:
            kwargs = {}
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else 'adamW'
        match optimizer:
            case 'adamW':
                return torch.optim.AdamW
            case 'adam':
                return torch.optim.Adam
            case 'rmsProp':
                return torch.optim.RMSprop
            case 'sgd':
                return torch.optim.SGD
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer}")

    @staticmethod
    def build(*args, **kwargs):
        nonlin = kwargs['conv_nonlin'] if 'conv_nonlin' in kwargs else 'square'
        criterion = kwargs['criterion'] if 'criterion' in kwargs else 'cross_entropy'

        return EEGClassifier(
            'ShallowFBCSPNet',
            module__final_conv_length='auto',
            module__conv_nonlin=ShallowFBCSPNetBuilder.__get_nonlin_by_str(nonlin),
            module__n_filters_time=kwargs['n_filters_time'] if 'n_filters_time' in kwargs else 40,
            module__n_filters_spat=kwargs['n_filters_spat'] if 'n_filters_spat' in kwargs else 40,
            criterion=ShallowFBCSPNetBuilder.__get_criteria_by_str(criterion),
            optimizer=ShallowFBCSPNetBuilder.__get_optimizer(kwargs),
            optimizer__lr=kwargs['lr'] if 'lr' in kwargs else 0.0625 * 0.01,
            optimizer__weight_decay=kwargs['weight_decay'] if 'weight_decay' in kwargs else 0,
            train_split=ShallowFBCSPNetBuilder.__get_train_split(kwargs),
            max_epochs=kwargs['max_epochs'] if 'max_epochs' in kwargs else 10,
            callbacks=get_callbacks(kwargs['callbacks']) if 'callbacks' in kwargs else None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            # 20% validation split
        )


class InceptionNetBuilder(StepBuilder):

    @staticmethod
    def __get_train_split(kwargs: Dict | None = None):
        if kwargs is None:
            kwargs = {}
        if 'experiment_type' in kwargs and kwargs['experiment_type'] == 'cross_subject':
            return None
        return ValidSplit(kwargs['valid_split'] if 'valid_split' in kwargs else 0.2,
                          random_state=kwargs['random_state'] if 'random_state' in kwargs else 1)

    @staticmethod
    def build(*args, **kwargs):
        model = copy.deepcopy(torchvision.models.inception_v3(pretrained=False, progress=True))
        class_weights = kwargs['class_weights'] if 'class_weights' in kwargs else [1] * kwargs['n_classes']
        return EEGClassifier(
            model,
            module__dropout=kwargs['dropout'] if 'drop_prob' in kwargs else 0.5,
            module__num_classes=kwargs['n_classes'] if 'n_classes' in kwargs else 2,
            module__aux_logits=False,
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)),
            train_split=InceptionNetBuilder.__get_train_split(kwargs),
            batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else 32,
            max_epochs=kwargs['max_epochs'] if 'max_epochs' in kwargs else 10,
            callbacks=get_callbacks(kwargs['callbacks']) if 'callbacks' in kwargs else None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )


class EEGNetBuilder(StepBuilder):

    @staticmethod
    def __get_train_split(kwargs: Dict | None = None):
        if kwargs is None:
            kwargs = {}
        if 'experiment_type' in kwargs and kwargs['experiment_type'] == 'cross_subject':
            return None
        return ValidSplit(kwargs['valid_split'] if 'valid_split' in kwargs else 0.2,
                          random_state=kwargs['random_state'] if 'random_state' in kwargs else 1)

    @staticmethod
    def build(*args, **kwargs):

        class_weights = kwargs['class_weights'] if 'class_weights' in kwargs else [1] * kwargs['n_classes']
        optimizer, optimizer_kwargs = get_optimizer_from_kwargs(kwargs)

        classifier = EEGClassifier(
            'EEGNetv4',
            module__final_conv_length='auto',
            module__n_chans=kwargs['n_channels'] if 'n_channels' in kwargs else 32,
            module__n_outputs=kwargs['n_classes'] if 'n_classes' in kwargs else 2,
            module__n_times=kwargs['n_times'] if 'n_times' in kwargs else 155,
            module__drop_prob=kwargs['drop_prob'] if 'drop_prob' in kwargs else 0.5,
            # module__F1=16,
            # module__D=4,
            # module__F2=64,  # usually set to F1*D (?)

            # Investigated to use kernel_length of 32 for Zhang dataset as suggested in the paper but 64 works better.
            # module__kernel_length=int(kwargs['sfreq'] / 2) if 'sfreq' in kwargs else 64,
            module__kernel_length=64,
            train_split=EEGNetBuilder.__get_train_split(kwargs),
            optimizer=optimizer,
            **optimizer_kwargs,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)),
            batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else 32,
            max_epochs=kwargs['max_epochs'] if 'max_epochs' in kwargs else 10,
            callbacks=get_callbacks(kwargs['callbacks']) if 'callbacks' in kwargs else None,
            # To train a neural network you need validation split, here, we use 20%.
        )
        return classifier


class Multiplier(TransformerMixin, BaseEstimator):
    factor: float

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.multiply(X, self.factor)

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y)

    def inverse_transform(self, X, y=None):
        return np.divide(X, self.factor)


class EpochSplitter(BaseEstimator, TransformerMixin):

    def __init__(self, extractor: any, window_size_ms: int = 100, sfreq: int = 250):
        self.extractor = self.build_extractor(extractor)
        self.window_size_ms = window_size_ms
        self.sfreq = sfreq
        self.window_size_samples = self.window_size_ms * self.sfreq // 1000

    def build_extractor(self, extractor: any):
        if isinstance(extractor, dict):
            assert 'name' in extractor, "Extractor must have a name to resolve."
            steps = []
            if 'scaling' in extractor and extractor['scaling'] is not None:
                steps.append(('extractor_scaler', get_step_by_name(extractor['scaling'])))
            extractor_name = extractor['name']
            args = extractor['args'] if 'args' in extractor else []
            kwargs = extractor['kwargs'] if 'kwargs' in extractor else {}
            steps.append(('extractor', get_step_by_name(extractor_name, *args, **kwargs)))
            return Pipeline(steps=steps)
        return extractor

    def split(self, X, y=None):
        assert X.ndim == 3, (f"Expected 3D data with shape (n_epochs, n_channels, n_times) "
                             f"for step {self.__class__.__name__}")
        X = X.copy()
        n_epochs, n_channels, n_times = X.shape
        n_windows = n_times // self.window_size_samples
        X = X[:, :, :n_windows * self.window_size_samples]

        # Reshape the data to split into windows
        data_reshaped = X.reshape(n_epochs, n_channels, n_windows, self.window_size_samples)

        # Transpose to match the desired shape (batch_size, n_windows, n_channels, n_times)
        data_transposed = data_reshaped.transpose(0, 2, 1, 3)
        return data_transposed

    def preprocess(self, X):
        X = X.copy()
        X = self.split(X)
        X = X.reshape(X.shape[0] * X.shape[1], -1)
        return X

    def fit(self, X, y=None):
        X = self.preprocess(X)
        self.extractor = self.extractor.fit(X)
        return self

    def transform(self, X):
        input = self.preprocess(X)
        out = self.extractor.transform(input)
        out = out.reshape(X.shape[0], -1)
        return out

    def fit_transform(self, X, y=None, **fit_params):
        input = self.preprocess(X)
        out = self.extractor.fit_transform(input, **fit_params)
        out = out.reshape(X.shape[0], -1)
        return out


class ChannelsReplicator(BaseEstimator, TransformerMixin):
    def __init__(self, n_replicas: int = 3, axis: int = 1):
        self.n_replicas = n_replicas
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x_transformed = X.copy()
        x_transformed = np.repeat(np.expand_dims(x_transformed, 1), self.n_replicas, 1)
        return x_transformed


class Padding(BaseEstimator, TransformerMixin):
    def __init__(self, desired_channels: int = 64, desired_times: int = 256):
        self.desired_channels = desired_channels
        self.desired_times = desired_times

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x_transformed = X.copy()
        channel_diff = max(self.desired_channels - x_transformed.shape[1], 0)
        time_diff = max(self.desired_times - x_transformed.shape[2], 0)
        x_transformed = np.pad(x_transformed, ((0, 0), (0, channel_diff), (0, time_diff)), constant_values=0)
        return x_transformed


def _sklearn_reshape_apply(func, return_result, X, *args, **kwargs):
    """Reshape epochs and apply function."""
    if not isinstance(X, np.ndarray):
        raise ValueError("data should be an np.ndarray, got %s." % type(X))
    orig_shape = X.shape
    X = np.reshape(X.transpose(0, 2, 1), (-1, orig_shape[1]))
    X = func(X, *args, **kwargs)
    if return_result:
        X.shape = (orig_shape[0], orig_shape[2], orig_shape[1])
        X = X.transpose(0, 2, 1)
        return X


class CustomScaler(TransformerMixin, BaseEstimator):
    """Standardize channel data.

    This class scales data for each channel. It differs from scikit-learn
    classes (e.g., :class:`sklearn.preprocessing.StandardScaler`) in that
    it scales each *channel* by estimating μ and σ using data from all
    time points and epochs, as opposed to standardizing each *feature*
    (i.e., each time point for each channel) by estimating using μ and σ
    using data from all epochs.

    Parameters
    ----------
    scalings : str, default None
        Scaling method to be applied to data channel wise.
        * if ``scalings=='median'``,
          :class:`sklearn.preprocessing.RobustScaler`
          is used (requires sklearn version 0.17+).
        * if ``scalings=='mean'``,
          :class:`sklearn.preprocessing.StandardScaler`
          is used.
        * if ``scalings=='minmax'``,
         :class:`sklearn.preprocessing.MinMaxScaler`
         is used.>

    with_mean : bool, default True
        If True, center the data using mean (or median) before scaling.
        Ignored for channel-type scaling.
    with_std : bool, default True
        If True, scale the data to unit variance (``scalings='mean'``),
        quantile range (``scalings='median``), or using channel type
        if ``scalings`` is a dict or None).
    feature_range : tuple, default (0, 1)
        Desired range of transformed data after scaling when using minmax scaling.
    """

    def __init__(self, scalings: str = 'mean', with_mean: bool = True, with_std: bool = True,
                 feature_range: tuple = (0, 1)):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scalings = scalings
        self.feature_range = feature_range

        if scalings is None:
            raise ValueError(
                'Need to specify "scalings" if scalings is' "%s" % type(scalings)
            )
        if isinstance(scalings, str):
            assert scalings in ("mean", "median", "minmax"), ("scalings should be mean (StandardScaler) or median "
                                                              "(RobustScaler) or minmax (MinMaxScaler)")
        if scalings == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            self._scaler = MinMaxScaler(feature_range=feature_range)
        elif scalings == "mean":
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler(
                with_mean=self.with_mean, with_std=self.with_std
            )
        else:  # scalings == 'median':
            from sklearn.preprocessing import RobustScaler

            self._scaler = RobustScaler(
                with_centering=self.with_mean, with_scaling=self.with_std
            )

    def fit(self, epochs_data: ndarray, y: ndarray | None = None):
        assert isinstance(epochs_data, np.ndarray), f"Data must be of type np.ndarray, got {type(epochs_data)}"
        if epochs_data.ndim == 2:
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        _sklearn_reshape_apply(self._scaler.fit, False, epochs_data, y=y)
        return self

    def transform(self, epochs_data: ndarray):
        assert isinstance(epochs_data, np.ndarray), f"Data must be of type np.ndarray, got {type(epochs_data)}"
        if epochs_data.ndim == 2:  # can happen with SlidingEstimator
            epochs_data = epochs_data[..., np.newaxis]
        assert epochs_data.ndim == 3, epochs_data.shape
        return _sklearn_reshape_apply(self._scaler.transform, True, epochs_data)

    def fit_transform(self, epochs_data: ndarray, y: ndarray | None = None, **fit_params):
        return self.fit(epochs_data, y).transform(epochs_data)

    def inverse_transform(self, epochs_data: ndarray):
        assert epochs_data.ndim == 3, epochs_data.shape
        return _sklearn_reshape_apply(self._scaler.inverse_transform, True, epochs_data)


class Noiser(BaseEstimator, TransformerMixin):

    def __init__(self, mean: float = 0.0, sigma: float = 1e-4):
        self.mean = mean
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X + np.random.normal(self.mean, self.sigma, X.shape)


def get_step_by_name(name: str, *args, **kwargs):
    match name:
        case PipelineSteps.MULTIPLIER.value:
            return Multiplier(*args, **kwargs)
        case PipelineSteps.SCALER.value:
            return Scaler(*args, **kwargs)
        case PipelineSteps.CUSTOM_SCALER.value:
            return CustomScaler(*args, **kwargs)
        case PipelineSteps.STANDARD_SCALER.value:
            return StandardScaler(*args, **kwargs)
        case PipelineSteps.MIN_MAX_SCALER.value:
            return MinMaxScaler(*args, **kwargs)
        case PipelineSteps.VECTORIZER.value:
            return Vectorizer()
        case PipelineSteps.EPOCH_SPLITTER.value:
            return EpochSplitter(*args, **kwargs)
        case PipelineSteps.CHANNELS_REPLICATOR.value:
            return ChannelsReplicator(*args, **kwargs)
        case PipelineSteps.PADDING.value:
            return Padding(*args, **kwargs)
        case PipelineSteps.NOISER.value:
            return Noiser(*args, **kwargs)
        
        case PipelineSteps.SPECTROGRAM_COMPUTER.value:
            return STFTComputer(*args, **kwargs)
        case PipelineSteps.SPECTROGRAM_AUGMENTER.value:
            return SpectrogramAugmenter(*args, **kwargs)
        case PipelineSteps.SPECTROGRAM_STACKER.value:
            return SpectrogramStacker(*args, **kwargs)

        case PipelineSteps.PCA.value:
            return PCA(*args, **kwargs)
        case PipelineSteps.XDAWN.value:
            return XdawnCovariances(*args, **kwargs)

        case PipelineSteps.EEG_NET.value:
            return EEGNetBuilder.build(*args, **kwargs)
        case PipelineSteps.FG_MDM.value:
            return FgMDM(*args, **kwargs)
        case PipelineSteps.INCEPTION.value:
            return InceptionNetBuilder.build(*args, **kwargs)
        case PipelineSteps.LDA.value:
            return LinearDiscriminantAnalysis(*args, **kwargs)
        case PipelineSteps.MDM.value:
            return MDM(*args, **kwargs)
        case PipelineSteps.R_SVM.value:
            return R_SVC(*args, **kwargs)
        case PipelineSteps.SHALLOW_FBCSP_NET.value:
            return ShallowFBCSPNetBuilder.build(*args, **kwargs)
        case PipelineSteps.SVM.value:
            return SVC(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown pipeline step name: {name}")


def get_pipeline_steps(configs: List[DictInit]):
    return [get_step_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]


def resolve_dynamic_step_args(configs: List[DictInit], dataset: BaseDataset) -> List[DictInit]:
    if type(configs) is ListConfig:
        configs = OmegaConf.to_container(configs)
    for config in configs:
        if 'dynamic_args' in config:
            for arg in config['dynamic_args']:
                if arg == 'info':
                    if not dataset.raw_exists():
                        dataset.load()
                    config['kwargs'][arg] = dataset.raw.info
    return configs

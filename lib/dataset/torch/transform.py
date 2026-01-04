import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
from mne import Info
from mne.decoding import Scaler
from numpy import ndarray
from enum import Enum
from torch import Size
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf, ListConfig
from lib.config import DictInit
from lib.config import config_to_primitive
import numpy as np
import mne

class TorchTransforms(Enum):
    EEG_NORMALIZE = 'eeg_normalize'
    EEG_SCALE = 'eeg_scale'
    CROP = 'crop'
    MIN_MAX_SCALER = 'min_max_scaler'
    PAD_OR_CROP = 'pad_or_crop'
    CHANNEL_SELECTOR = 'channel_selector'
    CHANNEL_SPLITTER = 'channel_splitter'


class TorchTransform(nn.Module):
    name: str

    def __init__(self, name: str, *args, **kwargs):
        super().__init__()
        self.name = name

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        pass


class EEGNormalize(TorchTransform):
    mean: float | None
    std: float | None
    inplace: bool
    dim: int | None

    def __init__(self,
                 mean: float | List[float] | None = None,
                 std: float | List[float] | None = None,
                 inplace: bool = False,
                 dim: int | None = None):
        super().__init__(TorchTransforms.EEG_NORMALIZE.value)
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.dim = dim

    def normalize(self, tensor: Tensor) -> Tensor:
        assert isinstance(tensor, torch.Tensor), f"Expected tensor, but got {type(tensor)}."
        assert tensor.ndim >= 3, f"Expected tensor with at least 3 dimensions, but got {tensor.ndim}."

        if not self.inplace:
            tensor = tensor.clone()

        if (self.mean is None) or (self.std is None):
            if self.dim is None:
                mean = tensor.mean()
                std = tensor.mean()
            else:
                mean = tensor.mean(dim=self.dim, keepdim=True)
                std = tensor.std(dim=self.dim, keepdim=True)
        else:
            mean = self.mean
            std = self.std
            if self.dim is None:
                self.dim = 2 if tensor.ndim > 3 else 1 if tensor.ndim > 2 else 0

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")

        shape = [1] * len(tensor.shape)
        shape[self.dim] = -1
        if mean.ndim == 1:
            mean = mean.view(*shape)
        if std.ndim == 1:
            std = std.view(*shape)

        return tensor.sub_(mean).div_(std)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        return self.normalize(x)


class EEGScale(TorchTransform):
    info: Info | None
    scalings: dict | str | None
    with_mean: bool
    with_std: bool

    scaler: Scaler

    def __init__(self,
                 info: Info | None = None,
                 scalings: dict | str | None = None,
                 with_mean: bool = True,
                 with_std: bool = True):
        super().__init__(TorchTransforms.EEG_SCALE.value)
        self.info = info
        self.scalings = scalings
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = Scaler(info=self.info, scalings=self.scalings, with_mean=self.with_mean, with_std=self.with_std)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        device = x.device
        dtype = x.dtype
        data = x.detach().cpu().numpy()
        scaled_data = self.scaler.fit_transform(data)
        return torch.tensor(scaled_data, device=device, dtype=dtype)


class Crop(TorchTransform):
    def __init__(self, n_channels: int | None = None, n_times: int | None = None):
        super().__init__(TorchTransforms.CROP.value)
        self.n_channels = n_channels
        self.n_times = n_times

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        if len(x.shape) == 3:
            if self.n_channels is not None:
                x = x[:, :self.n_channels, :]
            if self.n_times is not None:
                x = x[:, :, :self.n_times]
        elif len(x.shape) == 4:
            if self.n_channels is not None:
                x = x[:, :, :self.n_channels, :]
            if self.n_times is not None:
                x = x[:, :, :, :self.n_times]
        return x


class MinMaxScaler(TorchTransform):
    """
    This class scales data for each channel. It differs from scikit-learn
    classes (e.g., :class:`sklearn.preprocessing.StandardScaler`) in that
    it scales each *channel* by estimating μ and σ using data from all
    time points and epochs, as opposed to standardizing each *feature*
    (i.e., each time point for each channel) by estimating using μ and σ
    using data from all epochs.
    """

    def __init__(self,
                 data: Tensor | ndarray,
                 model_shape: Size,
                 feature_range: Tuple[int, int] = (0, 1)):
        """
        :param data: Input tensor of shape (n_epochs, n_channels, n_times)
        """
        super().__init__(TorchTransforms.MIN_MAX_SCALER.value)
        if isinstance(data, ndarray):
            data = torch.Tensor(data.copy())
        orig_shape = data.shape
        data = torch.reshape(data.transpose(1, 2), (-1, orig_shape[1]))

        self.channel_min = data.min(dim=0).values
        self.channel_max = data.max(dim=0).values
        
        self.model_shape = model_shape
        self.orig_shape = orig_shape
        self.n_stacks = self.model_shape[1] // data.shape[1]
        self.stacked =  self.model_shape[1] % data.shape[1] == 0

        if len(self.channel_min) < self.model_shape[1]:
            self.channel_min = self.channel_min.repeat(self.n_stacks)
            self.channel_max = self.channel_max.repeat(self.n_stacks)
        elif len(self.channel_min) > self.model_shape[1]:
            self.channel_min = self.channel_min[:self.model_shape[0]]
            self.channel_max = self.channel_max[:self.model_shape[0]]

        self.min = feature_range[0]
        self.max = feature_range[1]

    def verify_channels(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        channel_dim = 0 if len(x.shape) == 2 else 1 if len(x.shape) == 3 else 2
        channel_min = self.channel_min
        channel_max = self.channel_max
        if len(channel_min) < x.shape[channel_dim]:
            h = x.clone()
            h = h[0] if len(h.shape) == 3 else h[0][0] if len(h.shape) == 4 else h
            if torch.any(torch.all(h.eq(0), dim=1)):
                padded_channels = torch.where(torch.all(h.eq(0), dim=1))[0]
                for channel in padded_channels:
                    channel_min = torch.cat([channel_min[:channel], torch.zeros(1) - 1, channel_min[channel:]])
                    channel_max = torch.cat([channel_max[:channel], torch.zeros(1) + 1, channel_max[channel:]])
        if len(channel_min) > x.shape[channel_dim]:
            channel_min = self.channel_min[:x.shape[channel_dim]]
            channel_max = self.channel_max[:x.shape[channel_dim]]
        return channel_min, channel_max

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """
        :param x: Input tensor of shape (n_epochs, n_channels, n_times)
        :return: Tensor of shape (n_epochs, n_channels, n_times) with each channel scaled to feature_range
        """
        orig_shape = x.shape
        channel_min, channel_max = self.verify_channels(x)
        x = torch.reshape(x.transpose(1, 2), (-1, orig_shape[1]))
        x_std = (x - channel_min) / (channel_max - channel_min)
        x_scaled = x_std * (self.max - self.min) + self.min
        x_scaled = torch.reshape(x_scaled, (orig_shape[0], orig_shape[2], orig_shape[1]))
        x_scaled = x_scaled.transpose(1, 2)
        return x_scaled


class PadOrCrop(TorchTransform):

    def __init__(self, output_shape: Size, pad_value: float = 0):
        super().__init__(TorchTransforms.PAD_OR_CROP.value)
        self.output_shape = output_shape
        self.pad_value = pad_value

    def pad(self, x: Tensor, position: int, pad: int) -> Tensor:
        padding = torch.zeros(4, dtype=torch.int)
        padding[position] = pad
        return F.pad(x, tuple(padding.tolist()), value=self.pad_value)

    def crop(self, x: Tensor, dim: int, size: int) -> Tensor:
        num_dims = len(x.shape)
        if num_dims == 3:
            return x[:, :size, :] if dim == 0 else x[:, :size, :] if dim == 1 else x[:, :, :size] if dim == 2 else x
        elif num_dims == 4:
            return x[:, :size, :, :] if dim == 1 else x[:, :, :size, :] if dim == 2 else x[:, :, :,
                                                                                         :size] if dim == 3 else x
        return x

    def unsqueeze_output_shape(self, num_dims: int):
        output_shape = self.output_shape
        if num_dims == 4:
            output_shape = list(output_shape)
            output_shape.insert(1, 1)
            output_shape = tuple(output_shape)
        return output_shape

    def pad_or_crop(self, x: Tensor) -> Tensor:
        num_dims = len(x.shape)
        output_shape = self.unsqueeze_output_shape(num_dims)

        eeg_channel_dim = 1 if num_dims == 3 else 2
        times_dim = 2 if num_dims == 3 else 3

        if x.shape[eeg_channel_dim:] == output_shape:
            return x

        dims = [(eeg_channel_dim, 3), (times_dim, 1)]
        for (dim, position) in dims:
            if x.shape[dim] < output_shape[dim]:
                pad = output_shape[dim] - x.shape[dim]
                x = self.pad(x, position, pad)
            elif x.shape[dim] > output_shape[dim]:
                x = self.crop(x, dim, output_shape[dim])
        return x

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        return self.pad_or_crop(x)


class ChannelSelector(TorchTransform):
    def __init__(self, channels: int | List[int]):
        super().__init__(TorchTransforms.CHANNEL_SELECTOR.value)
        self.channels = channels

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        return x[:, self.channels, :]


class ChannelSplitter(TorchTransform):
    n_splits: int

    def __init__(self, n_splits: int = 8):
        super().__init__(TorchTransforms.CHANNEL_SPLITTER.value)
        self.n_splits = n_splits

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        if len(x.shape) == 4:
            return x.reshape(int(x.shape[0] * self.n_splits), x.shape[1], int(x.shape[2] // self.n_splits), -1)
        elif len(x.shape) == 3:
            return x.reshape(int(x.shape[0] * self.n_splits), int(x.shape[1] // self.n_splits), -1)
        else:
            raise ValueError(f"Expected 3 or 4 dimensions, but got {len(x.shape)} dimensions.")


def resolve_dynamic_args(configs: List[DictInit], dataset: Any) -> List[DictInit]:
    if type(configs) is ListConfig:
        configs = OmegaConf.to_container(configs)
    for config in configs:
        if 'dynamic_args' in config:
            for arg in config['dynamic_args']:
                if arg == 'data':
                    config['kwargs'][arg] = dataset.data
    return configs


def get_torch_transform_by_name(name: str, *args, **kwargs) -> TorchTransform:
    match name:
        case TorchTransforms.EEG_NORMALIZE.value:
            return EEGNormalize(*args, **kwargs)
        case TorchTransforms.EEG_SCALE.value:
            return EEGScale(*args, **kwargs)
        case TorchTransforms.CROP.value:
            return Crop(*args, **kwargs)
        case TorchTransforms.MIN_MAX_SCALER.value:
            return MinMaxScaler(*args, **kwargs)
        case TorchTransforms.PAD_OR_CROP.value:
            return PadOrCrop(*args, **kwargs)
        case TorchTransforms.CHANNEL_SELECTOR.value:
            return ChannelSelector(*args, **kwargs)
        case TorchTransforms.CHANNEL_SPLITTER.value:
            return ChannelSplitter(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown torch transform name: {name}")


def get_transforms(configs: List[Dict], dataset: Any | None = None) -> List[TorchTransform]:
    if len(configs) > 0 and (type(configs[0]) is DictInit or type(configs[0]) is DictConfig):
        configs = config_to_primitive(configs)
        configs = resolve_dynamic_args(configs, dataset)
    return [get_torch_transform_by_name(config['name'], *config['args'], **config['kwargs']) for config in configs]


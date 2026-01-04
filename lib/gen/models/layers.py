import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Pad, CenterCrop
import torch


class UpSampling2DCubical(nn.Module):
    """ A layer to perform 2D bicubic upsampling on spectrogram inputs """
    stride: int

    def __init__(self, stride: int=2):
        super(UpSampling2DCubical, self).__init__()
        self.stride = stride

    def forward(self, x: Tensor):
        input_shape = x.shape
        output_shape = (self.stride * input_shape[2], self.stride * input_shape[3])
        return nn.functional.interpolate(x, size=output_shape, mode='bicubic', align_corners=True)


class ClipLayer(nn.Module):
    """ A layer to crop or pad 2D inputs to a target height and width """
    def __init__(self, target_height: int, target_width: int):
        super(ClipLayer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x: Tensor):
        padding = (x.shape[2] - self.target_height, x.shape[3] - self.target_width)
        if padding[0] < 0 or padding[1] < 0:  # Correct for excess dimensions
            padding = (0, 0)
        crop = CenterCrop((self.target_height, self.target_width))
        return crop(x)


class MeanZeroLayer(nn.Module):
    """ A layer that normalizes 2D inputs to have a zero mean """
    def __init__(self):
        super(MeanZeroLayer, self).__init__()

    def forward(self, x):
        mu = torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True)
        x = x - mu
        return x


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer """

    def __init__(self, sigma=0.05, is_relative_detach=True):  # Reduced noise for higher fidelity spectrograms
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


def linear_kernel_initialize(stride, num_channels):
    stride = stride if isinstance(stride, int) else stride[0]
    filter_size = (2 * stride - stride % 2)
    # Create linear weights in numpy array
    linear_kernel = torch.zeros([filter_size, filter_size], dtype=torch.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            linear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                  (1 - abs(y - center) / scale_factor)
    weights = torch.zeros((num_channels, num_channels, filter_size, filter_size))
    for i in range(num_channels):
        weights[i, i, :, :] = linear_kernel
    weights = weights.type(torch.float32)
    return weights


def get_conv_dim_shape(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    return (input_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1


def get_conv_shape(input_shape: tuple[int, int],
                   kernel_size: int | tuple[int, int],
                   stride: int | tuple[int, int],
                   padding: int | tuple[int, int]) -> tuple[int, int]:
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    return (get_conv_dim_shape(input_shape[0], kernel_size[0], stride[0], padding[0]),
            get_conv_dim_shape(input_shape[1], kernel_size[1], stride[1], padding[1]))


def calc_conv_shape(
    input_shape: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int] | str,
    n_layers: int = 1
) -> tuple[int, int]:
    
    
    def to_tuple(param, name):
        """ Convert param to a tuple of integers if necessary """
        if isinstance(param, str):
            if param.lower() == "same":
                # Calculate 'same' padding dynamically
                return ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            raise ValueError(f"Unsupported padding type: '{param}' for {name}. Provide numeric padding.")
        if isinstance(param, int):
            return (param, param)
        return tuple(param)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    height, width = input_shape

    for _ in range(n_layers):
        current_padding = to_tuple(padding, "padding")

        height = (height + 2 * current_padding[0] - kernel_size[0]) // stride[0] + 1
        width = (width + 2 * current_padding[1] - kernel_size[1]) // stride[1] + 1

    return height, width

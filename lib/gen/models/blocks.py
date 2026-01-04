from typing import Tuple
from torch import nn


def cwgan_critic_conv_block(in_channels: int,
                            out_channels: int,
                            kernel_size: int | Tuple[int, int] | None = 4,
                            stride: int | Tuple[int, int] | None = (1, 2),
                            padding: int | Tuple[int, int] | None = (1, 1),
                            negative_slope: float | None = 0.2):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        ),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(negative_slope=negative_slope)
    )


def cwgan_generator_conv_block(in_channels: int,
                               out_channels: int,
                               kernel_size: int | Tuple[int, int] | None = 4,
                               stride: int | Tuple[int, int] | None = (2, 2),  # Symmetrical upsampling
                               padding: int | Tuple[int, int] | None = (1, 1)):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

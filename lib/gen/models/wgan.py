from torch import nn
from typing import Tuple
from lib.gen.models.blocks import cwgan_critic_conv_block, cwgan_generator_conv_block
from lib.gen.models.loss import gradient_penalty
from torch import Tensor
import torch


class Critic(nn.Module):

    def __init__(self, input_channels: int, features_d: int):
        super(Critic, self).__init__()

        kernel_size = 4
        stride = (1, 2)
        padding = 1
        negative_slope = 0.2

        self.critic = nn.Sequential(
            nn.Conv2d(self.input_channels, features_d, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope),
            cwgan_critic_conv_block(features_d, features_d * 2, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 2, features_d * 4, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 4, features_d * 8, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 8, features_d * 16, self.kernel_size, self.stride, self.padding),
            nn.Conv2d(features_d * 16, 1, kernel_size=(3, 4), stride=(1, 1), padding=0),
        )

    def forward(self, x, **kwargs):
        return self.critic(x)

    def initialize(self) -> 'Critic':
        initialize_weights(self)
        return self

    def loss(self, real: Tensor, fake: Tensor, labels: Tensor):
        critic_real = self(real, labels).view(-1)
        critic_fake = self(fake, labels).view(-1)
        gp = gradient_penalty(self, real, fake, labels, self.device)
        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + self.lambda_gp * gp
        return loss_critic


class Generator(nn.Module):

    def __init__(self, z_dim: int, input_channels: int, features_g: int):
        super(Generator, self).__init__()

        kernel_size = 4
        stride = (1, 2)
        padding = 1

        self.gen = nn.Sequential(
            cwgan_generator_conv_block(z_dim, features_g * 32, stride=(1, 1), padding=(0, 0)),
            cwgan_generator_conv_block(features_g * 32, features_g * 16, self.kernel_size, self.stride, padding=(1, 1)),
            cwgan_generator_conv_block(features_g * 16, features_g * 8, self.kernel_size, self.stride, padding=(1, 0)),
            cwgan_generator_conv_block(features_g * 8, features_g * 4, self.kernel_size, self.stride, padding=(1, 0)),
            cwgan_generator_conv_block(features_g * 4, features_g * 2, self.kernel_size, self.stride, padding=(1, 0)),
            nn.ConvTranspose2d(features_g * 2, self.input_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

    def initialize(self) -> 'Generator':
        initialize_weights(self)
        return self

    def loss(self, critic_output: Tensor):
        return -torch.mean(critic_output)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)

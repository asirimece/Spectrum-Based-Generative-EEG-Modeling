from torch import nn, Tensor
from torch.cuda import Device
from typing import Tuple
import torch
from torch import Size
from lib.gen.models.base import Critic, Generator, GAN
from lib.gen import GanTrainConfig
from lib.gen.models.loss import gradient_penalty
from lib.gen.models.blocks import cwgan_critic_conv_block, cwgan_generator_conv_block


class CWGANBase:
    _net: nn.Module

    @property
    def net(self) -> nn.Module:
        return self._net

    @net.setter
    def net(self, net: nn.Module):
        self._net = net


class CWGANCritic(Critic, CWGANBase):

    def __init__(self,
                 input_shape: Size,
                 n_classes: int,
                 features_d: int = 64,
                 kernel_size: int = 4,
                 stride: int = (1, 2),
                 padding: int = (1, 1),
                 negative_slope: float = 0.2,
                 lambda_gp: float = 10.0,
                 device: Device = 'cpu',
                 net: nn.Module | None = None
                 ):
        super(CWGANCritic, self).__init__(input_shape, device)

        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.negative_slope = negative_slope
        self.lambda_gp = lambda_gp

        self._net = nn.Sequential(
            nn.Conv2d(self.input_channels + 1, features_d, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope),
            cwgan_critic_conv_block(features_d, features_d * 2, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 2, features_d * 4, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 4, features_d * 8, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_d * 8, features_d * 16, self.kernel_size, self.stride, self.padding),
            nn.Conv2d(features_d * 16, 1, kernel_size=(3, 4), stride=(1, 1), padding=0),
        ) if net is None else net
        self.net.to(device)

        self.embedding = nn.Embedding(n_classes, self.eeg_channels * self.n_times, device=device)

    def forward(self, x: Tensor, labels: Tensor):
        embedding = self.embedding(labels).view(labels.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)

    def initialize(self) -> 'Critic':
        _initialize_weights(self)
        return self

    def loss(self, real: Tensor, fake: Tensor, labels: Tensor):
        critic_real = self(real, labels)
        loss_real = torch.mean(critic_real)

        critic_fake = self(fake.detach(), labels)
        loss_fake = torch.mean(critic_fake)

        gp = gradient_penalty(self, real, fake, labels, self.device)
        self.add_loss_history('c_loss_gp', gp.item())

        loss_critic = -loss_real + loss_fake + self.lambda_gp * gp
        self.add_loss_history('c_loss', loss_critic.item())
        return loss_critic

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CWGANCritic':
        critic = CWGANCritic(
            input_shape=config.input_shape,
            n_classes=config.n_classes,
            device=config.device,
            features_d=config.critic.kwargs['features_d'] if 'features_d' in config.critic.kwargs else None,
            kernel_size=config.critic.kwargs['kernel_size'] if 'kernel_size' in config.critic.kwargs else None,
            stride=config.critic.kwargs['stride'] if 'stride' in config.critic.kwargs else None,
            padding=config.critic.kwargs['padding'] if 'padding' in config.critic.kwargs else None,
            negative_slope=config.critic.kwargs['negative_slope'] if 'negative_slope' in config.critic.kwargs else None,
            lambda_gp=config.critic.kwargs['lambda_gp'] if 'lambda_gp' in config.critic.kwargs else None
        )
        critic.set_optim_by_config(config.critic.optim)
        return critic


class CWGANGenerator(Generator, CWGANBase):

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 z_dim: int,
                 features_g: int,
                 n_classes: int,
                 embed_size: int = 100,
                 kernel_size: int = 4,
                 stride: int = (1, 2),
                 padding: int = (1, 1),
                 device: Device = 'cpu',
                 net: nn.Module | None = None
                 ):
        super(CWGANGenerator, self).__init__(
            z_dim=z_dim,
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device
        )

        self.n_classes = n_classes
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._net = nn.Sequential(
            cwgan_generator_conv_block(z_dim + embed_size, features_g * 32, stride=(1, 1), padding=(0, 0)),
            cwgan_generator_conv_block(features_g * 32, features_g * 16, self.kernel_size, self.stride, padding=(1, 1)),
            cwgan_generator_conv_block(features_g * 16, features_g * 8, self.kernel_size, self.stride, padding=(1, 0)),
            cwgan_generator_conv_block(features_g * 8, features_g * 4, self.kernel_size, self.stride, padding=(1, 0)),
            cwgan_generator_conv_block(features_g * 4, features_g * 2, self.kernel_size, self.stride, padding=(1, 0)),
            nn.ConvTranspose2d(features_g * 2, self.input_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            nn.Tanh()
        ) if net is None else net
        self.net.to(device)

        self.embedding = nn.Embedding(n_classes, embed_size, device=device)

    def forward(self, x, labels):
        embedding = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)

    def initialize(self) -> 'Generator':
        _initialize_weights(self)
        return self

    def loss(self, critic_output: Tensor):
        loss = -torch.mean(critic_output)
        self.add_loss_history('g_loss', loss.item())
        return loss

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CWGANGenerator':
        gen = CWGANGenerator(
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            z_dim=config.generator.z_dim,
            n_classes=config.n_classes,
            device=config.device,
            features_g=config.generator.kwargs['features_g'] if 'features_g' in config.generator.kwargs else None,
            embed_size=config.generator.kwargs['embed_size'] if 'embed_size' in config.generator.kwargs else None,
            kernel_size=config.generator.kwargs['kernel_size'] if 'kernel_size' in config.generator.kwargs else None,
            stride=config.generator.kwargs['stride'] if 'stride' in config.generator.kwargs else None,
            padding=config.generator.kwargs['padding'] if 'padding' in config.generator.kwargs else None
        )
        gen.set_optim_by_config(config.generator.optim)
        return gen


class CWGAN(GAN):

    def __init__(self,
                 generator: CWGANGenerator | None,
                 critic: CWGANCritic | None,
                 ):
        super(CWGAN, self).__init__(generator, critic)

    def initialize(self) -> 'CWGAN':
        self.generator.initialize()
        self.critic.initialize()
        return self

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CWGAN':
        return CWGAN(
            generator=CWGANGenerator.from_configs(config),
            critic=CWGANCritic.from_configs(config)
        )


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)




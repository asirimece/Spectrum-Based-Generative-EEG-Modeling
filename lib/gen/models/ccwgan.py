from torch import nn, Tensor
from torch.cuda import Device
import torch
from torch import Size
from lib.gen.models.base import Critic, Generator, GAN
from lib.gen import GanTrainConfig
from lib.gen.models.loss import gradient_penalty
from lib.gen.models.layers import ClipLayer, MeanZeroLayer, GaussianNoise, linear_kernel_initialize
from lib.gen.models.layers import calc_conv_shape
from lib.gen.models.cwgan import CWGANBase
from lib.dataset.torch.transform import get_transforms
from typing import Callable, Any


class CCWGANCritic(Critic):  
    _backbone: nn.Module
    _head_validity: nn.Module
    _head_label: nn.Module

    criterion_label: nn.Module

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 n_classes: int,
                 negative_slope: float = 0.2,
                 lambda_gp: float = 10.0,
                 device: Device = 'cpu',
                 noise_sigma: float = 0.05,
                 stride: int | tuple[int, int] = 2,
                 padding: int | tuple[int, int] = 1,
                 backbone: nn.Module | None = None,
                 valid_head: nn.Module | None = None,
                 label_head: nn.Module | None = None,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None
                 ):
        super(CCWGANCritic, self).__init__(
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors
        )

        self.n_classes = n_classes
        self.negative_slope = negative_slope
        self.lambda_gp = lambda_gp

        noise1 = GaussianNoise(noise_sigma)

        conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=(3,3), padding='same')
        non_lin1 = nn.LeakyReLU(negative_slope=negative_slope)

        conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=stride, padding=padding)
        non_lin2 = nn.LeakyReLU(negative_slope=negative_slope)
        nn.init.kaiming_normal_(conv2.weight)

        conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=stride, padding=padding)
        non_lin3 = nn.LeakyReLU(negative_slope=negative_slope)

        conv4_extra = nn.Conv2d(256, 512, kernel_size=(3,3), stride=1, padding=1)
        nn.init.kaiming_normal_(conv4_extra.weight)
        non_lin4_extra = nn.LeakyReLU(negative_slope=negative_slope)

        out_dim = calc_conv_shape((self.model_eeg_channels, self.model_n_times), (3,3), stride, padding, 2)
        flatten4 = nn.Flatten()
        linear4 = nn.Linear(512 * out_dim[0] * out_dim[1], 1024)
        
        non_lin4 = nn.LeakyReLU(negative_slope=negative_slope)

        self._backbone = nn.Sequential(
            noise1,
            conv1,
            non_lin1,

            conv2,
            non_lin2,

            conv3,
            non_lin3,
            
            conv4_extra,
            non_lin4_extra,

            flatten4,
            linear4,
            non_lin4

        ) if backbone is None else backbone
        self.backbone.to(device)

        self.linear_head_validity = nn.Linear(1024, 1)
        nn.init.kaiming_normal_(self.linear_head_validity.weight)
        self._head_validity = nn.Sequential(
            self.linear_head_validity
        ) if valid_head is None else valid_head
        self.head_validity.to(device)

        self.linear_head_label = nn.Linear(1024, n_classes)
        nn.init.kaiming_normal_(self.linear_head_label.weight)
        self._head_label = nn.Sequential(
            self.linear_head_label,
            nn.Softmax(dim=1)
        ) if label_head is None else label_head
        self.head_label.to(device)

        self.criterion_label = nn.CrossEntropyLoss()

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def head_validity(self) -> nn.Module:
        return self._head_validity

    @property
    def head_label(self) -> nn.Module:
        return self._head_label

    def forward(self, x: Tensor, labels: Tensor):
        x = self.backbone(x)

        validity = self.head_validity(x)
        pred_labels = self.head_label(x)
        return validity.unsqueeze(1).unsqueeze(2), pred_labels

    def initialize(self) -> 'Critic':
        return self

    def loss_validity(self, real: Tensor, fake: Tensor, labels: Tensor):
        critic_real, _ = self(real, labels)
        critic_fake, _ = self(fake, labels)
        gp = gradient_penalty(self, real, fake, labels, self.device)
        self.add_loss_history('c_loss_gp', gp.item())
        loss_critic = -(torch.mean(critic_real.view(-1)) - torch.mean(critic_fake.view(-1))) + self.lambda_gp * gp
        self.add_loss_history('c_loss_validity', loss_critic.item())
        return loss_critic

    def loss_label(self, real: Tensor, fake: Tensor, labels: Tensor):
        _, pred_labels_real = self(real, labels)
        _, pred_labels_fake = self(fake, labels)
        label_loss_real = self.criterion_label(pred_labels_real, labels)
        self.add_loss_history('c_loss_label_real', label_loss_real.item())
        label_loss_fake = self.criterion_label(pred_labels_fake, labels)
        self.add_loss_history('c_loss_label_fake', label_loss_fake.item())
        loss_label = (label_loss_real + label_loss_fake) / 2
        self.add_loss_history('c_loss_label', loss_label.item())
        return loss_label

    def loss(self, real: Tensor, fake: Tensor, labels: Tensor):
        loss_validity = self.loss_validity(real, fake, labels)
        loss_label = self.loss_label(real, fake, labels)
        return loss_validity + loss_label

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CCWGANCritic':
        critic = CCWGANCritic(
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            n_classes=config.n_classes,
            device=config.device,
            negative_slope=config.critic.kwargs['negative_slope'] if 'negative_slope' in config.critic.kwargs else None,
            lambda_gp=config.critic.kwargs['lambda_gp'] if 'lambda_gp' in config.critic.kwargs else None,
            stride=config.critic.kwargs['stride'] if 'stride' in config.critic.kwargs else 2,
            padding=config.critic.kwargs['padding'] if 'padding' in config.critic.kwargs else 1,
        )
        critic.set_optim_by_config(config.critic.optim)
        return critic


class CCWGANGenerator(Generator, CWGANBase):

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 z_dim: int,
                 n_classes: int,
                 device: Device = 'cpu',
                 net: nn.Module | None = None,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None
                 ):
        super(CCWGANGenerator, self).__init__(
            z_dim=z_dim,
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors)

        self.n_classes = n_classes
        n_hidden_channels = int(self.model_eeg_channels / 4) + 1
        n_hidden_features = int(self.model_n_times / 4) + 1

        flatten1 = nn.Flatten()
        linear1 = nn.Linear(self.z_dim, 1024)
        non_lin1 = nn.LeakyReLU()

        linear2 = nn.Linear(1024, 128 * n_hidden_channels * n_hidden_features)
        bn2 = nn.BatchNorm1d(128 * n_hidden_channels * n_hidden_features)
        non_lin2 = nn.LeakyReLU()

        unflatten3 = nn.Unflatten(1, torch.Size([128, n_hidden_channels, n_hidden_features]))
        upsample3 = nn.Upsample(scale_factor=2, mode='bicubic')
        bn3 = nn.BatchNorm2d(128)
        non_lin3 = nn.LeakyReLU()

        conv4 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=1)
        bn4 = nn.BatchNorm2d(64)
        non_lin4 = nn.LeakyReLU()

        conv_trans5 = nn.ConvTranspose2d(64, 128, kernel_size=(4,4), stride=2, padding=1)
        clip5 = ClipLayer(self.model_input_shape[1], self.model_input_shape[2])
        bn5 = nn.BatchNorm2d(128)
        non_lin5 = nn.LeakyReLU()
        trans6_extra = nn.ConvTranspose2d(128, 128, kernel_size=(4,4), stride=2, padding=1)
        
        conv6 = nn.Conv2d(128, 8, kernel_size=(3,3), padding=1) 
        tanh6 = nn.Tanh()
        mean_zero6 = MeanZeroLayer()

        self._net = nn.Sequential(
            flatten1,
            linear1,
            non_lin1,

            linear2,
            bn2,
            non_lin2,

            unflatten3,
            upsample3,
            bn3,
            non_lin3,

            conv4,
            bn4,
            non_lin4,

            conv_trans5,
            clip5,
            bn5,
            non_lin5,

            conv6,
            tanh6,
            mean_zero6
        ) if net is None else net
        self.net.to(device)

        self.embedding = nn.Embedding(n_classes, z_dim, device=device)
        
    def forward(self, x, labels):
        assert x.shape[0] == labels.shape[0], "Batch size of x and labels must be the same"
        embedding = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.multiply(x, embedding)

        x = self.net(x)
        
        return x

    def initialize(self) -> 'Generator':
        _initialize_weights(self)
        return self

    def loss(self, critic_output: Tensor):
        loss = -torch.mean(critic_output)
        self.add_loss_history('g_loss', loss.item())
        return loss

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CCWGANGenerator':
        gen = CCWGANGenerator(
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            z_dim=config.generator.z_dim,
            n_classes=config.n_classes,
            output_preprocessors=get_transforms(config.model.output_preprocessors),
            device=config.device
        )
        gen.set_optim_by_config(config.generator.optim)
        return gen


class CCWGAN(GAN):

    def __init__(self,
                 generator: CCWGANGenerator | None,
                 critic: CCWGANCritic | None,
                 ):
        super(CCWGAN, self).__init__(generator, critic)

    def initialize(self) -> 'CCWGAN':
        self.generator.initialize()
        self.critic.initialize()
        return self

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CCWGAN':
        return CCWGAN(
            generator=CCWGANGenerator.from_configs(config),
            critic=CCWGANCritic.from_configs(config)
        )


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.ConvTranspose2d):
            weights = linear_kernel_initialize(m.stride, 1)
            m.weight.data.copy_(weights.expand_as(m.weight))

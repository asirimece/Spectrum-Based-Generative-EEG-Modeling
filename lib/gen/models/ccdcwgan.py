from torch import nn, Tensor
from torch.cuda import Device
import torch
from torch import Size
from lib.gen.models.base import Critic, GAN
from lib.gen import GanTrainConfig
from lib.gen.models.loss import gradient_penalty
from lib.gen.models.blocks import cwgan_critic_conv_block
from lib.gen.models.cwgan import CWGANGenerator


class CCDCWGANCritic(Critic):
    _backbone: nn.Module
    _head_validity: nn.Module
    _head_label: nn.Module

    criterion_label: nn.Module

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 n_classes: int,
                 features_c: int = 64,
                 kernel_size: int = 4,
                 stride: int = (1, 2),
                 padding: int = (1, 1),
                 negative_slope: float = 0.2,
                 lambda_gp: float = 10.0,
                 device: Device = 'cpu',
                 backbone: nn.Module | None = None,
                 valid_head: nn.Module | None = None,
                 label_head: nn.Module | None = None,
                 ):
        super(CCDCWGANCritic, self).__init__(
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device
        )

        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.negative_slope = negative_slope
        self.lambda_gp = lambda_gp

        self._backbone = nn.Sequential(
            nn.Conv2d(self.input_channels + 1, features_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=negative_slope),
            cwgan_critic_conv_block(features_c, features_c * 2, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_c * 2, features_c * 4, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_c * 4, features_c * 8, self.kernel_size, self.stride, self.padding),
            cwgan_critic_conv_block(features_c * 8, features_c * 16, self.kernel_size, self.stride, self.padding),
        ) if backbone is None else backbone
        self.backbone.to(device)

        self._head_validity = nn.Sequential(
            nn.Conv2d(features_c * 16, 1, kernel_size=(3, 4), stride=(1, 1), padding=0)
        ) if valid_head is None else valid_head
        self.head_validity.to(device)

        self._head_label = nn.Sequential(
            nn.Linear(features_c * 16 * 3 * 4, n_classes),
            nn.Softmax(dim=1)
        ) if label_head is None else label_head
        self.head_label.to(device)

        self.embedding = nn.Embedding(n_classes, self.eeg_channels * self.n_times, device=device)
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
        embedding = self.embedding(labels).view(labels.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, embedding], dim=1)
        z = self._backbone(x)
        validity = self._head_validity(z)
        pred_labels = self._head_label(z.view(z.size(0), -1))
        return validity, pred_labels

    def initialize(self) -> 'Critic':
        _initialize_weights(self)
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
        loss = loss_validity + loss_label
        self.add_loss_history('c_loss', loss.item())
        return loss

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CCDCWGANCritic':
        critic = CCDCWGANCritic(
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            n_classes=config.n_classes,
            device=config.device,
            features_c=config.critic.kwargs['features_c'] if 'features_c' in config.critic.kwargs else None,
            kernel_size=config.critic.kwargs['kernel_size'] if 'kernel_size' in config.critic.kwargs else None,
            stride=config.critic.kwargs['stride'] if 'stride' in config.critic.kwargs else None,
            padding=config.critic.kwargs['padding'] if 'padding' in config.critic.kwargs else None,
            negative_slope=config.critic.kwargs['negative_slope'] if 'negative_slope' in config.critic.kwargs else None,
            lambda_gp=config.critic.kwargs['lambda_gp'] if 'lambda_gp' in config.critic.kwargs else None
        )
        critic.set_optim_by_config(config.critic.optim)
        return critic


class CCDCWGAN(GAN):

    def __init__(self,
                 generator: CWGANGenerator | None,
                 critic: CCDCWGANCritic | None,
                 ):
        super(CCDCWGAN, self).__init__(generator, critic)

    def initialize(self) -> 'CCDCWGAN':
        self.generator.initialize()
        self.critic.initialize()
        return self

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'CCDCWGAN':
        return CCDCWGAN(
            generator=CWGANGenerator.from_configs(config),
            critic=CCDCWGANCritic.from_configs(config)
        )


def _initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.2)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

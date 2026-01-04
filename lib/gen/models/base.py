from lib.logging._logger import get_logger
from torch import nn
from torch.optim import Optimizer
from torch import Size
from lib.gen import GanTrainConfig, OptimConfig
from lib.gen.optim import get_optimizer
from torch.cuda import Device
from typing import Any, Callable, Dict
import numpy as np
from torch import Tensor
import torch


def _loss_unimplemented(self):
    raise NotImplementedError(f"Loss method not implemented for {self.__class__.__name__}")


class NNBase(nn.Module):
    input_shape: Size
    input_channels: int
    eeg_channels: int
    n_times: int

    model_input_shape: Size
    model_eeg_channels: int
    model_n_times: int

    input_preprocessors: list[Callable[..., Any]] = []
    output_preprocessors: list[Callable[..., Any]] = []

    _optim: Optimizer
    device: Device

    loss_history: dict[str, list[float]] = {}

    seed: int

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 device: Device = 'cpu',
                 seed: int = 1):
        super(NNBase, self).__init__()
        self.logger = get_logger()
        self.device = device
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.eeg_channels = input_shape[1]
        self.n_times = input_shape[2]

        self.model_input_shape = model_input_shape
        self.model_eeg_channels = model_input_shape[1]
        self.model_n_times = model_input_shape[2]

        self.input_preprocessors = input_preprocessors or []
        self.output_preprocessors = output_preprocessors or []
        self.seed = seed
        
        self.input_shape = input_shape
    
    loss: Callable[..., Any] = _loss_unimplemented

    @property
    def optim(self) -> Optimizer:
        return self._optim

    @optim.setter
    def optim(self, optimizer: Optimizer):
        self._optim = optimizer

    # verify if not self.net is required for cwgan
    def set_optim_by_config(self, config: OptimConfig):
        self.optim = get_optimizer(self, config)

    def initialize(self, **kwargs) -> 'Critic':
        pass

    def reset_history(self):
        self.loss_history = {}

    def add_loss_history(self, name: str, value: float):
        if name not in self.loss_history:
            self.loss_history[name] = []
        self.loss_history[name].append(value)

    def get_averaged_losses(self) -> Dict[str, float]:
        return {k: np.mean(np.array(v)) for k, v in self.loss_history.items()}

    def get_averaged_loss(self) -> float:
        return np.mean(np.array(list(self.get_averaged_loss_history().values())))

    def sample(self, labels: Tensor, **kwargs) -> Tensor:
        pass


class Critic(NNBase):

    def __init__(self,
                 input_shape: Size,
                 model_input_shape: Size,
                 device: Device = 'cpu',
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 **kwargs):
        super(Critic, self).__init__(
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors)

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'Critic':
        critic = Critic(
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            device=config.device
        )
        critic.set_optim_by_config(config.generator.optim)
        return critic


class Generator(NNBase):
    z_dim: int

    def __init__(self,
                 z_dim: int,
                 input_shape: Size,
                 model_input_shape: Size,
                 device: Device = 'cpu',
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 **kwargs):
        super(Generator, self).__init__(
            input_shape=input_shape,
            model_input_shape=model_input_shape,
            device=device,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors)
       
        self.z_dim = z_dim

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'Generator':
        gen = Generator(
            z_dim=config.generator.z_dim,
            input_shape=config.input_shape,
            model_input_shape=config.model.model_shape,
            device=config.device
        )
        gen.set_optim_by_config(config.generator.optim)
        return gen

    def sample(self, labels: Tensor, **kwargs) -> Tensor:
        self.logger.info(f"Sample method called with labels shape: {labels.shape}")
        z = torch.randn((len(labels), self.z_dim, 1, 1)).to(self.device)
        self.logger.info(f"Latent variable z shape: {z.shape}")
        
        self.logger.info(f"Input to generator: z shape {z.shape}, labels shape {labels.shape}")
        y = self(z, labels)
        if self.output_preprocessors is not None and len(self.output_preprocessors) > 0:
            for preprocessor in self.output_preprocessors:
                y = preprocessor(y)
        return y


class GAN:
    generator: Generator
    critic: Critic

    def __init__(self,
                 generator: Generator,
                 critic: Critic,
                 **kwargs):
        self.generator = generator
        self.critic = critic

    def initialize(self) -> 'GAN':
        self.generator.initialize()
        self.critic.initialize()
        return self

    def train_mode(self):
        self.generator.train()
        self.critic.train()

    def eval_mode(self):
        self.generator.eval()
        self.critic.eval()

    def get_losses_history(self) -> Dict[str, list[float]]:
        generator = self.generator.loss_history
        critic = self.critic.loss_history
        return {**generator, **critic}

    def get_aggregated_losses(self, aggregator: str = 'mean') -> Dict[str, float]:
        losses = self.get_losses_history()
        if aggregator == 'mean':
            return {k: np.mean(np.array(v)) for k, v in losses.items()}

    def get_averaged_loss(self) -> float:
        values = list(self.get_aggregated_losses().values())
        if len(values) == 0:
            return 0
        return np.mean(np.array(list(self.get_aggregated_losses().values())))

    def reset_loss_history(self):
        self.generator.reset_history()
        self.critic.reset_history()
        
    def generate(self, real_data: Tensor, labels: Tensor) -> Tensor:
        """
        Generate synthetic data using the GAN generator.

        Args:
            real_data (Tensor): Input real data.
            labels (Tensor): Corresponding labels.

        Returns:
            Tensor: Generated synthetic data.
        """
        self.eval_mode()  # Set generator to evaluation mode
        with torch.no_grad():
            synthetic_data = self.generator.sample(labels)
        return synthetic_data

    @staticmethod
    def from_configs(config: GanTrainConfig) -> 'GAN':
        return GAN(
            generator=Generator.from_configs(config),
            critic=Critic.from_configs(config)
        )

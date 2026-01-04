from torch import nn
import torch
from torch.utils.data import DataLoader, random_split, Subset
from typing import Sequence, Union, List, Callable, Any
from torch.cuda import Device
from torch.optim import Optimizer
from lib.gen import OptimConfig
from lib.gen.models.base import NNBase
from lib.gen.optim import get_optimizer
from torch import Size
from lib.gen.model import Training, GenTrainConfig
from lib.gen import DiffusionTrainConfig
from lib.dataset import TorchBaseDataset
from torch import Tensor


class Diffusion(NNBase):
    model: nn.Module
    dataset: TorchBaseDataset | None

    train_dataloader: DataLoader | None
    val_dataloader: DataLoader | None

    criterion: nn.Module

    device: Device
    _optim: Optimizer

    def __init__(self,
                 input_shape: Size,
                 model_shape: Size,
                 model: nn.Module | None = None,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 device: Device = 'cpu',
                 seed: int = 1):
        super().__init__(input_shape=input_shape,
                         model_input_shape=model_shape,
                         input_preprocessors=input_preprocessors,
                         output_preprocessors=output_preprocessors,
                         device=device,
                         seed=seed)
        self.model = model
        self.model_shape = model_shape

    def initialize(self, config: DiffusionTrainConfig, dataset: TorchBaseDataset):
        pass

    @property
    def optim(self) -> Optimizer:
        return self._optim

    @optim.setter
    def optim(self, optimizer: Optimizer):
        self._optim = optimizer

    def train_mode(self):
        pass

    def eval_mode(self):
        pass

    # verify if not self.net is required for cwgan
    def set_optim_by_config(self, config: OptimConfig):
        self.optim = get_optimizer(self, config)

    def split_dataset(self, lengths: Sequence[Union[int, float]]) -> List[Subset]:
        generator = torch.Generator(device='cpu')
        return random_split(self.dataset, lengths=lengths, generator=generator)

    def test_shapes(self):
        pass

    def sample(self, labels: Tensor, cfg_scale: int = 3, use_ema: bool = True) -> Tensor:
        pass

    def run_training(self, config: GenTrainConfig, training: Training) -> Training:
        pass

    def run_validation(self, config: GenTrainConfig, training: Training) -> Training:
        pass

    def forward(self, labels: Tensor) -> Tensor:
        pass

    def clean_up(self):
        self.dataset = None
        self.val_dataloader = None
        self.train_dataloader = None

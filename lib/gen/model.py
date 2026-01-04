import torch
from typing import Tuple
from torch import Tensor, Size
from torch.cuda import Device
from omegaconf import DictConfig
from datetime import datetime
from mne import Info
from typing import Dict, List
from lib.dataset import DatasetType
from lib.logging import get_logger

logger = get_logger()

class ModelConfig:
    name: str
    model_type: str
    model_shape: Size

    input_preprocessors: List[Dict]
    output_preprocessors: List[Dict]

    seed: int = 1

    def __init__(self,
                 name: str,
                 model_type: str,
                 model_shape: Size = (8, 129, 11),
                 input_preprocessors: List[Dict] | None = None,
                 output_preprocessors: List[Dict] | None = None,
                 seed: int = 1):
        self.name = name
        self.model_type = model_type
        self.model_shape = model_shape
        self.input_preprocessors = input_preprocessors or []
        self.output_preprocessors = output_preprocessors or []
        self.seed = seed

    @staticmethod
    def from_config(config: DictConfig) -> 'ModelConfig':
        return ModelConfig(
            name=config.name,
            model_type=config.type,
            model_shape=config.model_shape,
            input_preprocessors=config.input_preprocessors,
            output_preprocessors=config.output_preprocessors,
            seed=config.seed
        )


class TrainerConfig:
    batch_size: int
    num_epochs: int

    test_sample_size: int

    gen_dataset_type: DatasetType
    n_stacks: int

    seed: int

    def __init__(self,
                 batch_size: int = 64,
                 num_epochs: int = 100,
                 test_sample_size: int = 128,
                 gen_dataset_type: DatasetType = DatasetType.TORCH,
                 n_stacks: int = 1,
                 seed: int = 1):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.test_sample_size = test_sample_size
        self.gen_dataset_type = gen_dataset_type
        self.n_stacks = n_stacks
        self.seed = seed

    @staticmethod
    def from_config(config: DictConfig) -> 'TrainerConfig':
        return TrainerConfig(
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            test_sample_size=config.test_sample_size,
            gen_dataset_type=DatasetType.from_string(config.gen_dataset_type),
            n_stacks=config.n_stacks,
            seed=config.seed
        )


class OptimConfig:
    name: str
    lr: float
    kwargs: dict

    def __init__(self, name: str = 'Adam', lr: float = 0.0001, kwargs: dict | None = None):
        self.name = name
        self.lr = lr
        self.kwargs = kwargs if kwargs is not None else {}

    @staticmethod
    def from_config(config: DictConfig) -> 'OptimConfig':
        return OptimConfig(
            name=config.name,
            lr=config.lr,
            kwargs=config.kwargs if hasattr(config, 'kwargs') else {}
        )


class GeneratorConfig:
    name: str
    z_dim: int
    optim: OptimConfig
    kwargs: dict

    def __init__(self,
                 name: str = 'Generator',
                 z_dim: int = 120,
                 optim: OptimConfig = OptimConfig(),
                 kwargs: dict | None = None
                 ):
        self.name = name
        self.z_dim = z_dim
        self.optim = optim
        self.kwargs = kwargs if kwargs is not None else {}

    @staticmethod
    def from_configs(config: DictConfig) -> 'GeneratorConfig':
        return GeneratorConfig(
            name=config.name,
            z_dim=config.z_dim,
            optim=OptimConfig.from_config(config.optim),
            kwargs=config.kwargs
        )


class CriticConfig:
    name: str
    iterations: int
    optim: OptimConfig
    kwargs: dict

    def __init__(self,
                 name: str = 'Critic',
                 iterations: int = 2,
                 optim: OptimConfig = OptimConfig(),
                 kwargs: dict | None = None
                 ):
        self.name = name
        self.optim = optim
        self.iterations = iterations
        self.kwargs = kwargs if kwargs is not None else {}

    @staticmethod
    def from_configs(config: DictConfig) -> 'CriticConfig':
        return CriticConfig(
            name=config.name,
            iterations=config.iterations,
            optim=OptimConfig.from_config(config.optim),
            kwargs=config.kwargs
        )


class DiffusionConfig:
    noise_steps: int
    beta_start: float
    beta_end: float
    noise_schedule_name: str
    num_classes: int
    c_in: int
    c_out: int
    optim: OptimConfig
    shape_adjustment: str | None = None

    def __init__(self,
                 noise_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 noise_schedule_name: str = 'linear',
                 shape_adjustment: str | None = None,
                 num_classes: int = 10,
                 c_in: int = 3,
                 c_out: int = 3,
                 optim: OptimConfig = OptimConfig(),
                 kwargs: dict | None = None
                 ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule_name = noise_schedule_name
        self.num_classes = num_classes
        self.c_in = c_in
        self.c_out = c_out
        self.optim = optim
        self.kwargs = kwargs if kwargs is not None else {}
        self.shape_adjustment = shape_adjustment

    @staticmethod
    def from_config(config: DictConfig) -> 'DiffusionConfig':
        return DiffusionConfig(
            noise_steps=config.noise_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            noise_schedule_name=config.noise_schedule_name,
            num_classes=config.n_classes,
            c_in=config.c_in,
            c_out=config.c_out,
            shape_adjustment=config.shape_adjustment,
            optim=OptimConfig.from_config(config.optim),
            kwargs=config.kwargs
        )


class GenTrainConfig:
    device: Device
    n_classes: int

    input_shape: Size

    trainer: TrainerConfig
    model: ModelConfig

    epochs_info: Info | None = None

    seed: int

    def __init__(self,
                 device: str = 'cpu',
                 n_classes: int = 2,
                 input_shape: Size = (8, 129, 11),
                 trainer: TrainerConfig = TrainerConfig(),
                 model: ModelConfig = ModelConfig(name='CCWGAN', model_type='gan'),
                 seed: int = 1
                 ):
        self.device: Device = torch.device(device)
        self.n_classes = n_classes
        assert len(input_shape) == 3, f"Expected 3D input shape (channels, height, width), got {input_shape}"
        self.input_shape = input_shape
        self.trainer = trainer
        self.model = model
        self.seed = seed

    def set_shape(self, tensor: Tensor):
        self.input_shape = tensor.shape

    def set_epochs_info(self, epochs_info: Info):
        self.epochs_info = epochs_info

    @property
    def n_eeg_channels(self):
        return self.input_shape[0]
    
    @property
    def n_times(self):
        return self.input_shape[3] if len(self.input_shape) > 3 else self.input_shape[2]

    @property
    def n_input_channels(self):
        return self.input_shape[1] if len(self.input_shape) > 3 else self.input_shape[0]

    def __update_nested(self, obj, params: dict):
        for key, value in params.items():
            if '__' in key:
                nested_key, sub_key = key.split('__', 1)
                self.__update_nested(getattr(obj, nested_key), {sub_key: value})
            else:
                setattr(obj, key, value)

    def update(self, params: dict) -> 'GenTrainConfig':
        self.__update_nested(self, params)
        return self

    @staticmethod
    def from_config(config: DictConfig) -> 'GenTrainConfig':
        return GenTrainConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_classes=config.dataset.n_classes,
            trainer=TrainerConfig.from_config(config.gen.trainer),
            model=ModelConfig.from_config(config.gen.model)
        )


class GanTrainConfig(GenTrainConfig):
    generator: GeneratorConfig
    critic: CriticConfig

    def __init__(self,
                 device: str = 'cpu',
                 n_classes: int = 2,
                 input_shape: Size = (8, 129, 11),
                 trainer: TrainerConfig = TrainerConfig(),
                 model: ModelConfig = ModelConfig(name='CCWGAN', model_type='gan'),
                 generator: GeneratorConfig = GeneratorConfig(),
                 critic: CriticConfig = CriticConfig(),
                 seed: int = 1
                 ):
        super().__init__(device, n_classes, input_shape, trainer, model, seed)
        self.generator = generator
        self.critic = critic

    @staticmethod
    def from_config(config: DictConfig) -> 'GanTrainConfig':
        return GanTrainConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_classes=config.dataset.n_classes,
            input_shape=config.dataset.input_shape,
            trainer=TrainerConfig.from_config(config.gen.trainer),
            generator=GeneratorConfig.from_configs(config.gen.model.generator),
            critic=CriticConfig.from_configs(config.gen.model.critic),
            model=ModelConfig.from_config(config.gen.model),
            seed=config.seed
        )


class DiffusionTrainConfig(GenTrainConfig):

    diffusion: DiffusionConfig

    def __init__(self,
                 device: str = 'cpu',
                 n_classes: int = 2,
                 input_shape: Size = (8, 129, 11),
                 trainer: TrainerConfig = TrainerConfig(),
                 model: ModelConfig = ModelConfig(name='CDDPM', model_type=''),
                 diffusion: DiffusionConfig = DiffusionConfig(),
                 seed: int = 1):
        super().__init__(device, n_classes, input_shape, trainer, model, seed)
        self.diffusion = diffusion

    @staticmethod
    def from_config(config: DictConfig) -> 'DiffusionTrainConfig':
        return DiffusionTrainConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            n_classes=config.dataset.n_classes,
            input_shape=config.dataset.input_shape,
            trainer=TrainerConfig.from_config(config.gen.trainer),
            diffusion=DiffusionConfig.from_config(config.gen.model.diffusion),
            model=ModelConfig.from_config(config.gen.model),
            seed=config.seed
        )

class TrainResult:
    epoch: int
    batch_idx: int
    start_time: datetime
    end_time: datetime
    duration: float
    loss: float
    losses: dict

    def __init__(self,
                 loss: float = 0.0,
                 epoch: int = 0,
                 batch_idx: int = 0,
                 losses: dict | None = None):
        self.loss = loss
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.start_time = datetime.now()
        self.losses = {} if losses is None else losses

    def start(self):
        self.start_time = datetime.now()

    def finish(self):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()


class GanTrainResult(TrainResult):
    loss_critic: float
    loss_generator: float

    def __init__(self,
                 loss: float = 0.0,
                 loss_critic: float = 0.0,
                 loss_generator: float = 0.0,
                 epoch: int = 0,
                 batch_idx: int = 0):
        super().__init__(loss, epoch, batch_idx)
        self.loss_critic = loss_critic
        self.loss_generator = loss_generator


class Training:
    config: GenTrainConfig
    history: list[TrainResult]
    result: TrainResult | None

    start_time: datetime
    end_time: datetime
    duration: float

    def __init__(self, config: GenTrainConfig):
        self.config = config
        self.history = []
        self.start_time = datetime.now()

    def start(self):
        self.start_time = datetime.now()

    def finish(self):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()

    def finish_epoch(self):
        self.result.finish()
        self.history.append(self.result)

    def start_epoch(self, epoch: int = 0) -> 'Training':
        self.result = TrainResult(epoch=epoch)
        return self


class GanTraining(Training):
    config: GanTrainConfig

    history: list[GanTrainResult]
    result: GanTrainResult | None

    def __init__(self, config: GanTrainConfig):
        super().__init__(config)

    def start_epoch(self, epoch: int = 0) -> 'Training':
        self.result = GanTrainResult(epoch=epoch)
        return self

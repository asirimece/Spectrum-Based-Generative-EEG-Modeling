from lib.gen.model import GanTrainConfig, GenTrainConfig, DiffusionTrainConfig
from omegaconf import DictConfig


def get_train_config(config: DictConfig) -> GanTrainConfig | GenTrainConfig:
    pipeline_config = config.experiment.pipeline
    model_config = pipeline_config.model
    match model_config.type:
        case "gan":
            return GanTrainConfig.from_config(config)
        case "diffusion":
            return DiffusionTrainConfig.from_config(config)
        case _:
            raise ValueError(f"Unknown model type {model_config.type}")

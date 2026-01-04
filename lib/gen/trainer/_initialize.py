
from lib.gen.model import GenTrainConfig, GanTrainConfig, DiffusionTrainConfig
from lib.gen.models._utils import get_model
from lib.dataset import TorchBaseDataset
from lib.experiment.evaluation.run import RunVisualizer, RunEvaluator
from typing import Callable
from lib.utils import empty_func
from lib.logging import get_logger

logger = get_logger()

def get_trainer(
        config: GenTrainConfig | GanTrainConfig | DiffusionTrainConfig,
        dataset: TorchBaseDataset,
        visualizers: list[RunVisualizer] | None = None,
        evaluators: list[RunEvaluator] | None = None,
        call_hook: Callable = empty_func,
        inputs: tuple = None
):
    logger.info(f"Initializing trainer for model: {config.model.name}.")
    
    model = get_model(config)

    if isinstance(config, GanTrainConfig):
        from lib.gen.trainer.gan import GanTrainer  # Delayed import
        logger.info("Configuring GAN trainer for 2D spectrogram data.")
        return GanTrainer(model, dataset, visualizers, evaluators, call_hook, inputs=inputs)
    elif isinstance(config, DiffusionTrainConfig):
        from lib.gen.trainer.diffusion import DiffusionTrainer  # Delayed import
        logger.info("Configuring Diffusion trainer.")
        return DiffusionTrainer(config, model, dataset, visualizers, evaluators, call_hook)
    else:
        raise ValueError(f"Unsupported config type {config}")

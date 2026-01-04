from lib.dataset import load_dataset_by_name, BaseDataset
from lib.experiment.pipeline import AugPipeline, NeuralAugPipeline
from lib.experiment.base import Experiment
from lib.experiment.runner import ExperimentRunner
from omegaconf import DictConfig
from lib.utils import setup_environment
from lib.logging import get_logger
from typing import Tuple


def build_pipeline(config: DictConfig, dataset: BaseDataset) -> AugPipeline:
    logger = get_logger()
    pipeline: AugPipeline = AugPipeline() if config.experiment.pipeline.type != 'neural' else NeuralAugPipeline()
    (pipeline
     .set_dataset(dataset)
     .add_hooks(config.pipeline.hooks)
     .add_steps(config.pipeline.steps)
     .add_evaluators(config.pipeline.evaluators)
     .add_visualizers(config.pipeline.visualizers)
     .build())
    logger.info(f"Pipeline built: {pipeline.name}")
    return pipeline


def get_epochs_limits(config: DictConfig) -> Tuple[int, int]:
    t_min = config.dataset.raw_t_min if config.dataset.raw_t_min is not None else config.dataset.t_min
    t_max = config.dataset.raw_t_max if config.dataset.raw_t_max is not None else config.dataset.t_max
    return t_min, t_max


def build_dataset(config: DictConfig) -> BaseDataset:
    dataset: BaseDataset = load_dataset_by_name(config.dataset.name, local_path=config.dataset.home)
    dataset.set_classes(config.dataset.classes)
    t_min, t_max = get_epochs_limits(config)
    dataset.set_epoch_limits(t_min, t_max)
    dataset.add_raw_preprocessors(config.pipeline.raw_preprocessors)
    epochs_preprocessors = config.pipeline.epochs_preprocessors
    if epochs_preprocessors is not None and len(epochs_preprocessors) > 0:
        dataset.add_epochs_transforms(epochs_preprocessors)
    return dataset


def run(config: DictConfig):
    setup_environment(config)
    dataset: BaseDataset = build_dataset(config)
    experiment: Experiment = Experiment.from_yaml_config(config, dataset)
    pipeline: AugPipeline = build_pipeline(config, dataset)
    runner: ExperimentRunner = ExperimentRunner(experiment, pipeline, config.seed)
    runner.initialize_from_config(config)
    runner.run()

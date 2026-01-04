from lib.dataset import TorchBaseDataset, load_torch_dataset_by_name, DatasetType
from lib.experiment.pipeline import EEGenPipeline
from omegaconf import DictConfig
from lib.utils import setup_environment
from lib.experiment.base import Experiment
from lib.experiment.runner import ExperimentRunner
from lib.gen import get_train_config
from lib.logging import get_logger

logger = get_logger()


def get_dataset(config: DictConfig) -> TorchBaseDataset:
    dataset_type = DatasetType.from_string(config.dataset.gen_dataset_type)
    dataset: TorchBaseDataset = load_torch_dataset_by_name(name=config.dataset.name,
                                                           local_path=config.dataset.home,
                                                           dataset_type=dataset_type)
    (dataset
     .set_classes(config.dataset.classes)
     .set_epoch_limits(config.dataset.t_min, config.dataset.t_max)
     .add_raw_preprocessors(config.experiment.pipeline.raw_preprocessors))
    logger.info(f"=== Dataset Created: {config.dataset.name} ===")
    logger.info(f"-- Dataset class names: {dataset.class_names}")
    logger.info(f"-- Dataset available subjects: {dataset.available_subjects()}")
    return dataset


def build_pipeline(config: DictConfig, dataset: TorchBaseDataset):
    training_config = get_train_config(config)
    pipeline: EEGenPipeline = EEGenPipeline()
    (pipeline
     .set_dataset(dataset)
     .set_transforms_config(config.gen.model.transforms)
     .set_post_transforms_config(config.gen.model.output_preprocessors)
     .set_gen_config(training_config)
     .add_hooks(config.pipeline.hooks)
     .add_steps(config.pipeline.steps)
     .add_evaluators(config.pipeline.evaluators)
     .add_visualizers(config.pipeline.visualizers)
     .build())
    return pipeline


def run(config: DictConfig):
    setup_environment(config)
    dataset: TorchBaseDataset = get_dataset(config)
    experiment: Experiment = Experiment.from_yaml_config(config, dataset)
    pipeline: EEGenPipeline = build_pipeline(config, dataset)
    runner: ExperimentRunner = ExperimentRunner(experiment, pipeline, config.seed)
    runner.initialize_from_config(config)
    runner.run()

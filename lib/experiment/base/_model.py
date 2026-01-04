from lib.experiment._model import ExperimentType, ExperimentMode, ExperimentSubMode
from lib.experiment.run.generator import (
    RunConfigGenerator,
    IntraSubjectRunConfigGenerator,
    CVRunConfigGenerator,
    LOOCVSubjectRunConfigGenerator,
    AugmentationTuningRunConfigGenerator,
    ModelTuningRunConfigGenerator,
    BlockRunConfigGenerator,
    FractionalRunConfigGenerator,
    SingleRunConfigGenerator,
    EnsembleRunConfigGenerator
)
from lib.dataset import BaseDataset, load_dataset_by_name
from lib.experiment._utils import get_random_name
from lib.experiment.model import ModelTuningConfig
from typing import List, Dict
from datetime import datetime
from lib.experiment.augmentation import AugmentationInfo, VisualisationConfig
from lib.experiment.evaluation.result import VisualResult
from omegaconf import DictConfig
from lib.config import config_to_primitive
from lib.experiment.run.base import Run, RunConfig
from lib.logging import get_logger
from pathlib import Path
import json
import os
import numpy as np

logger = get_logger()


class ExperimentPlan:
    configs: List[RunConfig] = []
    finished: List[RunConfig] = []

    structure: Dict[str, int] = {}

    def __init__(self, configs: List[RunConfig] = None, finished: List[RunConfig] = None):
        if configs is not None:
            self.configs = configs
        if finished is not None:
            self.finished = finished

    def __iter__(self):
        return self

    def __next__(self) -> RunConfig:
        next_config = self.next()
        if next_config is None:
            raise StopIteration
        return next_config

    @property
    def number_of_configs(self) -> int:
        return len(self.configs) + len(self.finished)

    def next(self) -> RunConfig | None:
        if len(self.configs) == 0:
            return None
        return self.configs[0]

    def finish(self):
        if len(self.configs) > 0:
            self.finished.append(self.configs.pop(0))

    def to_dict(self):
        return {
            "configs": [config.to_dict() for config in self.configs],
            "finished": [config.to_dict() for config in self.finished],
            "structure": self.structure
        }


class ExperimentConfig:
    subjects: List[int] = []
    sessions: List[int] = []
    cv_folds: int = 5

    def __init__(self, subjects: List[int] = None, sessions: List[int] = None, cv_folds: int = 5):
        if subjects is not None:
            self.subjects = subjects
        if sessions is not None:
            self.sessions = sessions
        self.cv_folds = cv_folds

    def to_dict(self):
        return {
            "subjects": self.subjects,
            "sessions": self.sessions,
            "cv_folds": self.cv_folds
        }

    @staticmethod
    def from_yaml_config(config: DictConfig) -> 'ExperimentConfig':
        return ExperimentConfig(
            subjects=config_to_primitive(config.subjects),
            sessions=config_to_primitive(config.sessions),
            cv_folds=config.cv_folds
        )


class Experiment:
    name: str
    hash_str: str
    description: str

    mode: str | ExperimentMode
    sub_mode: str | ExperimentSubMode
    e_type: str | ExperimentType
    dataset: str

    subjects: List[int] = []
    sessions: List[int] = []
    cv_folds: int = 5

    partial_result_key: str | None = None

    # Currently saved separately to reduce memory usage
    runs: List[Run] = []

    # Changed to augmentation to reduce number of iterations
    # If you want to apply additional augmentations store it in separate filed
    augmentations: List[AugmentationInfo] = []

    tuning_augmentation: AugmentationInfo = None

    _storage_path: str
    date_created: str | datetime

    config: Dict = {}

    plan: ExperimentPlan = None

    metrics: Dict[str, float] = {}
    media: List[VisualResult] | List[Dict[str, str]] = []

    start_time: datetime = None
    end_time: datetime = None
    duration: float | None = None

    def __init__(self,
                 mode: str | ExperimentMode,
                 sub_mode: str | ExperimentSubMode,
                 e_type: str | ExperimentType,
                 dataset: str,
                 name: str = None,
                 description: str | None = None,
                 storage_path: str | None = './work_dirs',
                 date_created: str | datetime | None = None,
                 subjects: List[int] = None,
                 sessions: List[int] = None,
                 cv_folds: int = 5,
                 partial_result_key: str = None):
        self.name = name if name is not None else get_random_name()
        self.hash_str = get_random_name(name_type="hash")
        self.description = description
        self.mode = mode
        self.sub_mode = sub_mode
        self.e_type = e_type
        self.dataset = dataset
        self._storage_path = storage_path
        self.date_created = date_created if date_created is not None else datetime.now()
        self.subjects = subjects if subjects is not None else []
        self.sessions = sessions if sessions is not None else []
        self.cv_folds = cv_folds
        self.partial_result_key = partial_result_key

    @property
    def storage_path(self) -> str:
        return f"{self._storage_path}/{self.name}"

    def metrics_to_dict(self):
        for key, value in self.metrics.items():
            if not isinstance(value, dict) and hasattr(value, "to_dict"):
                self.metrics[key] = value.to_dict()
        return self.metrics

    def media_to_dict(self) -> Dict[str, any]:
        if len(self.media) == 0:
            return {}
        if isinstance(self.media[0], VisualResult):
            return {item.name: item.data for item in self.media}
        else:
            return {item['name']: item["path"] for item in self.media}

    def to_dict(self):
        return {
            "name": self.name,
            "hash_str": self.hash_str,
            "description": self.description,
            "mode": str(self.mode),
            "sub_mode": str(self.sub_mode),
            "e_type": str(self.e_type),
            "subjects": self.subjects,
            "sessions": self.sessions,
            "cv_folds": self.cv_folds,
            "partial_result_key": self.partial_result_key,
            "dataset": self.dataset,
            "date_created": self.date_created if isinstance(self.date_created, str) else self.date_created.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "duration": self.duration,
            # "runs": [run.to_dict() for run in self.runs],
            "augmentations": [aug.to_dict() for aug in self.augmentations],
            "tuning_augmentation": self.tuning_augmentation.to_dict() if self.tuning_augmentation is not None else None,
            "storage_path": str(self.storage_path),
            "plan": self.plan.to_dict(),
            "config": self.config,
            "metrics": self.metrics_to_dict()
        }

    def to_summary_dict(self):
        return {
            "name": self.name,
            "hash_str": self.hash_str,
            "description": self.description,
            "mode": str(self.mode),
            "sub_mode": str(self.sub_mode),
            "e_type": str(self.e_type),
            "subjects": self.subjects,
            "sessions": self.sessions,
            "dataset": self.dataset,
            "date_created": self.date_created if isinstance(self.date_created, str) else self.date_created.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "duration": self.duration,
            "config": self.config,
        }

    def __save_media(self) -> List[Dict[str, str]]:
        saved_media = []
        if self.media is not None:
            for item in self.media:
                if isinstance(item, VisualResult):
                    path = Path(self.storage_path)
                    filepath = item.save(path)
                    saved_media.append({"name": item.name, "path": str(filepath)})
                elif isinstance(item, dict):
                    saved_media.append(item)
        return saved_media

    def start(self):
        logger.info(f"Running experiment {self.name}")
        self.start_time = datetime.now()

    def finish(self):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Finished experiment {self.name} in {self.duration} seconds")

    def save(self):
        if self.storage_path is None:
            raise ValueError("Storage path is not set -> can't save experiment")
        else:
            storage_path = Path(self.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            experiment_dict = self.to_dict().copy()
            experiment_dict["media"] = self.__save_media()
            with open(storage_path.joinpath("experiment.json"), "w") as f:
                f.write(json.dumps(experiment_dict, indent=4))

    @staticmethod
    def from_yaml_config(config: DictConfig, dataset: BaseDataset | None = None) -> 'Experiment':
        logger.info(f"Building experiment from config")
        e_type = ExperimentType.from_string(config.experiment.type)
        partial_result_key = get_partial_result_key(e_type)
        experiment = Experiment(
            name=config.experiment.name,
            dataset=config.experiment.dataset.name,
            mode=ExperimentMode.from_string(config.experiment.mode),
            sub_mode=ExperimentSubMode.from_string(config.experiment.sub_mode),
            e_type=e_type,
            storage_path=f"{os.getcwd()}/{config.experiment.storage_path}",
            date_created=datetime.now(),
            subjects=config_to_primitive(config.experiment.subjects),
            sessions=config_to_primitive(config.experiment.sessions),
            cv_folds=config.experiment.cv_folds,
            partial_result_key=partial_result_key
        )
        experiment.description = build_experiment_description(experiment)
        experiment.plan = Experiment.plan_from_yaml_config(config, experiment, dataset)
        experiment.augmentations = Experiment.augmentations_from_yaml_config(config)
        experiment.tuning_augmentation = Experiment.tuning_augmentation_from_yaml_config(config)
        experiment.config = config_to_primitive(config)
        logger.info(f"Built experiment {experiment.name}")
        return experiment

    @staticmethod
    def plan_from_yaml_config(config: DictConfig,
                              experiment: 'Experiment',
                              dataset: BaseDataset | None = None) -> ExperimentPlan:
        logger.info(f"Building experiment plan from config")
        plan = ExperimentPlan()
        plan.configs, plan.structure = Experiment.run_configs_from_yaml_config(config, experiment, dataset)
        return plan

    @staticmethod
    def run_configs_from_yaml_config(
            config: DictConfig,
            experiment: 'Experiment',
            dataset: BaseDataset | None = None
    ) -> (List[RunConfig], Dict[str, int]):
        e_type = ExperimentType.from_string(config.experiment.type)
        mode = ExperimentMode.from_string(config.experiment.mode)
        sub_mode = ExperimentSubMode.from_string(config.experiment.sub_mode)
        dataset: BaseDataset = load_dataset_by_name(config.experiment.dataset.name) if dataset is None else dataset
        generator: RunConfigGenerator | None = None
        model_tuning_config = ModelTuningConfig.from_config(config.experiment.pipeline.model.tuning)
        ensemble_size = config.experiment.ensemble_size
        seed = config.seed

        context = RunConfig(
            experiment_name=experiment.name,
            experiment_hash=experiment.hash_str,
            work_dir=Path(experiment.storage_path),
            mode=mode,
            sub_mode=sub_mode,
            e_type=e_type,
            subjects=config_to_primitive(config.experiment.subjects),
            sessions=config_to_primitive(config.experiment.sessions),
            load_balanced=config.experiment.load_balanced,
            augmentations=Experiment.augmentations_from_yaml_config(config),
            tuning_augmentation_config=Experiment.tuning_augmentation_from_yaml_config(config),
            model_tuning=model_tuning_config,
            ensemble_size=ensemble_size,
            seed=config.seed
        )

        if e_type.value.lower().startswith("gen"):
            generator = Experiment.get_gen_model_tuning_generator(generator, e_type, sub_mode=sub_mode)
        elif e_type.value.lower().startswith("base"):
            generator = Experiment.get_base_generator(generator, e_type=e_type, mode=mode, sub_mode=sub_mode)
        else:
            if mode == ExperimentMode.INTRA_SUBJECT:
                cv_folds = config.experiment.cv_folds
                generator = Experiment.get_intra_subject_run_config_generator(context, generator, e_type, cv_folds,
                                                                              sub_mode, ensemble_size, seed)
            elif mode == ExperimentMode.CROSS_SUBJECT:
                generator = Experiment.get_cross_subject_run_config_generator(context, generator, e_type)
            else:
                raise ValueError(f"Unknown experiment")

        logger.info(f"Generating run configs")
        configs, _ = generator.generate(context, dataset)
        return configs, generator.structure

    @staticmethod
    def get_intra_subject_run_config_generator(
            context: RunConfig,
            generator: RunConfigGenerator = None,
            e_type: ExperimentType = ExperimentType.AUG_TRAIN_TEST,
            n_cv_folds: int = 5,
            sub_mode: ExperimentSubMode = ExperimentSubMode.DEFAULT,
            ensemble_size: int | None = None,
            seed: int = 1,
    ) -> RunConfigGenerator:

        if e_type == ExperimentType.AUG_TRAIN_VALID_TEST:
            generator = CVRunConfigGenerator(
                train_name="train",
                test_name="valid",
                child=generator, n_splits=n_cv_folds,
                random_state=seed
            )

        if sub_mode == ExperimentSubMode.BLOCK_BASED:
            generator = BlockRunConfigGenerator(child=generator)
        elif sub_mode == ExperimentSubMode.FRACTIONAL:
            generator = FractionalRunConfigGenerator(child=generator)
        else:
            generator = CVRunConfigGenerator(child=generator, n_splits=n_cv_folds, random_state=seed)

        if e_type == ExperimentType.AUG_AUGMENTATION_TUNING:
            generator = AugmentationTuningRunConfigGenerator(child=generator)
        elif e_type == ExperimentType.AUG_MODEL_TUNING:
            generator = ModelTuningRunConfigGenerator(child=generator, context=context)

        generator = IntraSubjectRunConfigGenerator(child=generator)

        if ensemble_size is not None and ensemble_size > 1:
            generator = EnsembleRunConfigGenerator(child=generator, ensemble_size=ensemble_size)

        return generator

    @staticmethod
    def get_cross_subject_run_config_generator(
            context: RunConfig,
            generator: RunConfigGenerator = None,
            e_type: ExperimentType = ExperimentType.AUG_TRAIN_TEST
    ) -> RunConfigGenerator:

        generator = LOOCVSubjectRunConfigGenerator(child=generator)

        if e_type == ExperimentType.AUG_AUGMENTATION_TUNING:
            generator = AugmentationTuningRunConfigGenerator(child=generator)
        elif e_type == ExperimentType.AUG_MODEL_TUNING:
            generator = ModelTuningRunConfigGenerator(child=generator, context=context)

        return generator

    @staticmethod
    def get_gen_model_tuning_generator(
            generator: RunConfigGenerator = None,
            e_type: ExperimentType = ExperimentType.GEN_MODEL_TUNING,
            sub_mode: ExperimentSubMode = ExperimentSubMode.DEFAULT) -> RunConfigGenerator:

        if sub_mode == ExperimentSubMode.BLOCK_BASED:
            generator = BlockRunConfigGenerator(child=generator, train_only=True)
        elif sub_mode == ExperimentSubMode.FRACTIONAL:
            generator = FractionalRunConfigGenerator(child=generator, train_only=True)

        if e_type == ExperimentType.GEN_MODEL_TUNING:
            return ModelTuningRunConfigGenerator(child=generator)
        else:
            generator = SingleRunConfigGenerator(child=generator)
        return generator

    @staticmethod
    def get_base_generator(
            generator: RunConfigGenerator = None,
            e_type: ExperimentType = ExperimentType.BASE_TRAIN_TEST,
            mode: ExperimentMode = ExperimentMode.CROSS_SUBJECT,
            sub_mode: ExperimentSubMode = ExperimentSubMode.DEFAULT) -> RunConfigGenerator:

        if sub_mode == ExperimentSubMode.BLOCK_BASED:
            generator = BlockRunConfigGenerator(child=generator)
        elif sub_mode == ExperimentSubMode.FRACTIONAL:
            generator = FractionalRunConfigGenerator(child=generator)

        generator = SingleRunConfigGenerator(child=generator)

        if mode == ExperimentMode.INTRA_SUBJECT:
            generator = IntraSubjectRunConfigGenerator(child=generator)

        return generator

    @staticmethod
    def augmentations_from_yaml_config(config: DictConfig) -> List[AugmentationInfo]:
        augmentations = []
        augmentation_configs = config.experiment.augmentations
        for aug in augmentation_configs:
            augmentation = AugmentationInfo(
                name=aug.name,
                kwargs=config_to_primitive(aug.kwargs)
            )
            augmentations.append(augmentation)
        return augmentations

    @staticmethod
    def tuning_augmentation_from_yaml_config(config: DictConfig) -> AugmentationInfo | None:
        yaml_config = config.experiment.tuning_augmentation
        if yaml_config is None:
            return None
        else:
            return AugmentationInfo(
                name=yaml_config.name,
                kwargs=config_to_primitive(yaml_config.kwargs),
                optimization_parameter=yaml_config.optimization_parameter,
                op_neutral_value=yaml_config.op_neutral_value,
                search=config_to_primitive(yaml_config.search),
                visualisation_config=Experiment.augmentation_visualisation_config_from_yaml_config(config)
            )

    @staticmethod
    def augmentation_visualisation_config_from_yaml_config(config: DictConfig) -> VisualisationConfig | None:
        augmentation_config = config.experiment.tuning_augmentation
        if augmentation_config is None:
            return None
        visualisation_config = augmentation_config.visualisation_config
        if visualisation_config is None:
            return None
        else:
            return VisualisationConfig(
                name=visualisation_config.name,
                label=visualisation_config.label,
                metric=visualisation_config.metric,
                metric_label=visualisation_config.metric_label,
                kwargs=config_to_primitive(visualisation_config.kwargs)
            )


def build_experiment_description(experiment: Experiment) -> str:
    augmentations = "-".join([aug.name for aug in experiment.augmentations])
    return f"{experiment.name} - {experiment.mode} - {experiment.dataset} - {augmentations}"


def get_partial_result_key(e_type: ExperimentType) -> str | None:
    if e_type == ExperimentType.AUG_AUGMENTATION_TUNING:
        return "augmentation"
    elif e_type == ExperimentType.AUG_MODEL_TUNING or e_type == ExperimentType.GEN_MODEL_TUNING:
        return "tuning"
    else:
        return None

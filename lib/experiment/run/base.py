from .._model import ExperimentType, ExperimentMode, Subset, ExperimentSubMode
from ..augmentation import AugmentationInfo
from typing import List, Dict
from datetime import datetime
from numpy import ndarray
from .._utils import get_random_name
import copy
from pathlib import Path
import json
from lib.experiment.evaluation.result import VisualResult
from lib.experiment.model import ModelTuningConfig
from skorch.history import History
from lib.logging import get_logger


logger = get_logger()


class RunConfig:
    name: str

    experiment_name: str
    experiment_hash: str

    idx: Dict[str, int] = None

    mode: str | ExperimentMode
    sub_mode: str | ExperimentSubMode
    e_type: str | ExperimentType

    subset: str = 'train'

    __experiment_path: str | Path

    subjects: List[int] = []
    sessions: List[int] = []
    blocks: List[int] | None = None
    subset_indexes: ndarray | List | None = None
    train_fraction: float | None = None
    load_balanced: bool = False
    props: List[Dict[str, str | int | float | bool]] = []

    test_subset: str | None = None
    test_subjects: List[int] = []
    test_sessions: List[int] = []
    test_blocks: List[int] | None = None

    tuning_augmentation_config: AugmentationInfo = None
    augmentations: List[AugmentationInfo] = []

    model_tuning: ModelTuningConfig | None
    pipeline_params: Dict | None = None

    seed: int | None = None

    def __init__(self,
                 mode: str | ExperimentMode = ExperimentMode.INTRA_SUBJECT,
                 sub_mode: str | ExperimentSubMode = ExperimentSubMode.DEFAULT,
                 e_type: str | ExperimentType = ExperimentType.AUG_AUGMENTATION_TUNING,
                 name: str | None = None,
                 experiment_name: str | None = None,
                 experiment_hash: str | None = None,
                 work_dir: str | Path | None = None,
                 subjects: List[int] | None = None,
                 sessions: List[int] | None = None,
                 blocks: List[int] | None = None,
                 subset_indexes: ndarray | List | None = None,
                 train_fraction: float | None = None,
                 load_balanced: bool = False,
                 props: List[Dict[str, str | int | float | bool]] | None = None,
                 augmentations: List[AugmentationInfo] | None = None,
                 tuning_augmentation_config: AugmentationInfo | None = None,
                 model_tuning: ModelTuningConfig | None = None,
                 pipeline_params: Dict | None = None,
                 ensemble_size: int | None = None,
                 seed: int | None = None
                 ):
        self.name = get_random_name() if name is None else name
        self.experiment_name = experiment_name
        self.experiment_hash = experiment_hash
        self.__experiment_path = Path(work_dir) if work_dir is not None else Path.cwd()
        self.mode = mode
        self.sub_mode = sub_mode
        self.e_type = e_type
        self.subjects = subjects if subjects is not None else []
        self.sessions = sessions if sessions is not None else []
        self.blocks = blocks
        self.subset_indexes = subset_indexes
        self.train_fraction = train_fraction
        self.load_balanced = load_balanced
        self.props = props if props is not None else []
        self.augmentations = augmentations if augmentations is not None else []
        self.tuning_augmentation_config = tuning_augmentation_config
        self.model_tuning = model_tuning
        self.pipeline_params = pipeline_params
        self.ensemble_size = ensemble_size
        self.seed = seed

    def copy(self, new_name: bool = True) -> 'RunConfig':
        config = copy.deepcopy(self)
        if new_name:
            config.name = get_random_name()
        return config

    @property
    def work_dir(self) -> str | Path:
        return Path(self.__experiment_path).joinpath("runs", self.name)

    def get_test_config(self) -> 'RunConfig':
        test_config = self.copy(new_name=False)
        test_config.subset = self.test_subset
        test_config.subjects = self.test_subjects
        test_config.sessions = self.test_sessions
        test_config.blocks = self.test_blocks
        return test_config

    def to_dict(self):
        config_dict = {
            "name": self.name,
            "mode": str(self.mode),
            "sub_mode": str(self.sub_mode),
            "e_type": str(self.e_type),
            "work_dir": str(self.work_dir),
            "subjects": self.subjects,
            "sessions": self.sessions,
            "blocks": self.blocks,
            "train_fraction": self.train_fraction,
            "load_balanced": self.load_balanced,
            "experiment_name": self.experiment_name,
            "experiment_hash": self.experiment_hash,
            "props": self.props,
            "augmentations": [aug.to_dict() for aug in self.augmentations],
            "tuning_augmentation_config": (self.tuning_augmentation_config.to_dict()
                                           if self.tuning_augmentation_config is not None else None),
            "model_tuning": (self.model_tuning.to_dict() if self.model_tuning is not None else None),
            "pipeline_params": self.pipeline_params,
            "ensemble_size": self.ensemble_size,
            "seed": self.seed
        }
        config_dict.update({f"idx_{key}": value for key, value in self.idx.items()})
        return config_dict




class Run:
    name: str

    idx: Dict[str, int] = None
    subset: str | Subset

    config: RunConfig

    props: Dict[str, str | int | float | bool] = {}
    tags: List[str] = []

    metrics: Dict[str, float] = {}
    media: List[Dict[str, str]] | List[VisualResult] = []
    history: History | list | None = None

    model: any = None

    date_created: str | datetime
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float | None = None

    def __init__(self,
                 name: str,
                 idx: Dict[str, int],
                 subset: str,
                 config: RunConfig,
                 metrics: Dict[str, float] | None = None,
                 media: List[Dict[str, str]] | None = None,
                 tags: List[str] | None = None,
                 props: Dict[str, str | int | float | bool] | None = None,
                 date_created: str | datetime | None = None,
                 history: History | list | None = None,
                 model: any = None
                 ):
        self.name = name
        self.idx = idx
        self.subset = subset
        self.config = config
        self.metrics = metrics if metrics is not None else {}
        self.media = media if media is not None else []
        self.tags = tags if tags is not None else []
        self.props = props if props is not None else {}
        self.date_created = date_created if date_created is not None else datetime.now()
        self.history = history
        self.model = model

    @staticmethod
    def from_run_config(config: RunConfig) -> 'Run':
        return Run(
            name=config.name,
            idx=config.idx,
            subset=config.subset,
            config=config
        )

    def __save_media(self) -> List[Dict[str, str]]:
        saved_media = []
        if self.media is not None:
            for item in self.media:
                if isinstance(item, VisualResult):
                    path = Path(self.config.work_dir)
                    filepath = item.save(path)
                    saved_media.append({"name": item.name, "path": str(filepath)})
                elif isinstance(item, dict):
                    saved_media.append(item)
        return saved_media

    def start(self) -> 'Run':
        if self.config is None:
            logger.info(f"Starting run {self.name}")
        else:
            logger.info(f"Starting run {self.name} in mode {self.config.mode} for subset {self.config.subset}")
        self.start_time = datetime.now()
        return self

    def finish(self) -> 'Run':
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Finished run {self.name} in {self.duration} seconds")
        return self

    def save(self) -> 'Run':
        if self.config is not None and self.config.work_dir is not None:
            work_dir = Path(self.config.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            dict_run = self.to_dict().copy()
            self.media = self.__save_media()
            dict_run['media'] = self.media
            with open(work_dir.joinpath('run.json'), 'w') as f:
                f.write(json.dumps(dict_run, indent=4))
            return self
        else:
            raise ValueError("Cannot save run without config and work_dir.")

    def load_media(self) -> List[VisualResult]:
        media = []
        for i, item in enumerate(self.media):
            if isinstance(item, dict):
                media.append(VisualResult.from_dict(item))
            elif isinstance(item, VisualResult):
                media.append(item)
            else:
                logger.error(f"Media item {i} is not a valid type.")
        return media

    def to_dict(self):
        return {
            "name": self.name,
            "idx": self.idx,
            "subset": str(self.subset),
            "date_created": self.date_created if isinstance(self.date_created, str) else self.date_created.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time is not None else None,
            "end_time": self.end_time.isoformat() if self.end_time is not None else None,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "media": self.media,
            "model": self.model,
            "history": self.history,
            "tags": self.tags,
            "props": self.props,
            "duration": self.duration
        }

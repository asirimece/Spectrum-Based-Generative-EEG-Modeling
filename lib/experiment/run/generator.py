from typing import List, Dict
from lib.dataset import BaseDataset
from lib.experiment.run.base import RunConfig
from sklearn.model_selection import StratifiedKFold
from numpy import ndarray
from lib.experiment.augmentation import AugmentationInfo
from lib.config import resolve_search_list
from lib.experiment import get_random_name
import numpy as np
from lib.experiment.model import ModelTuningConfig, get_iterations_from_config
import math


class RunConfigGenerator:
    name: str
    child: any

    structure: Dict[str, int] | None = None
    idx: Dict[str, int] | None = None

    def __init__(self, name: str, child=None):
        self.name = name
        self.child = child

    def create_structure(self,
                         parent_idx: int | None = None,
                         structure: Dict[str, int] | None = None) -> Dict[str, int]:
        idx = parent_idx + 1 if parent_idx is not None else 0
        if structure is not None:
            structure[self.__get_structure_name(structure)] = idx
        else:
            structure = {self.name: idx}
        if self.child is not None:
            return self.child.create_structure(idx, structure)
        else:
            return structure

    # Recursive function which increased the name of the own level by 1 if it is already in the structure
    def __get_structure_name(self, structure: Dict[str, int]):
        if self.name in structure:
            if self.name[-1].isdigit():
                self.name = self.name[:-1] + str(int(self.name[-1]) + 1)
            else:
                self.name = self.name + "_2"
            self.__get_structure_name(structure)
        return self.name

    def __initial_idx_from_structure(self):
        if self.structure is not None:
            idx = self.structure.copy()
            for key in idx:
                idx[key] = 0
            self.idx = idx

    def __setup(self):
        self.structure = self.create_structure()
        self.__initial_idx_from_structure()

    def _init_config_from_context(self, context: RunConfig, name_suffix: str | None = None) -> RunConfig:
        config = context.copy()
        config.idx = self.idx.copy()
        name = config.experiment_hash if config.experiment_hash is not None else config.name
        name_suffix = name_suffix if name_suffix is not None else f"{self.name}_{get_random_name('number')}"
        config.name = f"{name}_{idx_to_string(self.idx)}_{name_suffix}"
        return config

    def _init_test_config_from_train(self, train_config: RunConfig, name_suffix: str | None = None) -> RunConfig:
        train_config.test_subset = name_suffix
        train_config.test_subjects = train_config.subjects
        train_config.test_sessions = train_config.sessions
        train_config.test_blocks = train_config.blocks
        return train_config

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):
        if parent_idx is None:
            self.__setup()
        else:
            self.idx = parent_idx
        pass


class AugmentationTuningRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    def __init__(self, child: RunConfigGenerator | None = None):
        super().__init__(name='augmentation', child=child)

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        tuning_augmentations = get_iterations_from_tuning_augmentation(context.tuning_augmentation_config)
        configs = []
        for idx, ta in enumerate(tuning_augmentations):
            self.idx[self.name] = idx
            config = self._init_config_from_context(context)
            config.tuning_augmentation_config = ta
            if self.child is not None:
                child_configs, self.idx = self.child.generate(config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(config)
        return configs, self.idx


class ModelTuningRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None
    tuning_iterations: List[Dict | None] = [None]

    def __init__(self, child: RunConfigGenerator | None = None, context: RunConfig = None):
        if context is not None:
            self.tuning_iterations = get_iterations_from_tuning_config(context.model_tuning)
        super().__init__(name='tuning', child=child)

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        configs = []
        for idx, params in enumerate(self.tuning_iterations):
            self.idx[self.name] = idx
            config = self._init_config_from_context(context)
            config.pipeline_params = params
            if self.child is not None:
                child_configs, self.idx = self.child.generate(config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(config)
        return configs, self.idx


class IntraSubjectRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    def __init__(self, child: RunConfigGenerator | None = None):
        super().__init__(name='subject', child=child)

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        original_context = context.copy()
        original_subjects = original_context.subjects

        configs = []
        for idx, subject in enumerate(original_subjects):
            self.idx[self.name] = idx
            config = self._init_config_from_context(context)
            config.subjects = [subject]
            if self.child is not None:
                child_configs, self.idx = self.child.generate(config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(config)
        return configs, self.idx


class LOOCVSubjectRunConfigGenerator(RunConfigGenerator):
    train_name: str
    test_name: str
    name: str

    child: RunConfigGenerator | None = None

    def __init__(self, child: RunConfigGenerator | None = None,
                 name: str = "subject",
                 train_name: str = "train",
                 test_name: str = "test",
                 train_only: bool = False
                 ):
        super().__init__(name=name, child=child)
        self.train_name = train_name
        self.test_name = test_name
        self.train_only = train_only

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        original_context = context.copy()
        original_subjects = original_context.subjects

        configs = []
        for idx, subject in enumerate(original_subjects):
            self.idx[self.name] = idx
            iteration_subjects = original_subjects.copy()
            iteration_subjects.remove(subject)
            train_config = self._init_config_from_context(context, name_suffix=self.train_name)
            train_config.subset = self.train_name
            train_config.subjects = iteration_subjects if len(iteration_subjects) > 0 else original_context
            if self.child is not None:
                child_configs, self.idx = self.child.generate(train_config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(train_config)
            test_config = self._init_config_from_context(context, name_suffix=self.test_name)
            test_config.subset = self.test_name
            test_config.subjects = [subject]
            configs.append(test_config)
        return configs, self.idx


class CVRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    train_name: str = "train"
    test_name: str = "test"

    cv: StratifiedKFold

    def __init__(self,
                 child: RunConfigGenerator | None = None,
                 train_name: str = "train",
                 test_name: str = "test",
                 random_state: int = 1,
                 n_splits: int = 5
                 ):
        super().__init__(name='epoch', child=child)
        self.train_name = train_name
        self.test_name = test_name
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        configs = []

        dataset.load_by(subjects=context.subjects, sessions=context.sessions)

        if context.subset_indexes is None:
            context.subset_indexes = np.array(range(len(dataset.labels)))

        X: ndarray = dataset.data[context.subset_indexes]
        y: ndarray = dataset.labels[context.subset_indexes]

        for idx, (train_index, test_index) in enumerate(self.cv.split(X, y)):
            self.idx[self.name] = idx
            train_config = self._init_config_from_context(context, name_suffix=self.train_name)
            train_config.subset = self.train_name
            train_config.subset_indexes = context.subset_indexes[train_index]
            if self.child is not None:
                child_configs, self.idx = self.child.generate(train_config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(train_config)
            test_config = self._init_config_from_context(context, name_suffix=self.test_name)
            test_config.subset = self.test_name
            test_config.subset_indexes = context.subset_indexes[test_index]
            configs.append(test_config)

        return configs, self.idx


class FractionalRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    train_name: str = "train"
    test_name: str = "test"
    train_fraction: float

    def __init__(self,
                 child: RunConfigGenerator | None = None,
                 train_name: str = "train",
                 test_name: str = "test",
                 train_fraction: float = 0.8,
                 train_only: bool = False
                 ):
        super().__init__(name='epoch', child=child)
        self.train_name = train_name
        self.test_name = test_name
        self.train_fraction = train_fraction
        self.train_only = train_only

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        configs = []
        self.idx[self.name] = 0
        train_config = self._init_config_from_context(context, name_suffix=self.train_name)
        train_config.subset = self.train_name
        train_config.train_fraction = self.train_fraction

        if self.train_only:
            train_config = self._init_test_config_from_train(train_config, name_suffix=self.test_name)

        if self.child is not None:
            child_configs, self.idx = self.child.generate(train_config, dataset, self.idx)
            configs.extend(child_configs)
        else:
            configs.append(train_config)

        if not self.train_only:
            test_config = self._init_config_from_context(context, name_suffix=self.test_name)
            test_config.subset = self.test_name
            test_config.train_fraction = self.train_fraction
            configs.append(test_config)

        return configs, self.idx


class BlockRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    train_name: str = "train"
    test_name: str = "test"

    def __init__(self,
                 child: RunConfigGenerator | None = None,
                 train_name: str = "train",
                 test_name: str = "test",
                 train_only: bool = False
                 ):
        super().__init__(name='epoch', child=child)
        self.train_name = train_name
        self.test_name = test_name
        self.train_only = train_only

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None,
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)

        configs = []

        available_blocks = dataset.available_blocks()
        if len(available_blocks) > 1:
            num_train_blocks = math.ceil(len(available_blocks) * 0.5)
            train_blocks = available_blocks[:num_train_blocks]
            test_blocks = available_blocks[num_train_blocks:]
        else:
            train_blocks = None
            test_blocks = None

        self.idx[self.name] = 0
        train_config = self._init_config_from_context(context, name_suffix=self.train_name)
        train_config.subset = self.train_name
        train_config.blocks = train_blocks

        if self.train_only:
            train_config = self._init_test_config_from_train(train_config, name_suffix=self.test_name)
            train_config.test_blocks = test_blocks

        if self.child is not None:
            child_configs, self.idx = self.child.generate(train_config, dataset, self.idx)
            configs.extend(child_configs)
        else:
            configs.append(train_config)

        if not self.train_only:
            test_config = self._init_config_from_context(context, name_suffix=self.test_name)
            test_config.subset = self.test_name
            test_config.blocks = test_blocks
            configs.append(test_config)

        return configs, self.idx


class SingleRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    def __init__(self,
                 child: RunConfigGenerator | None = None,
                 ):
        super().__init__(name='single', child=child)

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):
        super().generate(context, dataset, parent_idx)
        configs = []
        self.idx[self.name] = 0
        if self.child is not None:
            child_configs, self.idx = self.child.generate(context, dataset, self.idx)
            configs.extend(child_configs)
        else:
            configs.append(self._init_config_from_context(context))
        return configs, self.idx


class EnsembleRunConfigGenerator(RunConfigGenerator):
    child: RunConfigGenerator | None = None

    def __init__(self, child: RunConfigGenerator | None = None, ensemble_size: int = 3):
        super().__init__(name='ensemble', child=child)
        self.ensemble_size = ensemble_size

    def generate(self,
                 context: RunConfig,
                 dataset: BaseDataset,
                 parent_idx: Dict[str, int] | None = None
                 ) -> (List[RunConfig], Dict[str, int]):

        super().generate(context, dataset, parent_idx)
        seeds = np.random.randint(0, 1000, self.ensemble_size)
        configs = []
        for ensemble_idx in range(self.ensemble_size):
            self.idx[self.name] = ensemble_idx
            config = self._init_config_from_context(context)
            config.seed = int(seeds[ensemble_idx])
            if self.child is not None:
                child_configs, self.idx = self.child.generate(config, dataset, self.idx)
                configs.extend(child_configs)
            else:
                configs.append(config)
        return configs, self.idx


def idx_to_string(idx: Dict[str, int]) -> str:
    name = ""
    for key in idx:
        name += f"{key[0]}_{idx[key]}_"
    return name[:-1]


def get_iterations_from_tuning_augmentation(tuning_augmentation: AugmentationInfo) -> List[AugmentationInfo | None]:
    if tuning_augmentation is None:
        return [None]
    if tuning_augmentation.search is not None:
        params = resolve_search_list(tuning_augmentation.search)
        # params.insert(0, 0)
        configs = []
        for param in params:
            kwargs = tuning_augmentation.kwargs.copy()
            kwargs.update({tuning_augmentation.optimization_parameter: param})
            augmentation = tuning_augmentation.copy()
            augmentation.kwargs = kwargs
            configs.append(augmentation)
        return configs
    else:
        return [tuning_augmentation]


def get_iterations_from_tuning_config(config: ModelTuningConfig) -> List[Dict | None]:
    if config is not None and config.scope == 'generator':
        return get_iterations_from_config(config)
    else:
        return [None]

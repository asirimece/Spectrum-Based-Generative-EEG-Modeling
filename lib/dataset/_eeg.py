import os
from pathlib import Path
from mne.io import BaseRaw
from mne import Epochs
import matplotlib.pyplot as plt
from lib.exception import NotLoadedError, DatasetLockedError
from .airplane import get_events_mapping
import mne
from typing import List, Set, Dict
from enum import Enum
from numpy import ndarray
from typing import TypedDict
from torch.utils.data import Dataset
import torch
from typing import Callable
from torch import Tensor, as_tensor
from lib.dataset.torch.transform import TorchTransform
from mne import EpochsArray
from lib.logging import get_logger
import numpy as np
from lib.preprocess import RawPreprocessor
from sklearn.utils import shuffle as shuffle_fn
from lib.config import DictInit
from lib.preprocess import get_raw_preprocessors
from lib.transform import MneEpochsTransform, get_epochs_transforms, TransformStep
from lib.experiment import ExperimentSubMode
from scipy.ndimage import uniform_filter1d
from lib.dataset.torch.transform import get_transforms
import torchvision
from lib.logging import get_logger

logger = get_logger()


class DatasetType(Enum):
    BASE: str = "base"
    TORCH: str = "torch"
    TORCH_STACKED: str = "torch_stacked"
    TORCH_AVERAGED: str = "torch_averaged"

    @staticmethod
    def from_string(string: str) -> 'DatasetType':
        return DatasetType[string.upper()]


class Label(TypedDict):
    name: str
    id: int


class Meta:
    subjects: Set[int]
    sessions: Set[int]
    blocks: Set[int]

    def __init__(self,
                 subjects: List[int] | Set[int] | None,
                 sessions: List[int] | Set[int] | None,
                 blocks: List[int] | Set[int] | None):
        self.subjects = subjects if subjects is not None else []
        self.sessions = sessions if sessions is not None else []
        self.blocks = blocks if blocks is not None else []

    def __eq__(self, other):
        return self.subjects == other.subjects and self.sessions == other.sessions and self.blocks == other.blocks


def build_epochs(raw: BaseRaw, t_min: float = -0.2, t_max: float = 1.0, event_id: List[int] | None = None) -> Epochs:
    if event_id is None:
        event_id = [0, 1]
    event_mapping = get_events_mapping()
    events, _ = mne.events_from_annotations(raw, event_id=event_mapping)
    epochs = mne.Epochs(raw, events, event_id, tmin=t_min, tmax=t_max, baseline=None, preload=True)
    return epochs


def balance_epochs(epochs: Epochs, classes: List[Label], n_epochs: int | None = None, shuffle: bool = True) -> Epochs:
    final_epochs = []
    separated_epochs = [epochs[str(label['id'])] for label in classes]
    epoch_lengths = [len(sep_epochs) for sep_epochs in separated_epochs]
    min_length = min(epoch_lengths)
    for sep_epochs in separated_epochs:
        indices = np.random.choice(np.arange(len(sep_epochs)), min_length, replace=False)
        final_epochs.append(sep_epochs[indices])
    final_epochs = mne.concatenate_epochs(final_epochs)
    if shuffle:
        final_epochs = final_epochs[shuffle_fn(range(len(final_epochs)))]
    return final_epochs[:n_epochs] if n_epochs is not None and len(final_epochs) > n_epochs else final_epochs


class BaseDataset:
    source_path: Path
    name: str
    reference: str | None = None

    meta: Meta

    __raw: BaseRaw | None = None
    raw_preprocessed: bool = False
    raw_preprocessors: List[RawPreprocessor] = []

    __epochs: Epochs | None = None
    epochs_transformed: bool = False
    epoch_preprocessors: List[MneEpochsTransform] = []

    __classes: List[Label] | None = None

    _lock: bool = False

    logger = get_logger()

    t_min: float = -0.2
    t_max: float = 1.0

    def __init__(self, name: str, source_path: str | Path | None = None, lock: bool = False):
        self.name = name
        self.source_path = Path(source_path) if type(source_path) is str else source_path
        self._lock = lock
        self._logger = get_logger()
        self.meta = Meta(None, None, None)

    @property
    def raw(self) -> BaseRaw:
        if self.__raw is None:
            raise NotLoadedError()
        return self.__raw

    def raw_exists(self) -> bool:
        return self.__raw is not None

    @raw.setter
    def raw(self, raw: BaseRaw):
        self.__verify_epochs_lock()
        self.__raw = raw
        self.raw_preprocessed = False

    @property
    def raw_data(self) -> ndarray:
        return self.raw.get_data()

    @property
    def epochs(self) -> Epochs:
        if self.epochs_transformed:
            return self.__epochs
        else:
            self.__epochs = self.build_epochs()
            self.epochs_transformed = True
            return self.__epochs

    @epochs.setter
    def epochs(self, epochs: Epochs):
        self.epochs_transformed = True
        self.__epochs = epochs

    @property
    def data(self) -> ndarray:
        return self.epochs.get_data(copy=True)
    
    @property
    def labels(self) -> ndarray:
        return self.epochs.events[:, 2]
    
    @property
    def dataset_classes(self) -> List[Label]:
        return self.__classes

    @property
    def class_names(self):
        if self.__classes is None:
            return None
        return [label['name'] for label in self.__classes]

    def available_subjects(self) -> List[int]:
        return []

    def available_sessions(self) -> List[int]:
        return []

    def available_blocks(self) -> List[int]:
        return []

    def set_classes(self, classes: List[Label]) -> 'BaseDataset':
        self.__classes = classes
        return self

    def set_epoch_limits(self, t_min: float, t_max: float) -> 'BaseDataset':
        self.t_min = t_min
        self.t_max = t_max
        return self

    def add_raw_preprocessors(self,
                              preprocessors: List[RawPreprocessor] | List[DictInit],
                              replace: bool = False) -> 'BaseDataset':
        if len(preprocessors) > 0 and not isinstance(preprocessors[0], RawPreprocessor):
            preprocessors = get_raw_preprocessors(preprocessors)
        if replace:
            self.raw_preprocessors = []
        self.raw_preprocessors.extend(preprocessors)
        return self

    def add_epochs_transforms(self,
                              transforms: List[MneEpochsTransform] | List[DictInit],
                              replace: bool = False, update_existing: bool = False) -> 'BaseDataset':
        if len(transforms) > 0:
            transforms = get_epochs_transforms(transforms)
        if replace:
            self.epoch_preprocessors = []
        if update_existing:
            for transform in transforms:
                existing_idx = next((i for i, p in enumerate(self.epoch_preprocessors)
                                     if p.name == transform.name), None)
                if existing_idx is not None:
                    self.epoch_preprocessors[existing_idx] = transform
                else:
                    self.epoch_preprocessors.append(transform)
        else:
            self.epoch_preprocessors.extend(transforms)
        return self

    def remove_epochs_transforms(self, transform_names: List[str]):
        self.epoch_preprocessors = [p for p in self.epoch_preprocessors if p.name not in transform_names]
        return self

    def get_events(self) -> List:
        event_mapping = get_events_mapping()
        events, _ = mne.events_from_annotations(self.raw, event_id=event_mapping)
        return events

    def __verify_epochs_lock(self):
        if self.epochs_transformed and self._lock:
            raise DatasetLockedError()
        elif self.epochs_transformed:
            self._logger.info("Epochs already built or dataset has already been transformed. "
                              "As lock mode is off, the dataset will automatically be reset.")
            self.reset_epochs()

    def build_epochs(self, event_id: List[int] | None = None) -> Epochs:
        self.__verify_epochs_lock()
        return build_epochs(self.raw, self.t_min, self.t_max, event_id)

    def load(self) -> BaseRaw:
        pass


    def load_by(self,
                subjects: List[int] | None,
                sessions: List[int] | None,
                blocks: List[int] | None = None,
                force: bool = False
                ) -> BaseRaw:
        pass

    def load_balanced(self,
                      subjects: List[int] | None,
                      sessions: List[int] | None,
                      blocks: List[int] | None = None) -> Epochs:
        raise NotImplementedError(f"Balanced loading is not supported for {self.name} dataset.")

    def load_fractional(self,
                        subjects: List[int] | None,
                        sessions: List[int] | None,
                        blocks: List[int] | None = None,
                        train_fraction: float = 0.8,
                        subset: str = 'train') -> BaseRaw:
        raise NotImplementedError(f"Fractional loading is not supported for {self.name} dataset.")

    def load_by_config(self, config):
        self.logger.info(f"Loading {self.name} data for subjects {config.subjects} "
                         f"and sessions {config.sessions} and blocks {config.blocks}")
        if config.load_balanced:
            self.load_balanced(subjects=config.subjects, sessions=config.sessions)
        else:
            if config.sub_mode == ExperimentSubMode.FRACTIONAL:
                self.load_fractional(subjects=config.subjects, sessions=config.sessions, blocks=config.blocks,
                                     train_fraction=config.train_fraction, subset=config.subset)
            else:
                self.load_by(subjects=config.subjects, sessions=config.sessions, blocks=config.blocks)
        return self

    def set_reference(self, reference: str = 'average', projection: bool = True):
        self.reference = reference
        self.raw.set_eeg_reference(reference, projection=projection)

    def plot_montage(self):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        self.raw.plot_sensors(show_names=True, axes=ax)
        plt.show()

    def plot_reference(self):
        if self.reference is not None:
            for proj in (False, True):
                with mne.viz.use_browser_backend("matplotlib"):
                    fig = self.raw.plot(proj=proj, scalings=dict(eeg=50e-6), show_scrollbars=False, start=5, duration=5,
                                        show=False)
                fig.subplots_adjust(top=0.9) 
                ref = "Average" if proj else "No"
                fig.suptitle(f"{ref} reference", weight="bold")
                fig.set_size_inches(10, 5)
                plt.show()
        else:
            print("No reference data available")

    def reset_epochs(self):
        try:
            del self.__epochs
        except AttributeError:
            pass
        self.__epochs = None
        self.epochs_transformed = False

    def reset_raw(self):
        try:
            del self.__raw
        except AttributeError:
            pass
        self.__raw = None
        self.raw_preprocessed = False

    def reset(self, unlock: bool = False):
        self.meta = Meta(None, None, None)
        if unlock:
            self._lock = False
        self.reset_epochs()
        self.reset_raw()
        if unlock:
            self._lock = False

    def _create_meta(self,
                     subjects: List | None = None,
                     sessions: List | None = None,
                     blocks: List | None = None) -> Meta:
        subjects = [int(subject) for subject in subjects] if subjects is not None else None
        sessions = [int(session) for session in sessions] if sessions is not None else None
        blocks = [int(block) for block in blocks] if blocks is not None else None
        return Meta(subjects, sessions, blocks)

    def raw_preprocess(self):
        if self.raw_exists() and not self.raw_preprocessed:
            for preprocessor in self.raw_preprocessors:
                self.raw = preprocessor.transform(self.raw)
            self.raw_preprocessed = True

    def epoch_process(self, transform_step: TransformStep | None = None):
        preprocessors = self.epoch_preprocessors.copy()
        if transform_step is not None:
            preprocessors = [p for p in preprocessors if p.transform_step == transform_step]
        if len(preprocessors) > 0:
            self.logger.info(f"Preprocessing epochs for step {transform_step} with {len(preprocessors)} transforms")
            for transform in preprocessors:
                self.epochs = transform.transform_epochs(self.epochs.copy())
            self.logger.info(f"Epochs shape: {self.epochs.get_data(copy=True).shape}")


class DatasetBuilder:
    local_path: Path

    def __init__(self, local_path: str | Path | None = None):
        if local_path is None:
            local_path = Path(os.getcwd()).joinpath("data")
        self.local_path: Path = Path(local_path) if type(local_path) is str else local_path

    def exists(self) -> bool:
        return self.local_path.exists()

    def download(self):
        pass

    def build(self, force_download: bool = False, verify_hash: bool = True) -> BaseDataset:
        pass


class TorchBaseDataset(BaseDataset, Dataset):
    transforms: TorchTransform | Callable | None = None
    post_transforms: TorchTransform | Callable | None = None

    transforms_config: List[Dict] | None = None
    post_transforms_config: List[Dict] | None = None

    _data_tensor: Tensor | None = None
    _labels_tensor: Tensor | None = None

    def __init__(self,
                 name: str,
                 source_path: str | Path | None = None,
                 transforms: Callable[[Tensor], Tensor] | None = None):
        super().__init__(name, source_path)
        self.transforms = transforms

    def __len__(self):
        return len(self._data_tensor) if self._data_tensor is not None else len(self.epochs)

    def __getitem__(self, idx):
        if self._data_tensor is None:
            self.lock_and_retrieve()
        data = self._data_tensor[idx]
        label = self._labels_tensor[idx].squeeze()

        if self.transforms is not None:
            data = self.transforms(data)

        return data, label

    def set_transforms_by_config(self, transforms_config: List[Dict]):
        self.transforms_config = transforms_config
        self.transforms = torchvision.transforms.Compose(get_transforms(self.transforms_config, self))
        return self

    def set_post_transforms_by_config(self, post_transforms_config: List[Dict]):
        self.post_transforms_config = post_transforms_config
        self.post_transforms = torchvision.transforms.Compose(get_transforms(self.post_transforms_config, self))
        return self

    def reinitialise_transforms(self):
        if self.transforms_config is not None:
            self.transforms = torchvision.transforms.Compose(get_transforms(self.transforms_config, self))
        if self.post_transforms_config is not None:
            self.post_transforms = torchvision.transforms.Compose(get_transforms(self.post_transforms_config, self))

    def get_transformed_epochs(self, apply_post_transforms: bool = True) -> EpochsArray:
        epochs = self.epochs.copy()
        data = epochs.get_data(copy=True)
        if self.transforms is not None:
            data = self.transforms(as_tensor(data)).numpy()
        if apply_post_transforms and self.post_transforms is not None:
            data = self.post_transforms(as_tensor(data)).numpy()
        return EpochsArray(data, info=epochs.info, tmin=epochs.tmin, events=epochs.events)

    def lock_and_retrieve(self) -> BaseDataset:
        self._lock = True
        self._data_tensor = torch.FloatTensor(self.epochs.get_data(copy=False)).unsqueeze(1)
        self._labels_tensor = torch.LongTensor(self.epochs.events[:, 2])
        return self

    def reset(self, unlock: bool = False):
        self.reset_epochs()
        self.reset_raw()
        self._lock = not unlock


class TorchStackedDataset(TorchBaseDataset):
    n_stacks: int
    averaged: bool
    average_window_size: int
    average_classes: List[int]

    def __init__(self,
                 name: str,
                 source_path: str | Path | None = None,
                 transforms: Callable[[Tensor], Tensor] | None = None,
                 n_stacks: int = 8,
                 averaged: bool = True,
                 average_window_size: int = 5,
                 average_classes: List[int] = None):
        super().__init__(
            name=name,
            source_path=source_path,
            transforms=transforms
        )
        self._n_stacks = n_stacks
        self.averaged = averaged
        self.average_window_size = average_window_size
        self.average_classes = average_classes if average_classes is not None else [1]

    @property
    def n_stacks(self) -> int:
        return self._n_stacks

    @n_stacks.setter
    def n_stacks(self, n_stacks: int):
        self._n_stacks = n_stacks

    def lock_and_retrieve(self) -> BaseDataset:
        self._lock = True
        class_names = np.unique(self.epochs.events[:, 2])
        epochs = []
        labels: List[int] = []
        for class_name in class_names:
            class_epochs = self.epochs[str(class_name)]
            class_epochs_data = class_epochs.get_data(copy=False)
            if self.averaged and class_name in self.average_classes:
                avg_class_epochs_data = uniform_filter1d(class_epochs_data, size=self.average_window_size, axis=0,
                                                         mode='nearest')
                class_epochs_data = np.concatenate([class_epochs_data, avg_class_epochs_data], axis=0)
            remar = len(class_epochs_data) % self.n_stacks
            if remainder > 0:
                class_epochs_data = class_epochs_data[:-remainder]
            batch_size = int(len(class_epochs_data) / self.n_stacks)
            n_channels = class_epochs_data.shape[1] * self.n_stacks
            class_epochs_data = class_epochs_data.reshape(batch_size, n_channels, -1)
            epochs.append(class_epochs_data)
            labels.extend([int(class_name)] * batch_size)
        epochs = np.concatenate(epochs, axis=0)
        epochs, labels = shuffle_fn(epochs, labels)
        self._data_tensor = torch.FloatTensor(epochs).unsqueeze(1)
        self._labels_tensor = torch.LongTensor(labels)
        return self


class TorchAveragedDataset(TorchBaseDataset):
    _window_size: int
    classes = List[int]
    additive: bool

    def __init__(self,
                 name: str,
                 source_path: str | Path | None = None,
                 transforms: Callable[[Tensor], Tensor] | None = None,
                 window_size: int = 5,
                 classes: List[int] = None,
                 additive: bool = True):
        super().__init__(
            name=name,
            source_path=source_path,
            transforms=transforms
        )
        self._window_size = window_size
        self.classes = classes if classes is not None else [1]
        self.additive = additive

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, window_size: int):
        self._window_size = window_size

    def lock_and_retrieve(self) -> BaseDataset:
        self._lock = True
        self._data_tensor = torch.FloatTensor(self.epochs.get_data(copy=False)).unsqueeze(1)
        self._labels_tensor = torch.LongTensor(self.epochs.events[:, 2])
        class_names = np.unique(self.epochs.events[:, 2])
        averaged_epochs = []
        averaged_labels: List[int] = []
        for class_name in class_names:
            if class_name in self.classes:
                class_epochs = self.epochs[str(class_name)]
                class_epochs_data = class_epochs.get_data(copy=False)
                class_epochs_data = uniform_filter1d(class_epochs_data, size=self.window_size, axis=0, mode='nearest')
                averaged_epochs.append(class_epochs_data)
                averaged_labels.extend([int(class_name)] * len(class_epochs_data))

        averaged_epochs = np.concatenate(averaged_epochs, axis=0)

        if self.additive:
            epochs = self.epochs.get_data(copy=False)
            labels = self.epochs.events[:, 2]

            epochs = np.concatenate([epochs, averaged_epochs], axis=0)
            labels = np.concatenate([labels, averaged_labels], axis=0)
        else:
            epochs = averaged_epochs
            labels = averaged_labels

        epochs, labels = shuffle_fn(epochs, labels, random_state=1)

        self._data_tensor = torch.FloatTensor(epochs).unsqueeze(1)
        self._labels_tensor = torch.LongTensor(labels)
        return self

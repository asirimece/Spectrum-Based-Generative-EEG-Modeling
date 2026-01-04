from typing import Callable, Tuple, Union, List
import mne
import numpy as np
from torch import Tensor
import torch
from torch import nn
from torch.cuda import Device
from lib.config import DictInit
from numpy import ndarray
from sklearn.utils import check_random_state, shuffle as shuffle_fn
from enum import Enum
from mne import Epochs, EpochsArray
from lib.logging import get_logger
from pathlib import Path
from collections import Counter
from lib.preprocess import RawPreprocessor, get_raw_preprocessors

Operation = Callable[[Tensor, Tensor, ...], Tuple[Tensor, Tensor]]

TransformResult = Union[Tensor | ndarray, Tuple[Tensor | ndarray, Tensor | ndarray]]

TorchTransformResult = Union[
    torch.Tensor,
    Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]
]

logger = get_logger()


class TransformType(Enum):
    IN_PLACE = "in_place"
    ADDITIVE = "additive"

    @staticmethod
    def from_string(string: str) -> 'TransformType':
        return TransformType[string.upper()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class TransformStep(Enum):
    PREPROCESS = "preprocess"
    AUGMENT = "augment"
    POST_AUGMENT = "post_augment"
    POSTPROCESS = "postprocess"

    @staticmethod
    def from_string(string: str) -> 'TransformStep':
        return TransformStep[string.upper()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Transform:
    _name: str
    _transform_type: TransformType | str
    _transform_step: TransformStep | str
    _random_state: int
    rng: any
    _classes: List[int] | None

    def __init__(self,
                 name: str,
                 transform_type: TransformType | str = TransformType.IN_PLACE,
                 transform_step: TransformStep | str = TransformStep.PREPROCESS,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 ):
        self._name = name
        self._transform_type = check_transform_type(transform_type)
        self._transform_step = check_transform_step(transform_step)
        self._random_state = random_state
        self.rng = check_random_state(random_state)
        self._classes = classes

    @property
    def name(self):
        return self._name

    @property
    def transform_type(self):
        return self._transform_type

    @property
    def transform_step(self):
        return self._transform_step

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def check_type(X: Tensor | ndarray, y: Tensor | ndarray = None, input_type=None) -> (
            (Tensor | ndarray, Tensor | ndarray, type)
    ):
        if input_type is None:
            if isinstance(X, ndarray):
                input_type = ndarray
                X = torch.from_numpy(X)
            if isinstance(y, ndarray):
                y = torch.from_numpy(y)
            return X, y, input_type
        else:
            if input_type == ndarray:
                if isinstance(X, Tensor):
                    X = X.numpy()
                if isinstance(y, Tensor):
                    y = y.numpy()
            return X, y, input_type

    def transform(self, X: Tensor | ndarray, y: Tensor | ndarray | None = None) -> TransformResult:
        return X, y


class AugmentationTransform(Transform):
    _operation: Operation
    _probability: float
    shuffle: bool

    def __init__(self,
                 name: str,
                 operation: Callable,
                 probability: float = 0.5,
                 transform_type: TransformType | str = TransformType.IN_PLACE,
                 transform_step: TransformStep | str = TransformStep.PREPROCESS,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 shuffle: bool = False):
        super().__init__(name, transform_type, transform_step, random_state, classes)
        assert 0 <= probability <= 1, "Probability must be between 0 and 1."
        self._operation = operation
        self._probability = probability
        self.shuffle = shuffle

    def _get_mask(self, batch_size, device, y: Tensor | None, probability: float | None = None) -> torch.Tensor:
        """Samples whether to apply operation or not over the whole batch
        """
        probability = probability if probability is not None else self._probability
        mask = torch.as_tensor(
            probability > self.rng.uniform(size=batch_size)
        ).to(device)
        if self._classes is not None and y is not None:
            class_mask = torch.isin(y, torch.tensor(self._classes))
            mask = mask & class_mask
        return mask

    def get_augmentation_params(self, *batch):
        return dict()

    def transform(self, X: Tensor | ndarray, y: Tensor | ndarray | None = None) -> TransformResult:
        logger.debug(f"Applying transform {self._name} to input of shape {X.shape} and type {type(X)}")
        assert isinstance(X, Tensor) or isinstance(X, ndarray), "X must be a Tensor or ndarray."
        assert y is None or isinstance(X, Tensor) or isinstance(X, ndarray), "y must be a Tensor or ndarray."
        assert y is None or len(y.shape) <= len(X.shape) - 1, "y must have a smaller dimension than X."

        X, y, input_type = Transform.check_type(X, y)

        X_out = X.clone()
        X_out_orig = None
        y_out_orig = None

        if len(X_out.shape) < 3:
            X_out = X_out[None, ...]

        if y is not None:
            y = torch.as_tensor(y).to(X_out.device)
            y_out = y.clone()
            if len(y_out.shape) == 0:
                y_out = y_out.reshape(1)
        else:
            y_out = torch.zeros(X_out.shape[0], device=X_out.device)

        if self._transform_type == TransformType.ADDITIVE:
            X_out_orig = X_out.clone()
            y_out_orig = y_out.clone()

        mask = self._get_mask(X_out.shape[0], X_out.device, y=y_out if y is not None else None)
        num_valid = mask.sum().long()

        if num_valid > 0:
            X_out[mask, ...], tr_y = self._operation(
                X_out[mask, ...], y_out[mask],
                **self.get_augmentation_params(X_out[mask, ...], y_out[mask])
            )
            if isinstance(tr_y, tuple):
                y_out = tuple(tmp_y[mask] for tmp_y in tr_y)
            else:
                y_out[mask] = tr_y
            if self._transform_type == TransformType.ADDITIVE:
                X_out = torch.cat([X_out_orig, X_out[mask, ...]], dim=0)
                y_out = torch.cat([y_out_orig, y_out[mask, ...]], dim=0)

        if self._transform_type != TransformType.ADDITIVE:
            X_out = X_out.reshape_as(X)
        X_out, y_out, _ = Transform.check_type(X_out, y_out, input_type)

        if self.shuffle:
            X_out, y_out = shuffle_fn(X_out, y_out, random_state=self._random_state)

        if y is not None:
            y_out = y_out.reshape_as(y)
            return X_out, y_out
        else:
            return X_out


class TorchTransform(Transform, torch.nn.Module):

    def __init__(self,
                 name: str,
                 operation: Callable,
                 probability: float = 0.5,
                 random_state: int | None = None,
                 classes: List[int] | None = None
                 ):
        super().__init__(
            name=name,
            operation=operation,
            probability=probability,
            transform_type=TransformType.IN_PLACE,
            random_state=random_state,
            classes=classes
        )
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor, y: Tensor | None = None) -> TorchTransformResult:
        return super().transform(X, y)


class MneEpochsTransform(Transform):
    epochs_transform_type: TransformType

    def __init__(self,
                 name: str,
                 transform_type: TransformType | str = TransformType.IN_PLACE,
                 transform_step: TransformStep | str = TransformStep.PREPROCESS,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 ):
        super().__init__(
            name=name,
            transform_type=TransformType.IN_PLACE,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
        )
        self.epochs_transform_type = check_transform_type(transform_type)

    def transform_epochs(self, epochs: Epochs | EpochsArray) -> Epochs | EpochsArray:
        return epochs


class MneAugmentationEpochsTransform(AugmentationTransform, MneEpochsTransform):
    epochs_transform_type: TransformType
    _epochs_shuffle: bool
    _epochs_probability: float

    def __init__(self,
                 name: str,
                 operation: Callable,
                 probability: float = 0.5,
                 transform_type: TransformType | str = TransformType.IN_PLACE,
                 transform_step: TransformStep | str = TransformStep.PREPROCESS,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 shuffle: bool = False,
                 remainder_operation: Union[Callable[[ndarray], ndarray], None] = None
                 ):
        super().__init__(
            name=name,
            operation=operation,
            probability=1.0,
            transform_type=TransformType.IN_PLACE,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
            shuffle=False,
        )
        self.epochs_transform_type = check_transform_type(transform_type)
        self._epochs_probability = probability
        self._epochs_shuffle = shuffle
        self.remainder_operation = remainder_operation

    def transform_epochs(self, epochs: Epochs | EpochsArray) -> Epochs | EpochsArray:
        labels = epochs.events[:, 2]
        assert len(epochs) == len(labels), "Epochs and labels must have the same length."

        mask = self._get_mask(len(epochs), 'cpu', y=torch.as_tensor(labels),
                              probability=self._epochs_probability).numpy()

        data = epochs.get_data(copy=True)
        transform_epochs = epochs[mask].get_data(copy=True)
        augmented_data = self.transform(X=transform_epochs)

        if self.remainder_operation is not None:
            data = self.remainder_operation(data)
            epochs = EpochsArray(data, info=epochs.info, tmin=epochs.tmin, events=epochs.events)

        if self.epochs_transform_type == TransformType.ADDITIVE:
            augmented_epochs = EpochsArray(augmented_data.copy(), info=epochs.info,
                                           tmin=epochs.tmin, events=epochs.events[mask])
            final_epochs = mne.concatenate_epochs([epochs, augmented_epochs])
        else:
            data[mask] = augmented_data
            final_epochs = EpochsArray(data.copy(), info=epochs.info, tmin=epochs.tmin, events=epochs.events)

        if self._epochs_shuffle:
            final_epochs = final_epochs[shuffle_fn(range(len(final_epochs)))]

        return final_epochs


def gen_unimplemented_operation():
    raise NotImplementedError("Operation is not supported yet forMneEpochsTransform ")


class MneGenEpochsTransform(MneAugmentationEpochsTransform):
    fraction: float | str
    model_path: str | Path | None
    model: nn.Module
    preprocessors: List[RawPreprocessor]
    device: Device

    def __init__(self,
                 name: str,
                 fraction: float | str = 1.5,
                 probability: float = 1.0,
                 model_path: str | Path | None = None,
                 preprocessors: List[RawPreprocessor] | List[DictInit] | None = None,
                 transform_step: TransformStep | str = TransformStep.PREPROCESS,
                 random_state: int | None = None,
                 classes: List[int] | None = None,
                 shuffle: bool = False
                 ):
        super().__init__(
            name=name,
            operation=gen_unimplemented_operation,
            probability=probability,
            transform_step=transform_step,
            random_state=random_state,
            classes=classes,
            shuffle=shuffle
        )

        self.fraction = fraction
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize()

        if preprocessors is not None:
            self.preprocessors = get_raw_preprocessors(preprocessors) if isinstance(preprocessors[0], dict) \
                else preprocessors

    def _initialize(self):
        if self.model_path is None:
            raise ValueError("Model path is not set.")
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        if hasattr(self.model, 'device'):
            self.model.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Loaded model from {model_path}")

    @staticmethod
    def _get_balanced_labels(labels: ndarray):
        gen_labels = []
        counter = Counter(labels)
        most_common = counter.most_common(1)[0]
        for value, count in counter.items():
            if value != most_common[0]:
                gen_labels.extend([value] * (most_common[1] - count))
        return gen_labels

    def _get_fractional_increased_labels(self, labels: ndarray):
        assert isinstance(self.fraction, (int, float)), "Only int or float fraction is currently supported."
        gen_labels = []
        extending_labels = set(labels) if self.classes is None else set(labels).intersection(set(self.classes))
        for label in extending_labels:
            count = np.count_nonzero(labels == label)
            multiply: int = max(int(self.fraction * count) - count, 0)
            gen_labels.extend([label] * multiply)
        return gen_labels

    def gen_epochs(self, epochs: Epochs | EpochsArray, gen_labels: ndarray) -> Epochs | EpochsArray:
        return epochs

    def transform_epochs(self, epochs: Epochs | EpochsArray) -> Epochs | EpochsArray:
        assert isinstance(self.fraction, (float, int)) or self.fraction == 'balance', \
            "Only balance or float fractions are currently supported."
        labels = epochs.events[:, 2]
        if self.fraction == "balance":
            gen_labels = MneGenEpochsTransform._get_balanced_labels(labels)
        else:
            gen_labels = self._get_fractional_increased_labels(labels)
        generated_epochs = self.gen_epochs(epochs, np.array(gen_labels))
        concat_epochs = mne.concatenate_epochs([epochs, generated_epochs])
        if self._epochs_shuffle:
            concat_epochs = concat_epochs[shuffle_fn(range(len(concat_epochs)))]

        return concat_epochs


def check_transform_type(transform_type: TransformType | str) -> TransformType:
    if isinstance(transform_type, str):
        return TransformType.from_string(transform_type)
    return transform_type


def check_transform_step(transform_step: TransformStep | str) -> TransformStep:
    if isinstance(transform_step, str):
        return TransformStep.from_string(transform_step)
    return transform_step

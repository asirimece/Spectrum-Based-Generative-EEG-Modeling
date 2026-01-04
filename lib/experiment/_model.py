from enum import Enum
from typing import List


class Subset(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @staticmethod
    def from_string(string: str):
        return Subset.__members__.get(string.lower(), None)

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ExperimentType(Enum):
    BASE_TRAIN_TEST = "base_train_test"
    AUG_AUGMENTATION_TUNING = "aug_augmentation_tuning"
    AUG_MODEL_TUNING = "aug_model_tuning"
    AUG_TRAIN_TEST = "aug_train_test"
    AUG_TRAIN_VALID_TEST = "aug_train_valid_test"
    GEN_MODEL_TUNING = "gen_model_tuning"
    GEN_TRAINING_EVALUATION = "gen_training_evaluation"

    @staticmethod
    def from_string(string: str) -> 'ExperimentType':
        return ExperimentType[string.upper()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ExperimentMode(Enum):
    INTRA_SUBJECT = "intra_subject"
    CROSS_SUBJECT = "cross_subject"

    @staticmethod
    def from_string(string: str) -> 'ExperimentMode':
        return ExperimentMode[string.upper()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ExperimentSubMode(Enum):
    DEFAULT = "default"
    BLOCK_BASED = "block_based"
    FRACTIONAL = "fractional"

    @staticmethod
    def from_string(string: str) -> 'ExperimentSubMode':
        return ExperimentSubMode[string.upper()]

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class SubjectSubsetSplit:
    train: List[int]
    val: List[int]
    test: List[int]

    def __init__(self, train: List[int], val: List[int], test: List[int]):
        self.train = train
        self.val = val
        self.test = test

    def to_dict(self):
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test
        }

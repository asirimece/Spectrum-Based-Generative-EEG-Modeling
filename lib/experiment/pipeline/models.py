
from enum import Enum


class PipelineMode(Enum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def from_string(string: str) -> 'PipelineMode':
        return PipelineMode[string.upper()]


class PipelineType(Enum):
    TRADITIONAL = "traditional"
    NEURAL = "neural"
    GEN = "gen"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def from_string(string: str) -> 'PipelineType':
        return PipelineType[string.upper()]


from enum import Enum
from typing import Union


class ModelVarType(Enum):
    LEARNED = 'LEARNED'
    FIXED = 'FIXED'

    def learned_sigma(self):
        return self in [ModelVarType.LEARNED]

    @staticmethod
    def from_string(string: str):
        return ModelVarType(string.upper())

    @staticmethod
    def check_instance(obj: Union[str, 'ModelVarType']) -> 'ModelVarType':
        if isinstance(obj, ModelVarType):
            return obj
        return ModelVarType.from_string(obj)


class ModelMeanType(Enum):
    X_PREV = 'X_PREV'
    X_START = 'X_START'
    NOISE = 'NOISE'
    PRED_V = 'PRED_V'  # https://arxiv.org/abs/2202.00512

    @staticmethod
    def from_string(string: str):
        return ModelMeanType(string.upper())

    @staticmethod
    def check_instance(obj: Union[str, 'ModelMeanType']) -> 'ModelMeanType':
        if isinstance(obj, ModelMeanType):
            return obj
        return ModelMeanType.from_string(obj)


class LossType(Enum):
    MSE = 'MSE'  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = 'RESCALED_MSE'  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = 'KL'  # use the variational lower-bound
    RESCALED_KL = 'RESCALED_KL'  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

    @staticmethod
    def from_string(string: str):
        return LossType(string.upper())

    @staticmethod
    def check_instance(obj: Union[str, 'LossType']) -> 'LossType':
        if isinstance(obj, LossType):
            return obj
        return LossType.from_string(obj)
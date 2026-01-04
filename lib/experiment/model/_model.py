from typing import Iterable, Union, List
from omegaconf import DictConfig
from lib.config import config_to_primitive


class TuningParameterConfig:
    name: str
    strategy: str
    children: List[str] | None
    start: Union[int, float] | None
    end: Union[int, float] | None
    step: Union[int, float] | None
    scale: float | None
    mean: float | None
    std: float | None
    values: List[Union[int, float]] | None
    size: int | None
    d_type: str

    def __init__(self,
                 name: str,
                 strategy: str,
                 d_type: str,
                 children: List[str] | None = None,
                 start: Union[int, float] | None = None,
                 end: Union[int, float] | None = None,
                 step: Union[int, float] | None = None,
                 scale: float | None = None,
                 mean: float | None = None,
                 std: float | None = None,
                 size: int | None = None,
                 values: List[Union[int, float]] | None = None):
        self.name = name
        self.strategy = strategy
        self.children = children
        self.start = start
        self.end = end
        self.step = step
        self.d_type = d_type
        self.scale = scale
        self.mean = mean
        self.std = std
        self.size = size
        self.values = values

    def to_dict(self):
        return {
            "name": self.name,
            "strategy": self.strategy,
            "children": self.children,
            "start": self.start,
            "end": self.end,
            "step": self.step,
            "d_type": self.d_type,
            "scale": self.scale,
            "mean": self.mean,
            "std": self.std,
            "size": self.size,
            "values": self.values
        }

    @staticmethod
    def from_config(config: DictConfig) -> 'TuningParameterConfig':
        assert 'name' in config, "TuningParameterConfig must have a name to define which parameter to tune"
        return TuningParameterConfig(
            name=config.name,
            strategy=config.strategy if 'strategy' in config else 'range',
            children=config_to_primitive(config.children) if 'children' in config else None,
            d_type=config.d_type if 'd_type' in config else 'float',
            start=config.start if 'start' in config else None,
            end=config.end if 'end' in config else None,
            step=config.step if 'step' in config else None,
            scale=config.scale if 'scale' in config else None,
            mean=config.mean if 'mean' in config else None,
            std=config.std if 'std' in config else None,
            size=config.size if 'size' in config else None,
            values=config_to_primitive(config.value_list) if 'value_list' in config else None
        )


class ModelTuningConfig:
    enabled: bool
    scope: str
    strategy: str
    n_iter: int | None
    cv: int
    parameters: List[TuningParameterConfig]

    def __init__(self,
                 strategy: str,
                 scope: str = 'generator',
                 cv: int = 3,
                 n_iter: int = None,
                 parameters: List[TuningParameterConfig] = None,
                 enabled: bool = False):
        self.enabled = enabled
        self.cv = cv
        self.strategy = strategy
        self.n_iter = n_iter
        self.scope = scope
        self.parameters = parameters if parameters is not None else []

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "scope": self.scope,
            "cv": self.cv,
            "n_iter": self.n_iter,
            "strategy": self.strategy,
            "parameters": [p.to_dict() for p in self.parameters]
        }

    @staticmethod
    def from_config(config: DictConfig) -> 'ModelTuningConfig':
        if 'parameters' in config and isinstance(config.parameters, Iterable):
            parameters = [TuningParameterConfig.from_config(p) for p in config.parameters]
        else:
            parameters = []
        return ModelTuningConfig(
            enabled=config.enabled if 'enabled' in config else False,
            scope=config.scope if 'scope' in config else 'generator',
            n_iter=config.n_iter if 'n_iter' in config else None,
            cv=config.cv if 'cv' in config else 3,
            strategy=config.strategy if 'strategy' in config else None,
            parameters=parameters
        )

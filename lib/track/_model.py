from enum import Enum
from typing import Dict, Any, List
from omegaconf import DictConfig
from typing import Callable


class TrackerTarget(Enum):
    ALL = "all"
    EXPERIMENT = "experiment"
    RUN = "run"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def from_string(string: str):
        return TrackerTarget[string.upper()]


class TrackerMode(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    @staticmethod
    def from_string(string: str):
        return TrackerMode[string.upper()]


class TrackerConfig:
    enabled: bool = True
    mode: TrackerMode = TrackerMode.ONLINE
    target: TrackerTarget = TrackerTarget.ALL

    def __init__(self,
                 enabled: bool = True,
                 mode: TrackerMode = TrackerMode.ONLINE,
                 target: TrackerTarget = TrackerTarget.ALL):
        self.enabled = enabled
        self.mode = mode
        self.target = target

    @staticmethod
    def from_config(config: DictConfig) -> 'TrackerConfig':
        return TrackerConfig(
            enabled=config.enabled,
            mode=TrackerMode.from_string(config.mode) if isinstance(config.mode, str) else config.mode,
            target=TrackerTarget.from_string(config.target) if isinstance(config.target, str) else config.target
        )


class Tracker:
    name: str
    config: TrackerConfig

    _initialized: bool = False

    auto_train_tracking: bool = False

    def __init__(self, name: str, config: TrackerConfig):
        self.name = name
        self.config = config

    @property
    def initialized(self):
        return self._initialized

    def init(self, *args, **kwargs):
        self._initialized = True
        pass

    def set_run_configuration(self, config: Dict):
        pass

    def log(self, data: Dict[str, Any], step: int | None = None):
        pass

    def log_image(self, data: Dict[str, Any], captions: str | List[str] = None, step: int | None = None):
        pass

    def finish(self):
        self._initialized = False
        pass

    @staticmethod
    def from_config(config: DictConfig) -> 'Tracker':
        tracker_config = TrackerConfig.from_config(config)
        return Tracker(name=config.name, config=tracker_config)
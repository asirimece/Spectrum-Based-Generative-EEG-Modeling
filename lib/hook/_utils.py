from .base import Hook
from ._hooks import Hooks, TrackerHook, ModelSaverHook
from omegaconf import DictConfig
from typing import List
from lib.logging import get_logger

logger = get_logger()


def get_hook_from_config(config: DictConfig) -> Hook:
    name = config.name
    match name:
        case Hooks.TRACKER.value:
            return TrackerHook.from_config(config)
        case Hooks.MODEL_SAVER.value:
            return ModelSaverHook.from_config(config)
        case _:
            raise ValueError(f"Unknown hook: {name}")
    logger.info(f"- Hooks: {config.pipeline.hooks}")


def get_hooks_from_configs(configs: List[DictConfig]) -> List[Hook]:
    return [get_hook_from_config(hook_config) for hook_config in configs]

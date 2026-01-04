from enum import Enum
from ._score import BalancedAccuracyScoring
from skorch.callbacks import Callback, WandbLogger, LRScheduler, EarlyStopping
from typing import Tuple, List, Dict
from wandb.wandb_run import Run
from wandb.sdk.wandb_settings import Settings
from lib.logging import get_logger

logger = get_logger()


class Callbacks(Enum):
    BALANCED_ACCURACY = "balanced_accuracy"
    LR_SCHEDULER = "lr_scheduler"
    WAND_LOGGER = "wandb_logger"
    EARLY_STOPPING = "early_stopping"


def get_callback_by_name(config: Dict) -> Callback:
    assert "name" in config, "Callback name is not specified"
    kwargs = config.get("kwargs", {})
    match config["name"]:
        case Callbacks.BALANCED_ACCURACY.value:
            return BalancedAccuracyScoring(**kwargs)
        case Callbacks.WAND_LOGGER.value:
            logger.debug("If you are using wandb_logger, it's recommended to use WandB Tracker with TrackerHook.")
            return WandbLogger(wandb_run=Run(settings=Settings()), **kwargs)
        case Callbacks.LR_SCHEDULER.value:
            return LRScheduler(**kwargs)
        case Callbacks.EARLY_STOPPING.value:
            return EarlyStopping(**kwargs)
        case _:
            raise ValueError(f"Unknown callback name: {config['name']}")


def get_callbacks(configs: List[Dict]) -> List[Tuple[str, Callback]]:
    return [(config['name'], get_callback_by_name(config)) for config in configs if 'name' in config]

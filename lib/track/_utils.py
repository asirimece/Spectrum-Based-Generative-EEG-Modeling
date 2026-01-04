from omegaconf import DictConfig
from ._model import Tracker
from ._wandb import WandBTracker


def get_tracker_from_config(config: DictConfig) -> Tracker:
    tracker_type = config.type
    match tracker_type:
        case "wandb":
            return WandBTracker.from_config(config)
        case _:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
from lib.gen.model import GenTrainConfig, GanTrainConfig, DiffusionTrainConfig
from lib.gen.models.base import GAN
from lib.gen.models.cwgan import CWGAN
from lib.gen.models.ccdcwgan import CCDCWGAN
from lib.gen.models.ccwgan import CCWGAN
from lib.gen.models.diffusion.cddpm import CDDPM
from lib.gen.models.diffusion.icddpm import ICDDPM
from lib.gen.models.diffusion.base import Diffusion


def get_model(config: GenTrainConfig) -> GAN | Diffusion:
    if isinstance(config, GanTrainConfig):
        match config.model.name.lower():
            case "cwgan":
                return CWGAN.from_configs(config)
            case "ccwgan":
                return CCWGAN.from_configs(config)
            case "ccdcwgan":
                return CCDCWGAN.from_configs(config)
            case _:
                raise ValueError(f"Unsupported gan model {config.model.name}")
    elif isinstance(config, DiffusionTrainConfig):
        match config.model.name.lower():
            case "cddpm":
                return CDDPM.from_configs(config)
            case "icddpm":
                return ICDDPM.from_configs(config)
            case _:
                raise ValueError(f"Unsupported diffusion model {config.model.name}")
    else:
        raise ValueError(f"Unsupported config type {config}")

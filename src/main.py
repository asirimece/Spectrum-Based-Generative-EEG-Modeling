from src import gen
from src import base
from omegaconf import DictConfig
import hydra
from lib.utils import setup_omegaconf

setup_omegaconf()

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(config: DictConfig):
    if config.experiment.type.startswith("base"):
        base.run(config)
    elif config.experiment.type.startswith("gen"):
        gen.run(config)
    else:
        raise ValueError(f"Unknown experiment type: {config.experiment.type}")


if __name__ == "__main__":
    main()

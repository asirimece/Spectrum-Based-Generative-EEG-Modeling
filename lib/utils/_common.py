import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from lib.logging import setup_logging, get_logger
from dotenv import load_dotenv
import logging
from datetime import timedelta


def set_seed(seed: int):
    """Set seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

        # Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or
        # `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses
        # CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment
        # variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
        # CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

        # torch.use_deterministic_algorithms(True)


def null_safe_resolver(cfg, path):
    try:
        value = OmegaConf.select(cfg, path)
        return value if value is not None else None
    except Exception:
        return None


def setup_omegaconf():
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("as_tuple", tuple)
    OmegaConf.register_new_resolver("as_tuple_list", lambda x: [tuple(item) for item in x])
    OmegaConf.register_new_resolver("null_safe", null_safe_resolver, use_cache=False)


def setup_environment(config: DictConfig):
    load_dotenv()
    setup_logging(library_log_level=logging.ERROR)
    set_seed(config.seed)
    logger = get_logger()
    logger.info(f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")


def format_seconds(seconds: float) -> str:
    duration = timedelta(seconds=seconds)
    formatted_duration = "{:02}H:{:02}m:{:02}s".format(
        duration.seconds // 3600,  # hours
        (duration.seconds // 60) % 60,  # minutes
        duration.seconds % 60,  # seconds
    )

    if duration.days > 0:
        formatted_duration = "{}d:{}".format(duration.days, formatted_duration)
    return formatted_duration


def empty_func(*args, **kwargs):
    pass

import logging
from logging import getLevelName
from pathlib import Path
import os
import mne

DEFAULT_LOGGER_NAME = 'logger'
LIBRARY_LOG_LEVEL = logging.ERROR


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[97m',  # Blue
        'INFO': '\033[97m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m'  # Red
    }
    RESET = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.RESET}"


def setup_logger(log_dir: str | None = None, log_file='app.log', log_level=logging.INFO):
    if log_dir is None:
        log_dir = f"{os.getcwd()}/logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir.joinpath(log_file)
    if log_file_path.exists():
        os.remove(log_file_path)

    # Create a logger
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(log_level)

    # Create a file handler and set the level to the specified log level
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)

    # Create a console handler and set the level to the specified log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create a formatter and set the format for the handlers
    message_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    colored_formatter = ColoredFormatter(message_format, datefmt=date_format)
    simple_formatter = logging.Formatter(message_format, datefmt=date_format)
    file_handler.setFormatter(simple_formatter)
    console_handler.setFormatter(colored_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def setup_library_logging(log_level: int = logging.INFO):
    mne.set_config(key='MNE_LOGGING_LEVEL', value=getLevelName(log_level).lower())


def setup_logging(log_level: int = logging.INFO,
                  logger_log_level: int | None = None,
                  library_log_level: int | None = None):
    if logger_log_level is None:
        logger_log_level = log_level
    if library_log_level is None:
        library_log_level = LIBRARY_LOG_LEVEL
    logging.basicConfig(level=log_level)
    setup_library_logging(library_log_level)
    setup_logger(log_level=logger_log_level)


def get_logger():
    return logging.getLogger(DEFAULT_LOGGER_NAME)

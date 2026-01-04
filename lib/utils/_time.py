import time
from datetime import datetime, timedelta
from lib.logging import get_logger

class StopwatchStep:
    name: str
    idx: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __init__(self, name, idx: int = 0):
        self.name = name
        self.start_time = None
        self.idx = idx

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        self.end_time = datetime.now()

    def elapsed(self):
        return self.end_time - self.start_time if self.end_time is not None else datetime.now() - self.start_time


class Stopwatch:
    name: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    steps: dict[str, StopwatchStep] = {}
    logger = get_logger()

    def __init__(self, name):
        self.name = name
        self.start_time = None

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Call start() before stopping.")
        self.end_time = datetime.now()

    def add_step(self, step_name):
        if self.start_time is None:
            raise ValueError("Call start() before adding steps.")
        if step_name in self.steps:
            raise ValueError("Step name already exists.")
        step = StopwatchStep(step_name, len(self.steps) + 1)
        self.steps[step_name] = step
        step.start()

    def stop_step(self, step_name):
        if step_name not in self.steps:
            raise ValueError("Step name does not exist.")
        self.steps[step_name].stop()

    def summary(self):
        if self.start_time is None:
            raise ValueError("Call start() before getting summary.")
        if self.end_time is None:
            self.stop()
        total_elapsed = self.end_time - self.start_time
        self.logger.info(f"Stopwatch summary for {self.name}:")
        self.logger.info(f"Started at: {self.start_time}")
        self.logger.info("Steps:")
        sorted_steps = sorted(self.steps.values(), key=lambda x: x.idx)
        for step in sorted_steps:
            self.logger.info(f"{step.name}: {format_seconds(step.elapsed().total_seconds())}")
        self.logger.info(f"Stopped at: {self.end_time}")
        self.logger.info(f"Total elapsed: {format_seconds(total_elapsed.total_seconds())}")


def format_seconds(seconds: float, include_milliseconds: bool = False) -> str:
    duration = timedelta(seconds=seconds)
    formatted_duration = "{:02}H:{:02}m:{:02}s".format(
        duration.seconds // 3600,  # hours
        (duration.seconds // 60) % 60,  # minutes
        duration.seconds % 60,  # seconds
    )

    if duration.days > 0:
        formatted_duration = "{}d:{}".format(duration.days, formatted_duration)
    if include_milliseconds:
        formatted_duration += f":{duration.microseconds // 1000}ms"
    return formatted_duration

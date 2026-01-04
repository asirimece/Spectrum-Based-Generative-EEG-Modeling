from lib.experiment.run.base import RunConfig, Run
from lib.experiment.evaluation.result import EvaluatedResult, NeuralTrainResult
from omegaconf import DictConfig
from typing import Dict


class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """

    name: str

    priority = 'NORMAL'

    stages = ('before_experiment', 'before_run', 'after_run'
              'after_experiment', 'before_train_run', 'after_train_run',
              'after_run_epoch', 'before_val_run', 'after_val_run')

    def __init__(self, name: str):
        self.name = name

    def before_experiment(self, runner) -> None:
        """Called before the experiment starts."""
        pass

    def before_run(self, runner, run_config: RunConfig) -> None:
        """Called before each run."""
        pass

    def before_train_run(self, runner, run: Run) -> None:
        """Called before each training run."""
        pass

    def after_run_epoch(self, result: EvaluatedResult) -> None:
        """Called after an epoch of a training run if supported by pipeline."""
        pass

    def after_train_run(self, runner, run: Run, result: NeuralTrainResult) -> None:
        """Called after each training run."""
        pass

    def before_val_run(self, runner, run: Run) -> None:
        """Called before each validation run."""
        pass

    def after_val_run(self, runner, run: Run, result: EvaluatedResult) -> None:
        """Called after each validation run."""
        pass

    def after_run(self, runner, run: Run) -> None:
        """Called after each run."""
        pass

    def after_experiment(self, runner) -> None:
        """Called after the experiment ends."""
        pass

    @staticmethod
    def from_config(config: DictConfig) -> 'Hook':
        """Create a hook from a configuration."""
        pass
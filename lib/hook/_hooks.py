from lib.hook.base import Hook
from lib.track import Tracker, TrackerTarget, get_tracker_from_config
from typing import List
from lib.experiment.evaluation.result import NeuralTrainResult, EvaluatedResult, Result
from lib.experiment.pipeline.models import PipelineType
from lib.experiment.run.base import RunConfig, Run
from typing import Callable, Any, Dict
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from torch import nn
import joblib
from pathlib import Path
import torch
from lib.callback.skorch import Callbacks
from enum import Enum
import wandb
import pickle

SSR = Callable[[Any, RunConfig], bool]


class Hooks(Enum):
    TRACKER = "tracker"
    MODEL_SAVER = "model_saver"


class TrackerHook(Hook):
    trackers: List[Tracker] = []
    _should_track_run: SSR

    def __init__(self, name: str, trackers: List[Tracker], should_track_run: SSR | None = None):
        super().__init__(name)
        self.trackers = trackers
        self._should_track_run = should_track_run if should_track_run is not None else default_should_track_run

    def _get_trackers_by_target(self, target: TrackerTarget) -> List[Tracker]:
        return [tracker for tracker in self.trackers
                if tracker.config.target == target or tracker.config.target == TrackerTarget.ALL]

    def _log_evaluated_result(self, result: EvaluatedResult, target: TrackerTarget = TrackerTarget.RUN) -> None:
        for tracker in self._get_trackers_by_target(target):
            if tracker.initialized:
                if result.metrics is not None and len(result.metrics) > 0:
                    tracker.log(result.metrics)
                if result.visualizations is not None and len(result.visualizations) > 0:
                    tracker.log_image(result.visualisations_to_dict())

    def before_run(self, runner, run_config: RunConfig) -> None:
        for tracker in self._get_trackers_by_target(TrackerTarget.RUN):
            if not tracker.initialized and self._should_track_run(runner, run_config):
                tracker.init(name=run_config.name, config=run_config.to_dict())

    def before_train_run(self, runner, run: Run) -> None:
        for tracker in self._get_trackers_by_target(TrackerTarget.RUN):
            if tracker.initialized:
                tracker.auto_train_tracking = handle_wandb_logger(runner)

    def after_run_epoch(self, result: EvaluatedResult) -> None:
        self._log_evaluated_result(result, TrackerTarget.RUN)

    def after_train_run(self, runner, run: Run, result: Result | NeuralTrainResult | None) -> None:
        self._log_evaluated_result(result, TrackerTarget.RUN)

    def after_val_run(self, runner, run: Run, result: EvaluatedResult) -> None:
        self._log_evaluated_result(result, TrackerTarget.RUN)

    def after_run(self, runner, run: Run) -> None:
        for tracker in self._get_trackers_by_target(TrackerTarget.RUN):
            if tracker.initialized:
                tracker.set_run_configuration(run.config.to_dict())
                tracker.finish()

    def after_experiment(self, runner) -> None:
        for tracker in self._get_trackers_by_target(TrackerTarget.EXPERIMENT):
            experiment = runner.experiment
            tracker.init(name=experiment.name, config=experiment.to_summary_dict())
            tracker.log(experiment.metrics)
            tracker.log_image(experiment.media_to_dict())
            tracker.finish()

    @staticmethod
    def from_config(config: DictConfig) -> 'TrackerHook':
        trackers = [get_tracker_from_config(tracker_config) for tracker_config in config.trackers]
        return TrackerHook(
            name=config.name,
            trackers=trackers,
            should_track_run=default_should_track_run,
        )


def default_should_track_run(runner, run_config: RunConfig) -> bool:
    if run_config.subset != 'train' or runner.pipeline.type != PipelineType.TRADITIONAL:
        return True
    return False


def handle_wandb_logger(runner) -> bool:
    wandb_run = wandb.run
    if wandb_run is not None and runner.pipeline.type == PipelineType.NEURAL:
        if hasattr(runner.pipeline.pipeline, 'named_steps') and 'eegclassifier' in runner.pipeline.pipeline.named_steps:
            classifier = runner.pipeline.pipeline.named_steps['eegclassifier']
            callbacks = classifier.callbacks
            wandb_loggers = [tup for tup in callbacks if tup[0] == Callbacks.WAND_LOGGER.value]
            if len(wandb_loggers) > 0:
                classifier.set_params(callbacks__wandb_logger__wandb_run=wandb_run)
                return True
    return False


class ModelSaverHook(Hook):
    def __init__(self, name: str):
        super().__init__(name)

    def after_train_run(self, runner, run: Run, result: Result | NeuralTrainResult | None) -> None:
        if result is not None:
            if isinstance(result, NeuralTrainResult):
                if result.model is not None:
                    runner.logger.info(f"Saving model for run {run.name} from NeuralTrainResult")
                    save_module(result.model, run.config.work_dir, run.name)
            if result.module is not None:
                runner.logger.info(f"Saving module for run {run.name} from Result")
                save_module(result.module, run.config.work_dir, run.name)
            else:
                runner.logger.info(f"No model to save for run {run.name}")
        elif run.model is not None:
            runner.logger.info(f"Saving model for run {run.name} from Run")
            save_module(run.model, run.config.work_dir, run.name)
        else:
            runner.logger.info(f"No model to save for run {run.name}")

    @staticmethod
    def from_config(config: DictConfig) -> 'ModelSaverHook':
        return ModelSaverHook(name=config.name)


def save_module(module: Pipeline | nn.Module,
                storage_dir: str | Path = './work_dirs/modules',
                filename: str = "module"):
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)
    if isinstance(module, Pipeline):
        filename = filename + ".pkl"
        joblib.dump(module, storage_path.joinpath(filename))
    elif isinstance(module, nn.Module):
        filename = filename + ".pth"
        torch.save(module, storage_path.joinpath(filename), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise Exception("Unknown module type to save")

from lib.experiment.run.base import RunConfig, Run
from lib.experiment.base import Experiment
from lib.experiment.evaluation.experiment import ExperimentEvaluator, ExperimentVisualizer
from lib.experiment.pipeline.base import EEGPipeline
from lib.experiment.evaluation.experiment import (get_experiment_evaluators, get_experiment_visualizers)
from lib.utils import format_seconds
from lib.logging import get_logger
from typing import List
from omegaconf import DictConfig
from lib.utils import set_seed
from lib.logging import get_logger

logger = get_logger()

class ExperimentRunner:
    experiment: Experiment
    pipeline: EEGPipeline

    evaluators: List[ExperimentEvaluator] = []
    visualizers: List[ExperimentVisualizer] = []

    logger = get_logger()

    def __init__(self, experiment: Experiment, pipeline: EEGPipeline, seed: int = None):
        self.experiment = experiment
        self.pipeline = pipeline
        self.pipeline.set_runner(self)
        self.seed = seed

    def add_evaluators(self, evaluators: List[ExperimentEvaluator]):
        self.evaluators.extend(evaluators)

    def add_visualizers(self, visualizers: List[ExperimentVisualizer]):
        self.visualizers.extend(visualizers)

    def initialize_from_config(self, config: DictConfig):
        self.add_evaluators(get_experiment_evaluators(config.experiment.evaluators))
        self.add_visualizers(get_experiment_visualizers(config.experiment.visualizers))

    def execute_run(self, run_config: RunConfig) -> Run:
        if self.seed != run_config.seed:
            self.logger.info(f"Executing run with configuration: {run_config}")
            self.logger.info(f"Setting seed to {run_config.seed}")
            self.seed = run_config.seed
            set_seed(self.seed)
            
        self.logger.info(f"Starting pipeline execution for run: {run_config.name}")
        run = self.pipeline.run(run_config)
        self.logger.info(f"Pipeline execution completed for run: {run_config.name}")
        return run

    def execute_experiment(self):
        n_runs = self.experiment.plan.number_of_configs
        partial_result_idx = 0
        for i, run_config in enumerate(self.experiment.plan):
            self.logger.info(f"----------------------------------\n")
            self.logger.info(f"Starting run {i + 1}/{n_runs}")
            run = self.execute_run(run_config)
            self.logger.info(f"Finished Run {i + 1}/{n_runs} in {format_seconds(run.duration)}")
            remaining_runs = n_runs - (i + 1)
            self.logger.info(f"Remaining runs: {remaining_runs}")
            self.logger.info(f"ETAs: {format_seconds(remaining_runs * run.duration)}\n")
            self.experiment.runs.append(run)
            self.experiment.plan.finish()
            partial_result_idx = self.evaluate_partial_result(run_config, partial_result_idx)


    def evaluate_metrics(self):
        self.logger.info(f"Evaluating metrics for experiment: {self.experiment.name}")
        metrics = {}
        for evaluator in self.evaluators:
            metrics.update(evaluator.evaluate(self.experiment))
        self.experiment.metrics = metrics
        self.logger.info(f"Metrics evaluation completed.")

    def evaluate_visuals(self):
            self.logger.info(f"Generating visualizations for experiment: {self.experiment.name}")
            visuals = []
            for visualizer in self.visualizers:
                result = visualizer.evaluate(self.experiment)
                if isinstance(result, list):
                    visuals.extend(result)
                else:
                    visuals.append(result)
            self.experiment.media = visuals
            self.logger.info(f"Visualizations generation completed.")


    def evaluate_experiment(self):
        self.logger.info(f"Evaluating experiment: {self.experiment.name}")
        self.evaluate_metrics()
        self.evaluate_visuals()

    def evaluate_partial_result(self, run_config: RunConfig, idx: int) -> int:
        if self.experiment.partial_result_key is not None and self.experiment.partial_result_key in run_config.idx:
            new_idx = run_config.idx[self.experiment.partial_result_key]
            if new_idx != idx:
                self.logger.info(f"Evaluating partial result for experiment: {self.experiment.name}")
                self.evaluate_experiment()
                self.save_experiment()
                self.experiment.media = []
                self.experiment.metrics = {}
                return new_idx
        return idx

    def save_experiment(self):
        self.logger.info(f"Saving experiment: {self.experiment.name}")
        self.experiment.save()

    def _call_hook(self, fn_name: str, **kwargs) -> None:
        for hook in self.pipeline.hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(**kwargs)

    def run(self):
        self._call_hook("before_experiment", runner=self)
        self.experiment.start()
        self.execute_experiment()
        self.evaluate_experiment()
        self.experiment.finish()
        self.save_experiment()
        self._call_hook("after_experiment", runner=self)

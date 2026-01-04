from lib.experiment.base import Experiment
from lib.experiment.evaluation.experiment.metrics import (
    get_augmentation_level_evaluation,
    get_augmentation_subject_level_evaluation,
    get_generic_augmentation_evaluation,
    get_tuning_level_evaluation,
    AugmentationMetricEvaluation
)
from lib.experiment.evaluation import Evaluator
from enum import Enum
from omegaconf import DictConfig
from typing import List, Dict


class ExperimentEvaluator(Evaluator):

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        pass


class ExperimentMetrics(Enum):
    GENERIC_EVALUATOR = 'generic_evaluator'
    AUGMENTATION_TUNING = 'augmentation_tuning_evaluator'
    MODEL_TUNING_PARAM = 'model_tuning_param_evaluator'
    MODEL_TUNING_BEST_PARAM = 'model_tuning_best_param_evaluator'
    BEST_RUN = 'best_run_evaluator'


class GenericEvaluator(ExperimentEvaluator):

    def __init__(self, keys: List[str]):
        super().__init__(name=ExperimentMetrics.GENERIC_EVALUATOR.value)
        self.keys = keys

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        structure = experiment.plan.structure
        if len(self.keys) > 0 and all([key in structure for key in self.keys]):
            evaluation = AugmentationMetricEvaluation()
            evaluation.augmentation = get_augmentation_level_evaluation(experiment.runs)
            if 'subject' in structure:
                evaluation.augmentation_subject = get_augmentation_subject_level_evaluation(experiment.runs)
            return {self.name: evaluation}
        return {}


class AugmentationTuningEvaluator(ExperimentEvaluator):

    def __init__(self):
        super().__init__(name=ExperimentMetrics.AUGMENTATION_TUNING.value)

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        structure = experiment.plan.structure
        if 'augmentation' in structure:
            evaluation = AugmentationMetricEvaluation()
            evaluation.augmentation = get_augmentation_level_evaluation(experiment.runs)
            if 'subject' in structure:
                evaluation.augmentation_subject = get_augmentation_subject_level_evaluation(experiment.runs)
            return {self.name: evaluation}
        return {}


class GenericAugmentationEvaluator(ExperimentEvaluator):

    def __init__(self, keys: List[str]):
        super().__init__(name=ExperimentMetrics.GENERIC_EVALUATOR.value)
        self.keys = keys

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        structure = experiment.plan.structure
        if len(self.keys) > 0 and all([key in structure for key in self.keys]):
            result = get_generic_augmentation_evaluation(experiment.runs, keys=self.keys)
            return {self.name: result}
        return {}


class ModelTuningParameterEvaluator(ExperimentEvaluator):

    def __init__(self):
        super().__init__(name=ExperimentMetrics.MODEL_TUNING_PARAM.value)

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        if len(experiment.runs) > 0:
            filtered_runs = [run for run in experiment.runs
                             if run.config.subset == 'test' and run.config.pipeline_params is not None]
            if len(filtered_runs) > 0:
                params = {}
                for run in filtered_runs:
                    for key, value in run.config.pipeline_params.items():
                        if key in params:
                            params[key].append(value)
                        else:
                            params[key] = [value]
                return {self.name: params}
        return {}


class ModelTuningBestParameterEvaluator(ExperimentEvaluator):
    metric: str
    subsets: list[str] | None

    def __init__(self, metric: str = 'balanced_accuracy', subsets: list[str] | None = None):
        super().__init__(name=ExperimentMetrics.MODEL_TUNING_BEST_PARAM.value)
        self.metric = metric
        self.subsets = subsets

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        structure = experiment.plan.structure
        if ('tuning' in structure or 'augmentation' in structure or 'single' in structure) and len(experiment.runs) > 0:
            key = 'tuning' if 'tuning' in structure else 'augmentation' if 'augmentation' in structure else 'single'
            evaluation = get_tuning_level_evaluation(experiment.runs, self.subsets, self.metric, key=key)
            return {self.name: evaluation}
        return {}


class BestRunEvaluator(ExperimentEvaluator):
    metric: str
    subsets: list[str] | None

    def __init__(self, metric: str = 'balanced_accuracy', subsets: list[str] | None = None):
        super().__init__(name=ExperimentMetrics.BEST_RUN.value)
        self.metric = metric
        self.subsets = subsets

    def evaluate(self, experiment: Experiment) -> dict[str, any]:
        structure = experiment.plan.structure
        if 'augmentation' in structure or 'tuning' in structure or 'single' in structure:
            key = 'augmentation' if 'augmentation' in structure else 'tuning' if 'tuning' in structure else 'single'
            evaluation = get_tuning_level_evaluation(experiment.runs, self.subsets, self.metric, key=key)
            metrics = {f"best_{key}": value for key, value in evaluation.best_metrics.items()}
            if evaluation.best_run is not None:
                metrics['best_run'] = evaluation.best_run.name
            return metrics
        return {}


def get_experiment_evaluator_by_name(name: str, **kwargs) -> ExperimentEvaluator:
    match name:
        case ExperimentMetrics.AUGMENTATION_TUNING.value:
            return AugmentationTuningEvaluator()
        case ExperimentMetrics.MODEL_TUNING_PARAM.value:
            return ModelTuningParameterEvaluator()
        case ExperimentMetrics.MODEL_TUNING_BEST_PARAM.value:
            return ModelTuningBestParameterEvaluator(**kwargs)
        case ExperimentMetrics.BEST_RUN.value:
            return BestRunEvaluator(**kwargs)
        case _:
            raise ValueError(f"Unknown evaluator name {name}")


def get_experiment_evaluators(evaluators: List[str | Dict | DictConfig]) -> list[ExperimentEvaluator]:
    if len(evaluators) == 0:
        return []
    if isinstance(evaluators[0], dict | DictConfig):
        return [get_experiment_evaluator_by_name(**evaluator) for evaluator in evaluators]
    else:
        return [get_experiment_evaluator_by_name(name) for name in evaluators]

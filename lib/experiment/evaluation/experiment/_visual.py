from lib.experiment.evaluation import Visualizer
from lib.experiment.evaluation.result import VisualResult
from lib.experiment.evaluation.experiment import AugmentationTuningEvaluator, GenericAugmentationEvaluator
from lib.experiment.evaluation.experiment.metrics import (AugmentationMetricEvaluation, get_tuning_level_evaluation,
                                                          AugMetricLevel)
from lib.experiment.evaluation.experiment.visual import (visualize_augmentation_result,
                                                         visualize_averaged_augmentation_result,
                                                         visualize_subject_path_augmentation_result,
                                                         visualize_ensemble_augmentation_result)
from enum import Enum
from typing import List, Dict
from omegaconf import DictConfig
from lib.experiment.run.base import Run
import numpy as np
import matplotlib.pyplot as plt


class ExperimentVisualizer(Visualizer):

    def evaluate(self, experiment: any) -> VisualResult | List[VisualResult]:
        pass


class VisualExperimentMetrics(Enum):
    AUGMENTATION_TUNING = 'augmentation_tuning_visualizer'
    AVERAGED_AUGMENTATION_TUNING = 'averaged_augmentation_tuning_visualizer'
    ENSEMBLE_AUGMENTATION_TUNING = 'ensemble_augmentation_tuning_visualizer'
    SUBJECT_PATH_AUGMENTATION_TUNING = 'subject_path_augmentation_tuning_visualizer'
    BEST_RUN = 'best_run_visualizer'
    HYPERPARAMETER = 'hyperparameter_visualizer'


class AugmentationTuningVisualizer(ExperimentVisualizer):

    def __init__(self):
        super().__init__(name=VisualExperimentMetrics.AUGMENTATION_TUNING.value)

    def evaluate(self, experiment: any) -> VisualResult:
        structure = experiment.plan.structure
        if 'augmentation' in structure:
            evaluator = AugmentationTuningEvaluator()
            evaluation: AugmentationMetricEvaluation = evaluator.evaluate(experiment)[evaluator.name]
            fig, df = visualize_augmentation_result(evaluation)
            plt.close(fig)
            return VisualResult(name=self.name, data=fig, dataframe=df)
        return VisualResult(name=self.name, data=None)


class AveragedAugmentationTuningVisualizer(ExperimentVisualizer):
    n_classes: int

    def __init__(self, n_classes: int = 2):
        super().__init__(name=VisualExperimentMetrics.AVERAGED_AUGMENTATION_TUNING.value)
        self.n_classes = n_classes

    def evaluate(self, experiment: any) -> VisualResult:
        structure = experiment.plan.structure
        if 'augmentation' in structure:
            evaluator = AugmentationTuningEvaluator()
            evaluation: AugmentationMetricEvaluation = evaluator.evaluate(experiment)[evaluator.name]
            baseline = 1 / self.n_classes
            fig = visualize_averaged_augmentation_result(evaluation, baseline=baseline)
            plt.close(fig)
            return VisualResult(name=self.name, data=fig)
        return VisualResult(name=self.name, data=None)


class EnsembleAugmentationTuningVisualizer(ExperimentVisualizer):
    n_classes: int

    def __init__(self, n_classes: int = 2):
        super().__init__(name=VisualExperimentMetrics.ENSEMBLE_AUGMENTATION_TUNING.value)
        self.n_classes = n_classes

    def evaluate(self, experiment: any) -> VisualResult:
        structure = experiment.plan.structure
        if 'augmentation' in structure and 'ensemble' in structure:
            evaluator = GenericAugmentationEvaluator(keys=['ensemble', 'augmentation', 'subject'])
            evaluation: AugMetricLevel = evaluator.evaluate(experiment)[evaluator.name]
            fig = visualize_ensemble_augmentation_result(evaluation)
            plt.close(fig)
            return VisualResult(name=self.name, data=fig)
        return VisualResult(name=self.name, data=None)


class BestRunVisualizer(ExperimentVisualizer):
    metric: str
    subsets: list[str] | None

    def __init__(self, metric: str = 'balanced_accuracy', subsets: list[str] | None = None):
        super().__init__(name=VisualExperimentMetrics.BEST_RUN.value)
        self.metric = metric
        self.subsets = subsets

    def evaluate(self, experiment: any) -> List[VisualResult]:
        structure = experiment.plan.structure
        if 'augmentation' in structure or 'tuning' in structure or 'single' in structure:
            key = 'augmentation' if 'augmentation' in structure else 'tuning' if 'tuning' in structure else 'single'
            evaluation = get_tuning_level_evaluation(experiment.runs, self.subsets, self.metric, key=key)
            if evaluation is not None and evaluation.best_run is not None:
                media = evaluation.best_run.load_media()
                return media
        return []


class SubjectPathAugmentationTuningVisualizer(ExperimentVisualizer):
    n_classes: int

    def __init__(self, n_classes: int = 2):
        super().__init__(name=VisualExperimentMetrics.SUBJECT_PATH_AUGMENTATION_TUNING.value)
        self.n_classes = n_classes

    def evaluate(self, experiment: any) -> VisualResult:
        structure = experiment.plan.structure
        if 'augmentation' in structure:
            evaluator = AugmentationTuningEvaluator()
            evaluation: AugmentationMetricEvaluation = evaluator.evaluate(experiment)[evaluator.name]
            baseline = 1 / self.n_classes
            fig = visualize_subject_path_augmentation_result(evaluation, baseline=baseline)
            plt.close(fig)
            return VisualResult(name=self.name, data=fig)
        return VisualResult(name=self.name, data=None)


class HyperparameterVisualizer(ExperimentVisualizer):
    parameter: str
    metric: str
    subsets: list[str] | None

    def __init__(self,
                 parameter: str = 'eegclassifier__optimizer__lr',
                 metric: str = 'balanced_accuracy',
                 subsets: list[str] | None = None
                 ):
        super().__init__(name=VisualExperimentMetrics.HYPERPARAMETER.value)
        self.parameter = parameter
        self.metric = metric
        self.subsets = subsets if subsets is not None else ['test']

    def group_runs_by_parameter(self, runs: List[Run]) -> Dict:
        if len(runs) > 0:
            grouped_runs = {}
            for run in runs:
                parameter_value = run.config.pipeline_params[self.parameter] \
                    if self.parameter in run.config.pipeline_params else 0
                metric_value = run.metrics[self.metric] if self.metric in run.metrics else 0
                if parameter_value in grouped_runs:
                    grouped_runs[parameter_value].append(metric_value)
                else:
                    grouped_runs[parameter_value] = [metric_value]
            return grouped_runs
        return {}

    def evaluate(self, experiment: any) -> VisualResult:
        runs = experiment.runs
        if len(runs) > 0:
            filtered_runs = [run for run in runs if run.config.subset in self.subsets and
                             run.config.pipeline_params is not None and self.parameter in run.config.pipeline_params]
            if len(filtered_runs) > 0:
                grouped_runs = self.group_runs_by_parameter(filtered_runs)
                averaged_metrics = {key: np.mean(values) for key, values in grouped_runs.items()}
                sorted_keys = sorted(averaged_metrics.keys())
                sorted_values = [averaged_metrics[key] for key in sorted_keys]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(sorted_keys, np.round(sorted_values, 4))
                ax.set_xscale('log')
                ax.set_xlabel(self.parameter)
                ax.set_ylabel(self.metric)
                ax.set_title(f"Hyperparameter tuning: {self.parameter}")
                plt.close(fig)
                return VisualResult(name=self.name, data=fig)
        return VisualResult(name=self.name, data=None)


def get_experiment_visualizer_by_name(name: str, **kwargs) -> ExperimentVisualizer:
    match name:
        case VisualExperimentMetrics.AUGMENTATION_TUNING.value:
            return AugmentationTuningVisualizer()
        case VisualExperimentMetrics.AVERAGED_AUGMENTATION_TUNING.value:
            return AveragedAugmentationTuningVisualizer()
        case VisualExperimentMetrics.SUBJECT_PATH_AUGMENTATION_TUNING.value:
            return SubjectPathAugmentationTuningVisualizer()
        case VisualExperimentMetrics.BEST_RUN.value:
            return BestRunVisualizer(**kwargs)
        case VisualExperimentMetrics.HYPERPARAMETER.value:
            return HyperparameterVisualizer(**kwargs)
        case VisualExperimentMetrics.ENSEMBLE_AUGMENTATION_TUNING.value:
            return EnsembleAugmentationTuningVisualizer()
        case _:
            raise ValueError(f"Unknown visualizer name {name}")


def get_experiment_visualizers(visualizers: List[str | Dict | DictConfig]) -> List[ExperimentVisualizer]:
    if len(visualizers) == 0:
        return []
    if isinstance(visualizers[0], dict | DictConfig):
        return [get_experiment_visualizer_by_name(**visualizer) for visualizer in visualizers]
    else:
        return [get_experiment_visualizer_by_name(name) for name in visualizers]

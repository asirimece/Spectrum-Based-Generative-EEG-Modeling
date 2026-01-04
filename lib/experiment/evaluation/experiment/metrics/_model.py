from lib.experiment.run.base import Run
from typing import List, Dict
from lib.experiment.evaluation.experiment.utils import (
    aggregate_runs_metrics,
    group_runs_by_index_key,
    group_runs_by_index_keys
)


class TuningLevelMetric:
    best_metrics: Dict[str, any]
    best_params: Dict[str, any]
    best_run: Run | None

    def __init__(self, best_metrics: Dict[str, any], best_params: Dict[str, any], best_run: Run | None = None):
        self.best_metrics = best_metrics
        self.best_params = best_params
        self.best_run = best_run

    def to_dict(self):
        return {
            "best_metrics": self.best_metrics,
            "best_params": self.best_params,
            "best_run": self.best_run.name if self.best_run is not None else None
        }


def get_tuning_level_evaluation(runs: list[Run],
                                subsets: List[str] | None = None,
                                metric: str = 'balanced_accuracy',
                                key: str = 'tuning') -> TuningLevelMetric:
    if subsets is None:
        subsets = ['test']
    grouped = group_runs_by_index_key(runs, key, subsets)
    best_metrics = {}
    best_params = {}
    best_run: Run | None = None
    for iteration, runs in grouped.items():
        metrics = aggregate_runs_metrics(runs, 'mean')
        if metric in metrics:
            if metric in best_metrics:
                if metrics[metric] > best_metrics[metric]:
                    best_metrics = metrics
                    best_params = runs[0].config.pipeline_params
                    best_run = runs[0]
            else:
                best_run = runs[0]
                best_metrics = metrics
                best_params = runs[0].config.pipeline_params

    return TuningLevelMetric(best_metrics=best_metrics, best_params=best_params, best_run=best_run)

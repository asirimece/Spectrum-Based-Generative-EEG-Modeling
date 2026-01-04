from lib.experiment.run.base import Run
from typing import List, Dict, Callable
import numpy as np


def filter_runs_by_index_key(runs: list[Run], key: str, value: int, subsets: List[str] | None = None) -> list[Run]:
    if subsets is None:
        subsets = ['test']
    return [run for run in runs if run.idx[key] == value and run.subset in subsets]


def group_runs_by_index_key(runs: list[Run], key: str, subsets: List[str] | None = None) -> Dict[int, list[Run]]:
    if subsets is None:
        subsets = ['test']
    grouped = {}
    for run in runs:
        if run.subset in subsets:
            if run.idx[key] not in grouped:
                grouped[run.idx[key]] = []
            grouped[run.idx[key]].append(run)
    return grouped


def group_runs_by_index_keys(runs: list[Run] | Dict, keys: List[str], subsets: List[str] | None = None) -> Dict[
    int, any]:
    if subsets is None:
        subsets = ['test']
    if isinstance(runs, list):
        runs: Dict = group_runs_by_index_key(runs, keys[0], subsets)
    if len(keys) == 1:
        return runs
    else:
        for group_key in runs.keys():
            runs[group_key] = group_runs_by_index_keys(runs[group_key], keys[1:], subsets)
    return runs


def aggregate_runs_metrics(runs: list[Run], aggregator: str | Callable = 'mean') -> dict:
    metrics = {}
    for run in runs:
        for metric in run.metrics.keys():
            if metric not in metrics:
                if run.metrics[metric] is None or isinstance(run.metrics[metric], (dict, str)):
                    continue
                metrics[metric] = []
            metrics[metric].append(run.metrics[metric])
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            return {metric: np.mean(metrics[metric]) for metric in metrics}
        elif aggregator == 'median':
            return {metric: np.median(metrics[metric]) for metric in metrics}
        else:
            raise ValueError(f"Aggregator string {aggregator} is not supported")
    elif isinstance(aggregator, Callable):
        return {metric: aggregator(metrics[metric]) for metric in metrics}
    else:
        raise ValueError(f"Aggregator type {aggregator} is not supported")

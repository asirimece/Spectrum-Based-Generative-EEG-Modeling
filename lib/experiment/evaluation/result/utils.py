from ._model import Result, EvaluatedResult, VisualResult
from lib.experiment.evaluation import Evaluator, Visualizer
from typing import List, Dict
from lib.logging import get_logger
from lib.experiment.evaluation.result import GenResult
import torch
import mne
from inspect import signature, Parameter
import numpy as np

logger = get_logger()


def evaluate_result(result: Result,
                    evaluators: List[Evaluator],
                    visualizers: List[Visualizer],
                    **kwargs
                    ) -> EvaluatedResult:
    metrics: Dict[str, any] = {}
    for evaluator in evaluators:
        evaluation_result: dict = evaluator.evaluate(result, **kwargs)
        metrics.update(evaluation_result)

    visualizations: List[VisualResult] = []
    for visualizer in visualizers:
        visual_results: VisualResult | List[VisualResult] | None = None
        try:
            visual_results = visualizer.evaluate(result, **kwargs)
        except Exception as e:
            logger.error(f"Error during visualization: {visualizer.name}")
            logger.error(e)
        if visual_results is not None:
            if isinstance(visual_results, list):
                visualizations.extend(visual_results)
            else:
                visualizations.append(visual_results)
    
    return EvaluatedResult(metrics=metrics, visualizations=visualizations, **result.to_dict())

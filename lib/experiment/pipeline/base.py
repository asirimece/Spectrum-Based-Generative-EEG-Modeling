from sklearn.pipeline import Pipeline
from lib.logging import get_logger
from lib.experiment.run.base import RunConfig
from lib.config import DictInit, config_to_primitive
from ._steps import get_pipeline_steps, resolve_dynamic_step_args
from lib.experiment.evaluation.run import RunEvaluator, RunVisualizer, get_run_evaluators, get_run_visualizers
from lib.experiment.evaluation.result import EvaluatedResult, NeuralTrainResult
from lib.experiment.pipeline.models import PipelineType
from lib.dataset import TorchBaseDataset, BaseDataset
from lib.decorator import assert_called_before, LogFunctionCallsMeta
from typing import List
from omegaconf import DictConfig, ListConfig
from lib.experiment.run.base import Run
from lib.hook.base import Hook
from lib.hook import get_hooks_from_configs
import numpy as np
from lib.logging import get_logger

logger = get_logger()

class EEGPipeline(metaclass=LogFunctionCallsMeta):
    name: str

    pipeline: Pipeline

    hooks: List[Hook] = []

    steps: List = []

    evaluators: List[RunEvaluator] = []
    visualizers: List = []

    dataset: TorchBaseDataset

    type: PipelineType = PipelineType.TRADITIONAL

    _runner: any

    logger = get_logger()

    def __init__(self, name: str = "EEGPipeline", pipeline_type: PipelineType = PipelineType.TRADITIONAL):
        self.name = name
        self.logger = get_logger()
        self.type = pipeline_type
        self.logger.info(f"Initializing pipeline {self.name}")

    def set_dataset(self, dataset: TorchBaseDataset | BaseDataset):
        self.dataset = dataset
        return self

    def set_runner(self, runner: any):
        self._runner = runner
        return self

    def load_data(self, config: RunConfig):
        self.dataset.load_by_config(config)
        return self

    def add_hooks(self, hooks: List[Hook] | List[DictConfig], replace: bool = False):
        if len(hooks) > 0 and not isinstance(hooks[0], Hook):
            hooks = get_hooks_from_configs(hooks)
        if replace:
            self.hooks = []
        self.hooks.extend(hooks)
        return self

    def _call_hook(self, fn_name: str, **kwargs) -> None:
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(**kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    @assert_called_before("set_dataset")
    def add_steps(self, steps: List | List[DictInit] | ListConfig):
        if len(steps) > 0 and (type(steps[0]) is DictInit or type(steps[0]) is DictConfig):
            steps = config_to_primitive(steps)
            steps = resolve_dynamic_step_args(steps, self.dataset)
            steps = get_pipeline_steps(steps)
        self.steps.extend(steps)
        return self

    def add_evaluators(self, evaluators: List[RunEvaluator] | List[str]):
        if len(evaluators) > 0 and isinstance(evaluators[0], str):
            evaluators = get_run_evaluators(evaluators)
        self.evaluators.extend(evaluators)
        return self

    def add_visualizers(self, visualizers: List[RunVisualizer] | List[str]):
        if len(visualizers) > 0 and isinstance(visualizers[0], str):
            visualizers = get_run_visualizers(visualizers)
        self.visualizers.extend(visualizers)
        return self

    def _save_run(self, run: Run, result: EvaluatedResult | NeuralTrainResult = None) -> Run:
        self.logger.info(f"Saving run {run.name}")
        if result is not None:
            self.logger.info(f"Saving {len(result.metrics)} metrics.")
            self.logger.info(f"Saving {len(result.visualizations)} visualizations.")
            """run.metrics = result.metrics
            run.media = result.visualizations
            run.history = result.history if hasattr(result, "history") else None"""
            # Convert metrics and related data to native Python types
            run.metrics = convert_numpy_types(result.metrics)
            run.media = convert_numpy_types(result.visualizations)
            run.history = convert_numpy_types(result.history) if hasattr(result, "history") else None

        run.save()
        return run

    def build(self):
        pass

    @assert_called_before("set_dataset")
    def preprocess(self, *args, **kwargs):
        pass

    @assert_called_before("build")
    @assert_called_before("set_dataset")
    def train(self, *args, **kwargs):
        pass

    @assert_called_before("train")
    def evaluate(self):
        pass

    @assert_called_before("build")
    def run(self, *args, **kwargs):
        pass

def convert_numpy_types(obj):
    """
    Recursively convert numpy data types in a dictionary or list to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj
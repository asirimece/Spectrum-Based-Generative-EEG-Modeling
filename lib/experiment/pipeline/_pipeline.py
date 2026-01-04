from sklearn.pipeline import Pipeline
from lib.logging import get_logger
from lib.transform import TransformStep
from ._utils import get_best_values_from_history
from lib.experiment.evaluation.result import NeuralTrainResult, Result
from lib.experiment.evaluation.result.utils import evaluate_result
from lib.experiment.evaluation.run import RunEvaluator
from sklearn.pipeline import make_pipeline
from lib.dataset import BaseDataset
from numpy import ndarray
from lib.decorator import assert_called_before
from typing import List, Dict
import numpy as np
from lib.experiment.run.base import RunConfig, Run
from lib.experiment.pipeline.models import PipelineType, PipelineMode
from lib.experiment.pipeline.base import EEGPipeline
from lib.hook.base import Hook
from lib.experiment.model import get_tuner
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lib.experiment import ExperimentType

logger = get_logger()


class AugPipeline(EEGPipeline):
    name: str

    pipeline: Pipeline

    hooks: List[Hook] = []

    steps: List = []

    evaluators: List[RunEvaluator] = []
    visualizers: List = []

    dataset: BaseDataset

    X: ndarray | None
    y: ndarray | None

    pipeline_params: Dict[str, any] | None = None

    def __init__(self, name: str = "AugPipeline", pipeline_type: PipelineType = PipelineType.TRADITIONAL):
        super().__init__(name, pipeline_type)

    def build(self):
        self.logger.info(f"Building pipeline {self.name} with steps:")
        self.logger.info(f"\n {self.steps}")
        self.pipeline = make_pipeline(*self.steps)
        return self.pipeline

    def __add_dynamic_preprocessors(self, config: RunConfig, update_existing: bool = True):
        if config.e_type == ExperimentType.AUG_AUGMENTATION_TUNING:
            ta_config = config.tuning_augmentation_config
            if ta_config is not None and ta_config.optimization_parameter in ta_config.kwargs:
                if ta_config.kwargs[ta_config.optimization_parameter] != ta_config.op_neutral_value:
                    self.dataset.add_epochs_transforms([config.tuning_augmentation_config.to_dict()],
                                                       update_existing=update_existing)
                else:
                    self.dataset.remove_epochs_transforms([config.tuning_augmentation_config.name])

    def __update_dynamic_config(self, run: Run) -> Run:
        config = run.config
        self.__add_dynamic_preprocessors(config)
        if self.pipeline_params is not None:
            config.pipeline_params = self.pipeline_params
        elif (config.model_tuning is not None and config.model_tuning.enabled and config.pipeline_params is not None and
              run.config.model_tuning.scope == 'generator' and run.config.subset == 'train'):
            self.pipeline.set_params(**config.pipeline_params)
        return run

    def __split_epochs(self, run_config: RunConfig):
        if run_config.subset_indexes is not None:
            self.dataset.reset_epochs()
            self.dataset.epochs = self.dataset.epochs[run_config.subset_indexes]

    @assert_called_before("set_dataset")
    def preprocess(self, mode: PipelineMode = PipelineMode.TRAIN, run_config: RunConfig = None):
        self.__split_epochs(run_config)
        self.dataset.epoch_process(transform_step=TransformStep.PREPROCESS)
        if mode == PipelineMode.TRAIN:
            logger.info(f"Preprocessing epochs for run {run_config.name}")
            self.dataset.epoch_process(transform_step=TransformStep.AUGMENT)
        self.dataset.epoch_process(transform_step=TransformStep.POST_AUGMENT)
        self.logger.info(f"Finished preprocessing for run {run_config.name}")
        return self

    @assert_called_before("set_dataset")
    @assert_called_before("preprocess")
    def retrieve_epochs(self, config: RunConfig):
        X = self.dataset.data
        y = self.dataset.labels

        self.X, self.y = X, y

    def __get_final_pipeline(self, config: RunConfig) -> GridSearchCV | RandomizedSearchCV:
        return get_tuner(pipeline=self.pipeline, config=config.model_tuning)

    @assert_called_before("build")
    @assert_called_before("set_dataset")
    def train(self, run: Run) -> (None | Result, Run):
        self.logger.info("Starting training")
        model_tuning = run.config.model_tuning
        if model_tuning is not None and model_tuning.enabled and model_tuning.scope == 'pipeline':
            pipeline = self.__get_final_pipeline(run.config)
            pipeline.fit(self.X, self.y)
            self.pipeline = pipeline.best_estimator_
            run.config.pipeline_params = pipeline.best_params_
            self.pipeline_params = pipeline.best_params_
        else:
            self.pipeline.fit(self.X, self.y)
        self.logger.info("Finished training")
        return None, run


    @assert_called_before("train")
    def evaluate(self):
        self.logger.info("Starting evaluation")
        proba = self.pipeline.predict_proba(self.X)
        pred = np.argmax(proba, axis=1)

        result: Result = Result(scores=proba, pred=pred, gt=self.y, labels=self.dataset.class_names,
                                module=self.pipeline)

        evaluated_result = evaluate_result(result, self.evaluators, self.visualizers)
        self.logger.info("Finished evaluation")
        evaluated_result.summary()

        return evaluated_result


    def run(self, run_config: RunConfig) -> Run:
        self._call_hook("before_run", runner=self._runner, run_config=run_config)

        run: Run = Run.from_run_config(run_config)
        run.start()
        mode = PipelineMode.from_string(run_config.subset)

        run = self.__update_dynamic_config(run)
        self.load_data(run_config)
        self.preprocess(mode, run_config)
        self.retrieve_epochs(run_config)

        if mode == PipelineMode.TRAIN:
            self._call_hook("before_train_run", runner=self._runner, run=run)
            result, run = self.train(run=run)
            self._call_hook("after_train_run", runner=self._runner, run=run, result=result)
        else:
            self._call_hook("before_val_run", runner=self._runner, run=run)
            result = self.evaluate()
            self._call_hook("after_val_run", runner=self._runner, run=run, result=result)

        run.finish()
        run = self._save_run(run, result)

        self._call_hook("after_run", runner=self._runner, run=run)

        return run


class NeuralAugPipeline(AugPipeline):
    type: PipelineType = PipelineType.NEURAL

    def __init__(self, name: str = "NeuralEEGPipeline"):
        super().__init__(name, PipelineType.NEURAL)

    def train(self, run: Run) -> (NeuralTrainResult, Run):
        _, run = super().train(run=run)
        history = None
        model = None
        if 'eegclassifier' in self.pipeline.named_steps:
            history = self.pipeline.named_steps['eegclassifier'].history
            model = self.pipeline.named_steps['eegclassifier'].module_

        metrics = get_best_values_from_history(history)

        return NeuralTrainResult(history=history, model=model, module=self.pipeline, metrics=metrics), run

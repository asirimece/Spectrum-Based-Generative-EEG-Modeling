from lib.gen.trainer.base import Trainer
from lib.gen.models.diffusion.base import Diffusion
from lib.gen import DiffusionTrainConfig, Training
from lib.dataset import TorchBaseDataset
from lib.experiment.evaluation.run import RunEvaluator, RunVisualizer
from lib.utils import empty_func
from typing import Callable
from lib.experiment.run.base import Run
from lib.experiment.evaluation.result import EvaluatedResult


class DiffusionTrainer(Trainer):
    diffusion: Diffusion
    dataset: TorchBaseDataset
    config: DiffusionTrainConfig

    def __init__(self,
                 config: DiffusionTrainConfig,
                 diffusion: Diffusion,
                 dataset: TorchBaseDataset,
                 visualizers: list[RunVisualizer] | None = None,
                 evaluators: list[RunEvaluator] | None = None,
                 call_hook: Callable = empty_func):
        self.config = config
        self.diffusion = diffusion
        self.dataset = dataset
        self.visualizers = visualizers or []
        self.evaluators = evaluators or []
        self.initialize()
        self.call_hook = call_hook

    def initialize(self):
        self.diffusion.initialize(config=self.config, dataset=self.dataset)

    def run(self, run: Run, config: DiffusionTrainConfig) -> (Run, EvaluatedResult):
        self.logger.info(f"Trainer: Starting {config.model.name} Training")
        self.diffusion.test_shapes()
        training: Training = Training(config=config)

        for epoch in range(config.trainer.num_epochs):
            self.logger.info(f"Trainer: Starting Epoch {epoch + 1}/{config.trainer.num_epochs}")
            self.diffusion.reset_history()
            training.start_epoch(epoch=epoch)
            training = self.diffusion.run_training(config=config, training=training)
            training = self.diffusion.run_validation(config=config, training=training)

            training.result.losses = self.diffusion.get_averaged_losses()
            training.finish_epoch()
            self.log_epoch(training, config)

        result: EvaluatedResult = self.evaluate(training=training, run=run, model=self.diffusion)
        self.diffusion.clean_up()
        return run, result


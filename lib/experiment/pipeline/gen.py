import mne
import numpy as np
from lib.experiment.pipeline.base import EEGPipeline
from lib.transform._augment import SpectrogramAugmenter
from lib.transform._spectrogram import STFTComputer
from lib.transform._stack import SpectrogramStacker
from lib.gen.model import GanTrainConfig, GenTrainConfig
from lib.gen.trainer.gan import GanTrainer, Trainer
from lib.gen.trainer._initialize import get_trainer
from lib.experiment.run.base import RunConfig, Run
from lib.decorator import assert_called_before
from lib.experiment.pipeline.models import PipelineType
from typing import Dict, List
from lib.dataset.torch.transform import get_transforms
import torchvision
from lib.dataset.utils import get_gen_real_test_indexes
from torch import as_tensor
from lib.logging import get_logger

logger = get_logger()


class EEGenPipeline(EEGPipeline):
    gen_config: GenTrainConfig | GanTrainConfig
    trainer: Trainer | GanTrainer

    transforms_config: List[Dict] | None = None
    post_transforms_config: List[Dict] | None = None

    def __init__(self, name: str = "EEGenPipeline"):
        super().__init__(name, PipelineType.GEN)
        self.spectrograms = None   

    def set_gen_config(self, config: GenTrainConfig | GanTrainConfig) -> 'EEGenPipeline':
        self.gen_config = config
        return self

    def set_transforms_config(self, config: List[Dict]) -> 'EEGenPipeline':
        self.transforms_config = config
        return self

    def set_post_transforms_config(self, config: List[Dict]) -> 'EEGenPipeline':
        self.post_transforms_config = config
        return self

    def add_steps(self, steps: List[Dict]) -> 'EEGenPipeline':
        self.steps = steps
        return self
        
    def _initialize_trainer(self, run_config: RunConfig):
        self._update_dynamic_training_parameters(run_config)
        
        if not hasattr(self, "spectrograms") or self.spectrograms is None:
            raise ValueError("Spectrograms must be computed and stacked before initializing the trainer.")
        
        stacked_spectrograms, stacked_labels = self.spectrogram_stack()
        self.trainer = get_trainer(
            self.gen_config, 
            self.dataset, 
            self.visualizers, 
            self.evaluators, 
            self._call_hook
        )

        self.trainer.stft_computer = self.stft_computer
        self.trainer.inputs = (stacked_spectrograms, stacked_labels)  # Pass inputs to trainer

    def _update_dynamic_training_parameters(self, run_config: RunConfig):
        if run_config.pipeline_params is not None and len(run_config.pipeline_params) > 0:
            self.gen_config.update(run_config.pipeline_params)

    @assert_called_before("set_dataset")
    @assert_called_before("set_gen_config")
    def build(self):
        return self

    def load_data(self, config: RunConfig):
        self.dataset.load_by_config(config)
        self.dataset.set_transforms_by_config(self.transforms_config)
        self.dataset.set_post_transforms_by_config(self.post_transforms_config)
        config.subset_indexes = get_gen_real_test_indexes(self.dataset, config, size=16)
    
    def preprocess(self):
        self.dataset.epochs = self.dataset.epochs
        logger.info(f"Preprocessed data shape: {self.dataset.epochs.get_data(copy=True).shape}")

        t_min, t_max = self.dataset.t_min, self.dataset.t_max
        sfreq = self.dataset.raw.info['sfreq']
        
        self.gen_config.set_epochs_info(self.dataset.epochs.info)

    def spectrogram_compute(self):
        spectrogram_computer_config = next(
            (step for step in self.steps if step['name'] == 'spectrogram_computer'), None
        )
        if spectrogram_computer_config is None:
            raise ValueError("STFTComputer configuration is missing in pipeline steps.")

        sfreq = spectrogram_computer_config['kwargs']['sfreq']
        params = spectrogram_computer_config['kwargs']['params']

        spectrogram_computer = STFTComputer(sfreq=sfreq, params=params)
        
        data = self.dataset.data
        self.spectrograms = spectrogram_computer.transform(self.dataset.data)

        self.stft_computer = spectrogram_computer
    
    def spectrogram_normalize(self, feature_range=(-1.0, 1.0)):
        if self.spectrograms is None:
            raise ValueError("Spectrograms must be computed before applying Min–Max scaling.")

        data = self.spectrograms
        data_min = data.min()
        data_max = data.max()

        if data_min == data_max:
            raise ValueError("All spectrogram values are identical; cannot min–max scale.")

        # Store in the STFTComputer object
        self.stft_computer.spectrogram_min = data_min
        self.stft_computer.spectrogram_max = data_max
        self.stft_computer.feature_range   = feature_range

        min_range, max_range = feature_range
        data_scaled = (data - data_min) / (data_max - data_min)  # => [0..1]
        data_scaled = data_scaled * (max_range - min_range) + min_range  # => [feature_range]

        self.spectrograms = data_scaled
        logger.info(f"Spectrograms min-max scaled.")

        
    def spectrogram_augment(self):
        if self.spectrograms is None:
            raise ValueError("Spectrograms must be computed before augmentation.")

        augmenter_config = next(
            (step for step in self.steps if step['name'] == 'spectrogram_augmenter'), None
        )
        if augmenter_config is None:
            raise ValueError("SpectrogramAugmenter configuration is missing.")

        augmenter = SpectrogramAugmenter(config=augmenter_config['kwargs']['config'])
        self.spectrograms = augmenter.augment(self.spectrograms)

        self.saved_augmented_spectrograms = self.spectrograms

    def spectrogram_stack(self):
        if self.spectrograms is None:
            raise ValueError("Spectrograms must be computed and augmented before stacking.")

        stacker_config = next(
            (step for step in self.steps if step['name'] == 'spectrogram_stacker'), None
        )
        if stacker_config is None:
            raise ValueError("SpectrogramStacker configuration is missing.")

        labels = self.dataset.labels
        stacker = SpectrogramStacker(labels=labels)
        stacked_spectrograms, labels = stacker.stack(self.spectrograms)
        
        return stacked_spectrograms, labels

    @assert_called_before("build")
    def run(self, run_config: RunConfig) -> Run:
        self._call_hook("before_run", runner=self._runner, run_config=run_config)
        logger.info(f"Starting pipeline run for config: {run_config.name}")
        
        # Create a new run object
        run: Run = Run.from_run_config(run_config)
        run.start()

        logger.info("Loading data.")
        self.load_data(run_config)
        logger.info("Data loaded successfully.")

        logger.info("Starting preprocessing.")
        self.preprocess()
        logger.info("Preprocessing completed.")
        
        logger.info("Computing spectrograms.")
        self.spectrogram_compute()
        
        logger.info("Augmenting spectrograms.")
        self.spectrogram_augment()

        self.spectrogram_normalize()
        
        logger.info("Stacking spectrograms.")
        self.spectrogram_stack()
        
        logger.info("Initializing trainer.")
        self._initialize_trainer(run_config)
        logger.info("Trainer initialized successfully.")
  
        self._call_hook("before_train_run", runner=self._runner, run=run)
        
        logger.info("Starting training process.")
        run, result = self.trainer.run(run, self.gen_config)
        logger.info("Training completed.")

        self._call_hook("after_train_run", runner=self._runner, run=run, result=result)
        run.finish()
        run = self._save_run(run, result)
        self._call_hook("after_run", runner=self._runner, run=run)

        return run

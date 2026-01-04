from lib.transform._inverse import TimeSignalReconstructor
from lib.gen.model import GenTrainConfig
from lib.logging import get_logger
from lib.gen import Training
from lib.experiment.run.base import Run
from lib.experiment.evaluation.run import RunEvaluator, RunVisualizer
from lib.dataset import DatasetType
from typing import List, Callable
from lib.utils import empty_func, format_seconds
from lib.dataset import TorchBaseDataset
from lib.utils import format_seconds
from lib.gen.trainer.utils import get_eta
from lib.experiment.evaluation.result import EvaluatedResult
from lib.gen.trainer.utils import get_transformed_epochs_per_class, get_labels, get_events
from lib.gen.models.base import NNBase
import yaml
import torch
import numpy as np
from torch import Tensor
from mne import EpochsArray
from datetime import datetime
from lib.logging import get_logger


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class Trainer:
    logger = get_logger()

    dataset: TorchBaseDataset

    visualizers: List[RunVisualizer] = []
    evaluators: List[RunEvaluator] = []

    call_hook: Callable = empty_func

    def initialize(self):
        pass

    def set_visualizers(self, visualizers: List[RunVisualizer]):
        self.visualizers = visualizers
        return self

    def set_evaluators(self, evaluators: List[RunEvaluator]):
        self.evaluators = evaluators
        return self

    def train(self, config: GenTrainConfig, x: Tensor, labels: Tensor, epoch: int, batch_idx: int):
        pass

    def log_epoch(self, training: Training, config: GenTrainConfig):
        eta = get_eta(training, config)
        message = (f"\n\nSummary Epoch {training.result.epoch + 1}/{training.config.trainer.num_epochs},\n" +
                   f"Duration: {format_seconds(training.result.duration)}\n" +
                   f"ETA: {format_seconds(eta)}\n")
        for key, value in training.result.losses.items():
            message += f"{key}: {value:.4f}\n"
        self.logger.info(message)
        self.call_hook('after_run_epoch', result=EvaluatedResult(metrics=training.result.losses))
        

    def evaluate(self, training: Training, run: Run, model: NNBase) -> EvaluatedResult:
        self.logger.info(f"Start Evaluating Result of {training.config.model.name} for Run {run.name}")
        torch.cuda.empty_cache()
        model.eval()  # Put model in eval mode

        config = training.config
        dataset_type = config.trainer.gen_dataset_type
        batch_size_to_generate = config.trainer.test_sample_size
        
        with torch.no_grad():
            # --- Load test subset ---
            if run.config.test_subset is not None:
                test_run_config = run.config.get_test_config()

                self.dataset.reset(unlock=True)
                self.dataset.load_by_config(test_run_config)
                self.logger.info(f"Dataset loaded for evaluation with subset {run.config.test_subset}")

            apply_post_transforms: bool = (dataset_type != DatasetType.TORCH_STACKED)

            if run.config.test_subset is None and run.config.subset_indexes is not None:
                real = get_transformed_epochs_per_class(
                    self.dataset,
                    indexes=run.config.subset_indexes,
                    apply_post_transforms=apply_post_transforms
                )
            else:
                real = get_transformed_epochs_per_class(
                    self.dataset,
                    num=[('1', config.trainer.test_sample_size),
                        ('0', config.trainer.test_sample_size)],
                    apply_post_transforms=apply_post_transforms
                )

            labels = get_labels(real, dataset_type=dataset_type, n_stacks=config.trainer.n_stacks).to(config.device)
            n_real_epochs = real.get_data().shape[0]
            batch_size_to_generate = n_real_epochs
            
            # --- Generate synthetic spectrograms from the generator ---
            z = torch.randn((batch_size_to_generate, config.generator.z_dim, 1, 1), device=config.device)
            synthetic_labels = torch.randint(0, config.n_classes, (batch_size_to_generate,), device=config.device)

            fake_log_power = model(z, synthetic_labels)  
            fake_log_power_np = fake_log_power.detach().cpu().numpy()

            # --- iSTFT ---
            reconstructor = TimeSignalReconstructor(
                fs=128,
                nperseg=64,
                noverlap=48,
                n_fft=256,
                target_signal_length=self.stft_computer.original_signal_length,
                original_data_min=self.stft_computer.spectrogram_min,
                original_data_max=self.stft_computer.spectrogram_max,
                feature_range=self.stft_computer.feature_range
            )
            # Ensure phase_subset has at least n_real_epochs samples
            if self.stft_computer.phase_data.shape[0] < n_real_epochs:
                raise ValueError(
                    f"Not enough phase data to match the required number of synthetic samples: "
                    f"required={n_real_epochs}, available={self.stft_computer.phase_data.shape[0]}"
                )
            phase_subset = self.stft_computer.phase_data[:batch_size_to_generate]

            time_signals = reconstructor.transform(
                fake_log_power_np,   
                phase_subset       
            )
            self.logger.info(f"Reconstructed time-domain signals: shape={time_signals.shape}")

            # --- Ensure to match real data epochs count ---
            if time_signals.shape[0] != n_real_epochs:
                self.logger.info(
                    f"Adjusting synthetic signals from shape={time_signals.shape[0]} to match real={n_real_epochs}"
                )
                time_signals = time_signals[:n_real_epochs]

            sample_start_datetime = datetime.now()
            fake_torch = torch.from_numpy(time_signals).float() 
            sample_end_datetime = datetime.now()
            sample_duration = (sample_end_datetime - sample_start_datetime).total_seconds()
            self.logger.info(f"Finished iSTFT for {fake_torch.shape[0]} samples. "
                            f"Duration: {format_seconds(sample_duration, include_milliseconds=True)}")

            # --- Build events & wrap in MNE for final comparison ---
            events = get_events(real, labels, fake_torch, config, dataset_type)
            fake_epochs = EpochsArray(fake_torch.numpy(), real.info, tmin=real.tmin, events=events)

            assert fake_epochs.events.shape[0] == real.events.shape[0], \
                "Mismatch in number of epochs between real & synthetic."
            assert np.all(real.events[:, 2] == fake_epochs.events[:, 2]), \
                "Label mismatch between real and synthetic data."

            # --- Evaluate real vs. synthetic time-domain data ---
            from lib.experiment.evaluation.result import GenResult
            from lib.experiment.evaluation.result.utils import evaluate_result

            result = GenResult(real=real, synthetic=fake_epochs, labels=events[:, 2], module=model)
            evaluated_result = evaluate_result(result, self.evaluators, self.visualizers, gen_config=config)

            self.logger.info(f"Finished evaluation of {training.config.model.name} for Run {run.name}")
            return evaluated_result


    def run(self, run: Run, config: GenTrainConfig) -> (Run, EvaluatedResult): # type: ignore
        pass


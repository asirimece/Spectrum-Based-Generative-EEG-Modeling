from lib.experiment.evaluation.result._metric import SpectrogramSimilarityEvaluator
from lib.gen.trainer.base import Trainer
from lib.gen import GanTrainConfig, GanTraining
from lib.gen.models.base import GAN
from lib.dataset import TorchBaseDataset
import numpy as np
import torch
from torch import Tensor
from lib.experiment.run.base import Run
from lib.logging import get_logger
from tqdm import tqdm
from lib.experiment.evaluation.run import RunEvaluator, RunVisualizer
from lib.experiment.evaluation.result import EvaluatedResult
from typing import Callable
import wandb
from tqdm import tqdm
from lib.utils import empty_func, format_seconds
from torchsummary import summary

logger = get_logger()


class GanTrainer(Trainer):
    gan: GAN
    dataset: TorchBaseDataset

    def __init__(self,
                 gan: GAN,
                 dataset: TorchBaseDataset,
                 visualizers: list[RunVisualizer] | None = None,
                 evaluators: list[RunEvaluator] | None = None,
                 call_hook: Callable = empty_func,
                 inputs: tuple = None,
                 stft_computer=None
                 ):
        self.gan = gan
        self.dataset = dataset
        self.visualizers = visualizers or []
        self.evaluators = evaluators or []
        self.inputs = inputs
        self.stft_computer = stft_computer
        self.initialize()
        self.call_hook = call_hook

    def initialize(self):
        self.gan.initialize()
        if wandb.run is not None:
            wandb.watch(self.gan.generator, log="all", log_freq=5)
            wandb.watch(self.gan.critic, log="all", log_freq=5)

    def dispose(self):
        if wandb.run is not None:
            wandb.unwatch(self.gan.generator)
            wandb.unwatch(self.gan.critic)
    

    def train_critic(self, training: GanTraining, x: Tensor, labels: Tensor) -> (GanTraining, Tensor): # type: ignore
        self.gan.critic.optim.zero_grad()
        batch_size = x.shape[0]  
        config = training.config
        z = torch.randn((batch_size, config.generator.z_dim, 1, 1)).to(config.device) 
        fake = self.gan.generator(z, labels)
        loss = self.gan.critic.loss(x, fake, labels)
        loss.backward(retain_graph=True)
        self.gan.critic.optim.step()
            
        return training, fake

    def train_generator(self, training: GanTraining, fake: Tensor, labels: Tensor) -> GanTraining:
        # Zero the gradient
        self.gan.generator.optim.zero_grad()

        output = self.gan.critic(fake, labels)
        if isinstance(output, tuple):
            output = output[0]
        output = output.view(-1)

        loss = self.gan.generator.loss(output)
        
        loss.backward()
        self.gan.generator.optim.step()
        
        return training

    def train(self, training: GanTraining, x: Tensor, labels: Tensor, epoch: int, batch_idx: int) -> GanTraining:
        config = training.config

        training, fake = self.train_critic(training, x, labels)
        
        # Train generator if the critic iterations threshold is met
        if (batch_idx + 1) % config.critic.iterations == 0:
            
            training = self.train_generator(training, fake, labels)

        training.result.loss = self.gan.get_averaged_loss()
        training.result.losses = self.gan.get_aggregated_losses()

        return training, fake


    def run(self, run: Run, config: GanTrainConfig) -> (Run, EvaluatedResult): # type: ignore
        if not self.inputs:
            raise ValueError("Inputs (stacked_spectrograms, labels) must be provided to the trainer.")

        stacked_spectrograms, labels_all = self.inputs
                
        self.spectrograms = stacked_spectrograms
        
        self.logger.info(f"Trainer: Starting {config.model.name} training.")
        
        x = torch.tensor(stacked_spectrograms, dtype=torch.float32, device=config.device)
        labels = torch.tensor(labels_all, dtype=torch.long, device=config.device)

        test_gan_shapes(self.gan, config)

        training = GanTraining(config)
        self.gan.train_mode()
        training.start()

        # Decide how many mini-batches we accumulate before stepping
        accum_steps = getattr(config.trainer, "accum_steps", 1)
        physical_batch_size = config.trainer.batch_size // accum_steps

        accum_counter_critic = 0
        accum_counter_generator = 0

        for epoch in range(config.trainer.num_epochs):
            logger.info(f"Starting Epoch {epoch+1}/{config.trainer.num_epochs} ")
            self.gan.reset_loss_history()
            training.start_epoch(epoch=epoch)

            # Slice the data into sub-batches
            num_samples = x.shape[0] 
            num_batches = (num_samples + physical_batch_size - 1) // physical_batch_size

            with tqdm(range(num_batches), unit='batch') as t:
                for batch_idx in t:
                    start_idx = batch_idx * physical_batch_size
                    end_idx = min(start_idx + physical_batch_size, num_samples)

                    # Extract the sub-batch
                    x_batch = x[start_idx:end_idx]           
                    labels_batch = labels[start_idx:end_idx] 

                    # --- Critic forward/backward ---
                    if accum_counter_critic == 0:
                        # Only zero gradients at start of a new accumulation cycle
                        self.gan.critic.optim.zero_grad(set_to_none=True)

                    batch_size_now = x_batch.shape[0]
                    z = torch.randn((batch_size_now, config.generator.z_dim, 1, 1), device=config.device)

                    # Forward pass
                    fake = self.gan.generator(z, labels_batch)
                    loss_critic = self.gan.critic.loss(x_batch, fake, labels_batch)
                    loss_critic.backward(retain_graph=True)
                    accum_counter_critic += 1
                    
                    if accum_counter_critic == accum_steps:
                        self.gan.critic.optim.step()
                        accum_counter_critic = 0

                    # --- Train the generator ---
                    if (batch_idx + 1) % config.critic.iterations == 0:
                        if accum_counter_generator == 0:
                            self.gan.generator.optim.zero_grad(set_to_none=True)

                        # Critic forward on fake
                        output = self.gan.critic(fake, labels_batch)
                        if isinstance(output, tuple):
                            output = output[0]
                        output = output.view(-1)

                        loss_generator = self.gan.generator.loss(output)
                        loss_generator.backward()
                        accum_counter_generator += 1
                        
                        if accum_counter_generator == accum_steps:
                            self.gan.generator.optim.step()
                            accum_counter_generator = 0
                            
                    t.set_postfix({
                        "C_loss": f"{loss_critic.item():.4f}",
                        "G_loss": (f"{loss_generator.item():.4f}" 
                                if (batch_idx + 1) % config.critic.iterations == 0 else "-")
                    })

            # End of epoch: if leftover sub-batches haven't triggered an optimizer step
            if accum_counter_critic > 0:
                self.gan.critic.optim.step()
                accum_counter_critic = 0
            if accum_counter_generator > 0:
                self.gan.generator.optim.step()
                accum_counter_generator = 0

            training.result.losses = self.gan.get_aggregated_losses()
            training.finish_epoch()
            self.log_epoch(training, config)

        training.finish()
        self.logger.info(
            f"Trainer: Finished {config.model.name} Training, "
            f"took {format_seconds(training.result.duration)}"
        )
        
        # Evaluate similarity metrics after training
        similarity_evaluator = SpectrogramSimilarityEvaluator(
            config=config, dataset=self.dataset, gan=self.gan
        )
        similarity_metrics = similarity_evaluator.evaluate(run)
        self.logger.info(f"Spectrogram similarity metrics: {similarity_metrics}")

        result: EvaluatedResult = self.evaluate(training, run, self.gan.generator)
        self.dispose()
        
        self.logger.info(f"Trainer: Finished {config.model.name} Training")
        return run, result


def test_gan_shapes(gan: GAN, config: GanTrainConfig):
    logging = get_logger()

    N = config.trainer.batch_size                   
    in_channels, H, W = config.model.model_shape    
    z_dim = config.generator.z_dim                  

    x = torch.randn((N, in_channels, H, W)).to(config.device)
    labels = torch.randint(0, config.n_classes, (N,)).to(config.device)

    # --- Forward pass through critic ---
    critic_output = gan.critic(x, labels)
    if not isinstance(critic_output, tuple):
        raise ValueError("Critic output must be a tuple (validity, label_scores).")

    critic_validity, pred_label_scores = critic_output

    expected_validity_shape = (N, 1, 1, 1)
    assert critic_validity.shape == expected_validity_shape, (
        f"Unexpected critic validity shape: {critic_validity.shape} != {expected_validity_shape}"
    )

    expected_label_shape = (N, config.n_classes)
    assert pred_label_scores.shape == expected_label_shape, (
        f"Unexpected label scores shape: {pred_label_scores.shape} != {expected_label_shape}"
    )

    z = torch.randn((N, z_dim, 1, 1)).to(config.device)

    # --- Forward pass through Generator ---
    gen_output = gan.generator(z, labels)
    gen_output_shape = gen_output.shape

    expected_gen_shape = (N, in_channels, H, W)
    assert gen_output_shape == expected_gen_shape, (
        f"Generator output shape mismatch: {gen_output_shape} != {expected_gen_shape}"
    )

    logging.info("Dimension check passed!")
    logging.info("-------------------------------")
    logging.info("Model Details <Critic>")
    summary(gan.critic, input_size=(in_channels, H, W))
    logging.info("-------------------------------")
    logging.info("Model Details <Generator>")
    summary(gan.generator, input_size=(z_dim, 1, 1))


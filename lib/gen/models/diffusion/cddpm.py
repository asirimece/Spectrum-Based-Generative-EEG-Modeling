from lib.gen.models.diffusion.base import Diffusion
import torch
from torch import nn, optim
import copy
from lib.logging import get_logger
from logging import Logger
from lib.gen.models.diffusion.modules import CUNet, EMA
from tqdm import tqdm
import numpy as np
import os
from lib.gen import DiffusionTrainConfig
from lib.gen.model import Training, GenTrainConfig
from lib.dataset import TorchBaseDataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch import Size
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler
from torchsummary import summary
from typing import Callable, Any
from lib.dataset.torch.transform import get_transforms
import math


class CDDPM(Diffusion):
    logger: Logger
    scheduler: LRScheduler
    ema: EMA
    scaler: GradScaler

    def __init__(self,
                 input_shape: Size = (1, 62, 301),
                 model_shape: Size = (1, 64, 301),
                 noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 num_classes=2,
                 c_in=1,
                 c_out=1,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 device=torch.device("cuda"),
                 noise_schedule_name: str = "linear",
                 seed: int = 1,
                 **kwargs):
        model = CUNet(c_in, c_out, num_classes=num_classes, **kwargs).to(device)

        super().__init__(
            input_shape=input_shape,
            model_shape=model_shape,
            model=model,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors,
            device=device,
            seed=seed
        )

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule_name = noise_schedule_name

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes
        self.logger = get_logger()

    @staticmethod
    def from_configs(config: DiffusionTrainConfig) -> 'CDDPM':
        diffusion = CDDPM(
            input_shape=config.input_shape,
            model_shape=config.model.model_shape,
            noise_steps=config.diffusion.noise_steps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            noise_schedule_name=config.diffusion.noise_schedule_name,
            num_classes=config.diffusion.num_classes,
            c_in=config.diffusion.c_in,
            c_out=config.diffusion.c_out,
            input_preprocessors=get_transforms(config.model.input_preprocessors),
            output_preprocessors=get_transforms(config.model.output_preprocessors),
            device=config.device,
            seed=config.seed
        )
        return diffusion

    def initialize(self, config: DiffusionTrainConfig, dataset: TorchBaseDataset):
        self.dataset = dataset
        self.dataset.lock_and_retrieve()
        train_dataset, val_dataset = self.split_dataset(lengths=[0.8, 0.2])
        g = torch.Generator()
        g.manual_seed(config.seed)
        pin_memory = True if config.device == torch.device('cuda') else False
        self.train_dataloader = DataLoader(train_dataset, batch_size=config.trainer.batch_size, shuffle=True,
                                           generator=g, pin_memory=pin_memory)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.trainer.batch_size, shuffle=False, generator=g,
                                         pin_memory=pin_memory)
        self.optim = optim.AdamW(self.model.parameters(), lr=config.diffusion.optim.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=config.diffusion.optim.lr,
                                                       steps_per_epoch=len(self.train_dataloader),
                                                       epochs=config.trainer.num_epochs)
        self.criterion = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def prepare_noise_schedule(self):
        if self.noise_schedule_name == "linear":
            scale = 1000 / self.noise_steps
            beta_start = scale * self.beta_start
            beta_end = scale * self.beta_end
            return torch.linspace(beta_start, beta_end, self.noise_steps, dtype=torch.float64)
        elif self.noise_schedule_name == "cosine":
            return betas_for_alpha_bar(
                self.noise_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule_name}")

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def post_process(self, x: Tensor) -> Tensor:
        if len(self.output_preprocessors) > 0:
            for preprocessor in self.output_preprocessors:
                x = preprocessor(x)
        return x

    @torch.inference_mode()
    def sample(self, labels: Tensor, cfg_scale: int = 3, use_ema: bool = True) -> Tensor:
        model = self.ema_model if use_ema else self.model
        batch_size = 8
        n_labels = len(labels)
        self.logger.info(f"Sampling {n_labels} new signals...")
        self.eval_mode()
        results = []
        with torch.inference_mode():
            for batch_idx in range(0, n_labels, batch_size):
                labels_batch = labels[
                               batch_idx:batch_idx + batch_size] if batch_idx + batch_size < n_labels else labels[
                                                                                                           batch_idx:]
                batch_size = len(labels_batch)
                x = torch.randn((batch_size, self.c_in, self.model_eeg_channels, self.model_n_times)).to(self.device)
                for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):

                    labels_batch = labels_batch.to(self.device)
                    t = (torch.ones(batch_size) * i).long().to(self.device)
                    predicted_noise = model(x, t, labels_batch)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise
                x = x.clamp(-1, 1)
                x = self.post_process(x)
                results.append(x)

        return torch.concat(results, dim=0)

    def train_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def test_shapes(self):
        with torch.no_grad():
            x = torch.randn(
                (self.train_dataloader.batch_size, self.c_in, self.model_eeg_channels, self.model_n_times)).to(
                self.device)
            t = self.sample_timesteps(x.shape[0]).to(self.device)
            x_t, noise = self.noise_images(x, t)
            labels = torch.ones(x.shape[0], dtype=torch.int).to(self.device)
            predicted_noise = self.model(x_t, t, labels)
            assert predicted_noise.shape == noise.shape, (
                f"Shape Test Failed: Predicted Noise Shape: {predicted_noise.shape},"
                f"Noise Shape: {noise.shape}")
            self.logger.info("Shape Test Passed. Yuppie!")
            self.logger.info("--------------------------------")
            self.logger.info("Model Details:")
            summary(self.model, (x_t, t, labels))

    def loss(self, noise: Tensor, predicted_noise: Tensor, train: bool = True):
        key: str = 'train_loss' if train else 'val_loss'
        loss = self.criterion(noise, predicted_noise)
        self.add_loss_history(key, loss.item())
        return loss

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def one_epoch(self, config: GenTrainConfig, training: Training, train: bool = True):
        self.train_mode() if train else self.eval_mode()
        dataloader = self.train_dataloader if train else self.val_dataloader

        with tqdm(dataloader, unit='batch') as t:
            for batch_idx, (x, labels) in enumerate(t):
                # with torch.autocast(self.device) and (torch.inference_mode() if not train else torch.enable_grad()):
                with torch.enable_grad() if train else torch.inference_mode():
                    if train:
                        self.optim.zero_grad()
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    t = self.sample_timesteps(x.shape[0]).to(self.device)
                    x_t, noise = self.noise_images(x, t)
                    if np.random.random() < 0.1:
                        labels = None
                    predicted_noise = self.model(x_t, t, labels)
                    loss = self.loss(noise, predicted_noise, train)
                    if train:
                        self.train_step(loss)
                t.comment = f"MSE={loss.item():2.3f}"
        return training

    def run_training(self, config: GenTrainConfig, training: Training) -> Training:
        return self.one_epoch(config, training)

    def run_validation(self, config: GenTrainConfig, training: Training) -> Training:
        return self.one_epoch(config, training, train=False)

    def forward(self, labels: Tensor) -> Tensor:
        return self.sample(labels)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

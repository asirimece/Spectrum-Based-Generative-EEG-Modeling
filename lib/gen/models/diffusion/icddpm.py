from lib.gen.models.diffusion.base import Diffusion
import torch
from torch import nn, optim
import copy
from lib.logging import get_logger
from logging import Logger
from .unet import UNet
from tqdm import tqdm
import numpy as np
from .utils import extract
from lib.gen import DiffusionTrainConfig
from lib.gen.model import Training, GenTrainConfig
from lib.dataset import TorchBaseDataset
from torch.utils.data import DataLoader
from torch import Tensor
from torch import Size
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler
from torchsummary import summary
from typing import Callable, Any, Union
from lib.dataset.torch.transform import get_transforms
import math
from torch.cuda import Device
from .models import ModelVarType, ModelMeanType, LossType
from .resample import UniformSampler, get_sampler
import functools
from .nn import mean_flat, update_ema
from .losses import normal_kl, discretized_gaussian_log_likelihood
from einops import rearrange
from scipy.signal import welch

INITIAL_LOG_LOSS_SCALE = 20.0


class ICDDPM(Diffusion):
    logger: Logger
    scheduler: LRScheduler
    scaler: GradScaler

    def __init__(self,
                 input_shape: Size = (1, 62, 64),
                 model_shape: Size = (1, 64, 64),
                 noise_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 noise_schedule_name: str = "linear",
                 num_classes: int = 2,
                 c_in: int = 1,
                 c_out: int = 1,
                 input_preprocessors: list[Callable[..., Any]] | None = None,
                 output_preprocessors: list[Callable[..., Any]] | None = None,
                 device: Device = torch.device("cuda"),
                 offset_noise_strength: int = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
                 model_var_type: ModelVarType = ModelVarType.LEARNED,
                 model_mean_type: ModelMeanType = ModelMeanType.NOISE,
                 loss_type: LossType = LossType.RESCALED_MSE,
                 rescale_timesteps: bool = False,
                 ema_rate: str | float = 0.9999,  # 0.995,
                 sampler: dict[str, Any] | None = None,
                 psd_method: str = 'welch',
                 psd_weight: float | None = None,
                 upsampling_factor: float | None = None,
                 sfreq: int = 64,
                 seed: int = 1,
                 **kwargs):

        super().__init__(
            input_shape=input_shape,
            model_shape=model_shape,
            input_preprocessors=input_preprocessors,
            output_preprocessors=output_preprocessors,
            device=device,
            seed=seed
        )

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_schedule_name = noise_schedule_name
        self.model_var_type = ModelVarType.check_instance(model_var_type)
        self.model_mean_type = ModelMeanType.check_instance(model_mean_type)
        self.loss_type = LossType.check_instance(loss_type)
        self.rescale_timesteps = rescale_timesteps
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.upsampling_factor = upsampling_factor

        self.device = device
        self.c_in = c_in
        self.c_out = c_out
        self.num_classes = num_classes

        self.model = self.create_model(self.model_var_type).to(device)

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.ema_rate = self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.ema_params = [copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))]
        self.psd_weight = psd_weight
        self.psd_method = psd_method

        self.sfreq = sfreq

        self.logger = get_logger()

        betas = self.prepare_noise_schedule().to(device)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        assert len(betas) == noise_steps, f"Expected {noise_steps} betas, got {len(betas)}"

        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat((torch.tensor([1], dtype=torch.float32, device=device), alphas_bar[:-1]), dim=0)

        posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

        self.register("betas", betas, device=device)
        self.register("alphas", alphas, device=device)
        self.register("alphas_bar", alphas_bar, device=device)
        self.register("alphas_bar_prev", alphas_bar_prev, device=device)
        self.register("posterior_variance", posterior_variance, device=device)
        self.register('posterior_log_variance_clipped', torch.log(
            torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:]))), device=device)

        self.register("sqrt_alphas_bar", torch.sqrt(alphas_bar), device=device)
        self.register("sqrt_one_minus_alphas_bar", torch.sqrt(1 - alphas_bar), device=device)
        self.register("log_one_minus_alphas_bar", torch.log(1 - alphas_bar), device=device)
        self.register("sqrt_recip_alphas_bar", torch.sqrt(1 / alphas_bar), device=device)
        self.register('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1), device=device)

        posterior_mean_coef1 = (betas * torch.sqrt(alphas_bar_prev)) / (1 - alphas_bar)
        posterior_mean_coef2 = (torch.sqrt(alphas_bar) * (1 - alphas_bar_prev)) / (1 - alphas_bar)
        self.register("posterior_mean_coef1", posterior_mean_coef1, device=device)
        self.register("posterior_mean_coef2", posterior_mean_coef2, device=device)

        self.offset_noise_strength = offset_noise_strength
        self.schedule_sampler = UniformSampler(noise_steps) if sampler is None \
            else get_sampler(model_mean_type=self.model_mean_type, alphas_bar=self.alphas_bar, **sampler)

    @staticmethod
    def from_configs(config: DiffusionTrainConfig) -> 'ICDDPM':
        diffusion = ICDDPM(
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
            seed=config.seed,
            **config.diffusion.kwargs
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
        self.optim = optim.AdamW(self.model.parameters(), lr=config.diffusion.optim.lr, weight_decay=0.0, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=config.diffusion.optim.lr,
                                                       steps_per_epoch=len(self.train_dataloader),
                                                       epochs=config.trainer.num_epochs)
        self.scaler = torch.cuda.amp.GradScaler()

    def create_model(self, model_var_type: ModelVarType = ModelVarType.LEARNED) -> nn.Module:
        assert self.model_eeg_channels == self.model_n_times, "Currently, UNet only supports square input signals."
        input_size = self.model_n_times
        in_channels = self.c_in
        attention_resolutions = "16,8"

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(input_size // int(res))

        if input_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif input_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif input_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported input size: {input_size}")

        return UNet(
            out_shape=self.model_shape,
            in_channels=in_channels,
            out_channels=in_channels * 2 if model_var_type.learned_sigma() else in_channels,
            model_channels=128,
            num_classes=self.num_classes,
            num_res_blocks=2,
            attention_resolutions=tuple(attention_ds),
            dropout=0.0,
            channel_mult=channel_mult,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            upsampling_factor=self.upsampling_factor
        )

    def register(self, name: str, tensor: Tensor, device: Device):
        tensor = tensor.type(torch.float32)
        tensor = tensor.to(device)
        self.register_buffer(name, tensor)

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

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_bar, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_bar, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_bar, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.generate_noise(x_start)

        x_t = (
                extract(self.sqrt_alphas_bar, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * noise
        ).type(torch.float32)
        return x_t, noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_x_start_from_x_prev(self, x_t, t, x_prev):
        assert x_t.shape == x_prev.shape
        return (  # (x_prev - coef2*x_t) / coef1
                extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * x_prev
                - extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def predict_x_start_from_noise(self, x_t, t, noise):
        """
        :param x_t:
        :param t:
        :param noise: equals to eps in the formula
        :return:
        """
        assert x_t.shape == noise.shape
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * noise
        )

    def predict_eps_from_x_start(self, x_t, t, pred_x_start):
        return (
                extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
                - pred_x_start
        ) / extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_bar, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_bar, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * v
        )

    def p_mean_variance(self, model, x, t, y: None | Tensor = None, clip: bool = True,
                        denoised_fn: Union[None, Callable] = None):

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self.scale_timesteps(t), y)

        if self.model_var_type == ModelVarType.LEARNED:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)

            model_log_variance = model_var_values
            model_variance = torch.exp(model_log_variance)
        else:
            model_variance = extract(self.posterior_variance, t, x.shape)
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        if self.model_mean_type == ModelMeanType.X_PREV:
            pred_x_start = self.predict_x_start_from_x_prev(x, t, model_output)
            pred_x_start = process_input(pred_x_start, denoised_fn, clip)
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.X_START, ModelMeanType.NOISE, ModelMeanType.PRED_V]:
            if self.model_mean_type == ModelMeanType.X_START:
                pred_x_start = process_input(model_output, denoised_fn, clip)
            elif self.model_mean_type == ModelMeanType.PRED_V:
                v = model_output
                pred_x_start = self.predict_start_from_v(x_t=x, t=t, v=v)
                pred_x_start = process_input(pred_x_start, denoised_fn, clip)
            else:
                pred_x_start = self.predict_x_start_from_noise(x_t=x, t=t, noise=model_output)
                pred_x_start = process_input(pred_x_start, denoised_fn, clip)
            model_mean, _, _, = self.q_posterior_mean_variance(x_start=pred_x_start, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_x_start.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x_start": pred_x_start,
        }

    def generate_noise(self, x):
        noise = torch.randn_like(x, dtype=torch.float32)
        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x.shape[:2], device=self.device, dtype=torch.float32)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
        return noise

    def post_process(self, x: Tensor) -> Tensor:
        if len(self.output_preprocessors) > 0:
            for preprocessor in self.output_preprocessors:
                x = preprocessor(x)
        return x

    def p_sample(self, model, x, t, y, clip=True):
        out = self.p_mean_variance(model=model, x=x, t=t, y=y, clip=clip)
        noise = self.generate_noise(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_start": out["pred_x_start"]}

    def run_epoch(self, config: GenTrainConfig, training: Training, train: bool = True):
        self.train_mode() if train else self.eval_mode()
        dataloader = self.train_dataloader if train else self.val_dataloader

        with tqdm(dataloader, unit='batch') as batches:
            for batch_idx, (x, labels) in enumerate(batches):
                with torch.enable_grad() if train else torch.inference_mode():
                    if train:
                        self.optim.zero_grad()

                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    t, weights = self.schedule_sampler.sample(x.shape[0], device=self.device)

                    compute_losses = functools.partial(self.training_losses, self.model, x, t, labels,
                                                       None, self.psd_weight)

                    losses = compute_losses()
                    loss = (losses["loss"] * weights).mean()
                    if train:
                        self.train_step(loss)
                        self.add_loss_history("train_loss", loss.item())
                    else:
                        self.add_loss_history("val_loss", loss.item())
                batches.comment = f"MSE={loss.item():2.3f}"
        return training

    def train_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        self.scheduler.step()

    @torch.inference_mode()
    def sample(self, labels=None, noise=None, clip=True):
        model = self.model
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
                if noise is not None:
                    x = noise
                else:
                    x = torch.randn((batch_size, self.c_in, self.model_eeg_channels, self.model_n_times)).to(
                        self.device).type(torch.float32)
                for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
                    labels_batch = labels_batch.to(self.device)
                    t = (torch.ones(batch_size) * i).long().to(self.device)
                    with torch.no_grad():
                        out = self.p_sample(
                            model,
                            x,
                            t,
                            labels_batch,
                            clip=clip,
                        )
                        x = out['sample'].type(torch.float32)
                x = x.clamp(-1, 1)
                x = self.post_process(x)
                results.append(x)
        return torch.cat(results, dim=0)

    def run_training(self, config: GenTrainConfig, training: Training) -> Training:
        return self.run_epoch(config, training)

    def run_validation(self, config: GenTrainConfig, training: Training) -> Training:
        return self.run_epoch(config, training, train=False)

    def forward(self, labels: Tensor) -> Tensor:
        return self.sample(labels)

    def scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.noise_steps)
        return t

    def compute_periodogram_psd(self, x):
        """
        Compute the power spectral density (PSD) of a signal using FFT.
        Args:
            x (torch.Tensor): Input signal tensor of shape (batch_size, channels, signal_length).
            n_fft (int): Number of FFT points. Default is 256.
        Returns:
            psd (torch.Tensor): Power spectral density of the input signal.
        """
        x_fft = torch.fft.fft(x, n=self.model_n_times, dim=-1)
        psd = torch.abs(x_fft) ** 2
        return psd

    def compute_welchs_psd(self, x: Tensor, n_per_seg: int | None = None):
        """
        Compute the power spectral density (PSD) of a signal using Welch's method.
        :param x:
        :param n_per_seg:
        :return:
        """
        batch_size, _, n_channels, n_times = x.size()
        n_per_seg = self.sfreq if n_per_seg is None else n_per_seg

        psd = []
        for i in range(batch_size):
            for j in range(n_channels):
                f, Pxx = welch(x[i, 0, j].cpu().numpy(), fs=self.sfreq, nperseg=n_per_seg)
                psd.append(Pxx)
        psd = np.array(psd)
        psd = torch.tensor(psd, device=self.device).reshape(batch_size, n_channels, -1)

        return psd

    def compute_psd(self, x: Tensor):
        match self.psd_method:
            case 'welch':
                return self.compute_welchs_psd(x)
            case 'periodogram':
                return self.compute_periodogram_psd(x)
            case _:
                raise ValueError(f"Unknown PSD method: {self.psd_method}")


    def training_losses(self, model, x_start, t, labels, noise=None, psd_weight: float | None = None):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_t, noise = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip=False,
                labels=labels,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.noise_steps
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self.scale_timesteps(t), labels)

            if self.model_var_type in [
                ModelVarType.LEARNED
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = torch.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    labels=labels,
                    clip=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.noise_steps / 1000.0

            target = {
                ModelMeanType.X_PREV: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.X_START: x_start,
                ModelMeanType.NOISE: noise,
                ModelMeanType.PRED_V: self.predict_v(x_start, t, noise),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)

            if psd_weight is not None:
                with torch.no_grad():
                    out = self.p_sample(model, x_t, t, labels)
                    pred_x_start = out["pred_x_start"]
                real_psd = self.compute_psd(x_start)
                pred_psd = self.compute_psd(pred_x_start)
                terms["psd"] = mean_flat((real_psd - pred_psd) ** 2) * psd_weight

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            elif "psd" in terms:
                terms["loss"] = terms["mse"] + terms["psd"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip=True, labels=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_x_start': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip=clip, y=labels
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_x_start": out["pred_x_start"]}

    def test_shapes(self):
        with torch.no_grad():
            x = torch.randn(
                (self.train_dataloader.batch_size, self.c_in, self.model_eeg_channels, self.model_n_times)).to(
                self.device).type(torch.float32)
            t, weights = self.schedule_sampler.sample(x.shape[0], device=self.device)
            x_t, noise = self.q_sample(x, t)
            labels = torch.ones(x.shape[0], dtype=torch.int).to(self.device)
            predicted_noise = self.model(x_t, t, labels)
            target_shape = noise.repeat(1, 2, 1, 1).shape if self.model_var_type.learned_sigma() else noise.shape
            assert predicted_noise.shape == target_shape, (
                f"Shape Test Failed: Predicted Noise Shape: {predicted_noise.shape},"
                f"Noise Shape: {noise.shape}")
            self.logger.info("Shape Test Passed")
            summary(self.model, (x_t, t, labels))


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


def process_input(x: Tensor, denoised_fn: Union[Callable, None] = None, clip: bool = True):
    if denoised_fn is not None:
        x = denoised_fn(x)
    if clip:
        return x.clamp(-1, 1)
    return x

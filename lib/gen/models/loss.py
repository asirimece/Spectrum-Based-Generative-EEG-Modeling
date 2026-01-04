import torch
from torch import nn
from torch import Tensor
from torch.cuda import Device


def gradient_penalty(critic: nn.Module, real: Tensor, fake: Tensor, labels: Tensor, device: Device):
    batch_size, c, h, w = real.shape
    epsilon = torch.randn((batch_size, 1, 1, 1), device=device)

    interpolates = (epsilon * real + ((1 - epsilon) * fake)).requires_grad_(True)
    interpolate_scores = critic(interpolates, labels)
    if isinstance(interpolate_scores, tuple):
        interpolate_scores = interpolate_scores[0]

    grad_outputs = torch.ones_like(interpolate_scores, device=device, requires_grad=False)

    if isinstance(interpolate_scores, tuple):
        interpolate_scores = interpolate_scores[0]

    gradient = torch.autograd.grad(
        inputs=interpolates,
        outputs=interpolate_scores,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

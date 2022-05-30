
import torch
import torch.nn as nn

from typing import Tuple

from .base import Transform


class ActNorm(Transform):
    """
    Activation Norm as described by
    Kingma et al.
    Glow: Generative Flow with Invertible 1x1 Convolutions
    """
    def __init__(self):
        super().__init__()
        self.scale = None
        self.shift = None

    @property
    def initialized(self):
        return self.scale is None or self.shift is None

    def initialize(self, x: torch.Tensor) -> None:
        batch_size = x.shape[0]
        x = x.detach().reshape(batch_size, -1)
        std, mean = torch.std_mean(x, dim=0, unbiased=False)

        self.scale = nn.Parameter(1 / std)
        self.shift = nn.Parameter(-mean)

    def reset(self):
        self.scale = None
        self.shift = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self.initialize(x)

        z = self.scale * x + self.shift

        logabsdet = torch.log(torch.abs(self.scale))

        return z, logabsdet

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (z - self.shift) / self.scale

        logabsdet = -torch.log(torch.abs(self.scale))

        return x, logabsdet

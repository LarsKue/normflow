
import torch
import torch.nn as nn

from .base import Transform

import normflow.utils as utils


class ActNorm(Transform):
    """
    Activation Norm, similar to Batch Norm, normalizes the input with an affine scale and shift
    which are initialized upon first forward iteration (activation) and afterwards
    treated as regular trainable parameters

    This makes the normalization invertible and lowers activation noise
    for small per-PU (processing unit) batch sizes when compared to BatchNorm.

    Introduced by arXiv:1807.03039
    """
    def __init__(self, shape: tuple | torch.Size) -> None:
        super().__init__()
        # need to explicitly register these as parameters
        self.logscale = nn.Parameter(torch.zeros(shape))
        self.shift = nn.Parameter(torch.zeros(shape))
        # we want this bool in the state_dict so register it as a buffer
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @property
    def scale(self):
        return torch.exp(self.logscale)

    def initialize(self, x: torch.Tensor) -> None:
        mean = utils.mean_except(x, 1)
        std = utils.std_except(x, 1)

        self.logscale.data = -torch.log(std).reshape(self.logscale.data.shape)
        self.shift.data = -mean.reshape(self.shift.data.shape)
        self.initialized.data = torch.tensor(True, dtype=torch.bool)

    def reset(self):
        self.logscale.data = torch.zeros_like(self.logscale.data)
        self.shift.data = torch.zeros_like(self.shift.data)

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and not self.initialized:
            self.initialize(x)

        scale = self.scale.to(x.device)
        shift = self.shift.to(x.device)

        z = scale * x + shift
        logabsdet = utils.sum_except_batch(scale).to(x.device)

        return z, logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale.to(z.device)
        shift = self.shift.to(z.device)
        x = (z - shift) / scale
        logabsdet = -utils.sum_except_batch(self.logscale).to(z.device)

        return x, logabsdet

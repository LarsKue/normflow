import torch
import torch.nn.functional as F


import normflow.utils as utils

from .autoregressive import AutoRegressiveCoupling


class AffineCoupling(AutoRegressiveCoupling):
    """
    Coupling that applies a scale and shift to inputs
    The scale is constrained to positive values with a softplus
    The shift is unconstrained
    """

    def __init__(self, *args, dim: int = -1, **kwargs):
        """
        @param dim: The dimension along which parameters will be split
        """
        super().__init__(*args, **kwargs)
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int64))

    def _forward(self, params: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @param params: Tensor of shape (..., 2, ...) that specifies the scale and shift
        """
        unconstrained_scale, shift = torch.tensor_split(params, sections=2, dim=self.dim)

        scale = F.softplus(unconstrained_scale)

        logscale = torch.log(scale)

        z2 = scale * x2 + shift

        logabsdet = utils.sum_except_batch(logscale)

        return z2, logabsdet

    def _inverse(self, params: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @param params: Tensor of shape (..., 2) that specifies scale and shift
        """
        unconstrained_scale, shift = torch.tensor_split(params, sections=2, dim=self.dim)

        scale = F.softplus(unconstrained_scale)

        logscale = torch.log(scale)

        x2 = (z2 - shift) / scale

        logabsdet = -utils.sum_except_batch(logscale)

        return x2, logabsdet

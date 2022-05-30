import torch
import torch.nn.functional as F

from typing import Tuple

import normflow.utils as utils

from .autoregressive import AutoRegressiveCoupling


class AffineCoupling(AutoRegressiveCoupling):
    """
    Coupling that applies a scale and shift to inputs
    The scale is constrained to positive values with a softplus
    The shift is unconstrained
    """

    def _forward(self, params: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unconstrained_scale, shift = torch.tensor_split(params, 2, -1)

        scale = F.softplus(unconstrained_scale)

        logscale = torch.log(scale)

        z2 = scale * x2 + shift

        logabsdet = utils.sum_except_batch(logscale)

        return z2, logabsdet

    def _inverse(self, params: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unconstrained_scale, shift = torch.tensor_split(params, 2, -1)

        scale = F.softplus(unconstrained_scale)

        logscale = torch.log(scale)

        x2 = (z2 - shift) / scale

        logabsdet = -utils.sum_except_batch(logscale)

        return x2, logabsdet

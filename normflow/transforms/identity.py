
import torch

from typing import Optional

from .base import Transform


class IdentityTransform(Transform):
    """
    This transform does nothing.
    """
    def forward(self,
                x: torch.Tensor,
                *,
                condition: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z = x
        logabsdet = x.new_zeros(x.shape[0])

        return z, logabsdet, condition

    def inverse(self,
                z: torch.Tensor,
                *,
                condition: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = z
        logabsdet = z.new_zeros(z.shape[0])

        return x, logabsdet, condition

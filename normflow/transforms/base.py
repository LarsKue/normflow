
import torch

from typing import Optional

from normflow.common import Invertible


class Transform(Invertible):
    """
    Base class for invertible transforms as needed by normalizing flows
    """
    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward transform
        @param x: Input Tensor
        @param condition: Keyword-Only Condition for Conditional Transforms
        @return: 1. Transformed Input z
                 2. log(|det J|) where J is the corresponding Jacobian
        """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse transform
        @param z: Input Tensor
        @param condition: Keyword-Only Condition for Conditional Transforms
        @return: 1. Inverse Transformed Input x
                 2. log(|det J|) where J is the corresponding Jacobian
        """
        raise NotImplementedError

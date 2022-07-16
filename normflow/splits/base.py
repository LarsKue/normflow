
import torch

from typing import Optional

from normflow.common import Invertible


class Split(Invertible):
    """
    Base class for invertible input splits
    """
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Create disjoint outputs from input
        @param x: Input Tensor
        @return: Disjointly Split Input Parts
        """
        raise NotImplementedError

    def inverse(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Merge disjoint inputs into output
        @param parts: tuple of Input Parts
        @return: Joined Inputs
        """
        raise NotImplementedError

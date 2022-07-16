

import torch

from typing import Union

from .base import Split


class SizedSplit(Split):
    """
    Split input into chunks of given sizes
    """
    def __init__(self, sizes: Union[tuple[int, ...], list[int]], dim: int = -1):
        super().__init__()
        self.sizes = sizes
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(x, self.sizes, self.dim)

    def inverse(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(parts, dim=self.dim)

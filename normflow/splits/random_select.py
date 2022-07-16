
import torch

from typing import Union

from .base import Split


class RandomSelect(Split):
    """
    Split input by selecting the given number of random indices along the given axis
    """
    def __init__(self, sizes: Union[tuple[int, ...], list[int]], dim: int = -1):
        super().__init__()
        self.sizes = sizes
        self.permutation = torch.randperm(sum(sizes))
        self.inverse_permutation = torch.argsort(self.permutation)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        permuted = torch.index_select(x, dim=self.dim, index=self.permutation.to(x.device))
        parts = torch.split(permuted, self.sizes, self.dim)

        return parts

    def inverse(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        permuted = torch.cat(parts, dim=self.dim)
        x = torch.index_select(permuted, dim=self.dim, index=self.inverse_permutation.to(permuted.device))

        return x

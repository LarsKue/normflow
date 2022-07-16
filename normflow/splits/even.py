
import torch

from typing import Optional

from .base import Split


class EvenSplit(Split):
    """
    Split input into sections of even size along the given dimension
    If `input.shape[dim]` is not divisible by `sections`,
    the first few sections will have one additional element.
    """
    def __init__(self, sections: int = 2, dim: int = -1):
        super().__init__()
        self.sections = sections
        self.dim = dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.tensor_split(x, sections=self.sections, dim=self.dim)

    def inverse(self, parts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(parts, dim=self.dim)

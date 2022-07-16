
import torch

from .base import Permutation


class ReversePermutation(Permutation):
    """ Reverses Input """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flip(self.dim)

        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z.flip(self.dim)

        return x


import torch

from normflow.common import Invertible


class Permutation(Invertible):
    """ Permute Input on a Given Dimension """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Perform the Forward Permutation """
        raise NotImplementedError

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """ Perform the Inverse Permutation """
        raise NotImplementedError

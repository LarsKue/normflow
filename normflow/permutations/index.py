
import torch


from .base import Permutation


class IndexPermutation(Permutation):
    """ Permute Based on a Set of Indices """

    def __init__(self, indices: torch.Tensor, dim: int = -1) -> None:
        super().__init__()
        # TODO: more checks like these
        if indices.ndim != 1:
            raise ValueError(f"Indices must be 1D, but got {indices.shape}")
        self.indices = indices
        self.dim = dim

    @property
    def inverse_indices(self) -> torch.Tensor:
        return torch.argsort(self.indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.index_select(x, self.dim, self.indices.to(x.device))

        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.index_select(z, self.dim, self.inverse_indices.to(z.device))

        return x


import torch


from normflow.couplings import Coupling
from normflow.splits import Split
from normflow.permutations import Permutation
from .base import Block


class SCPBlock(Block):
    """
    An SCP (Split-Couple-Permute) Block Encompasses
    1. Splitting input into disjoint parts
    2. Transforming at least one part with a coupling
    3. Concatenating the transformed and untransformed parts
    4. Applying some permutation
    """
    def __init__(self, split: Split, coupling: Coupling, permutation: Permutation) -> None:
        super().__init__()
        self.split = split
        self.coupling = coupling
        self.permutation = permutation

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.split.forward(x)
        transformed, logabsdet = self.coupling.forward(parts, condition=condition)
        joined = self.split.inverse(transformed)
        permuted = self.permutation.forward(joined)

        return permuted, logabsdet

    def inverse(self, permuted: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        joined = self.permutation.inverse(permuted)
        transformed = self.split.forward(joined)
        parts, logabsdet = self.coupling.inverse(transformed, condition=condition)
        x = self.split.inverse(parts)

        return x, logabsdet

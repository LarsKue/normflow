
import torch
import torch.nn as nn


from normflow.couplings import Coupling
from normflow.splits import Split
from .base import Block


class SCCBlock(Block):
    """
    An SCC (Split-Couple-Couple) Block Encompasses
    1. Splitting input into disjoint parts
    2. Transforming all input parts with an even number of couplings (usually 2)
    3. Concatenating the transformed parts
    """
    def __init__(self, split: Split, *couplings: Coupling) -> None:
        super().__init__()
        self.split = split
        self.couplings = nn.ModuleList(couplings)

        # TODO: find a better way to do this (parts could also be shifted each step)
        if not len(couplings) % 2 == 0:
            raise ValueError(f"{self.__class__.__name__} needs an even number of couplings, "
                             f"since parts are reversed in every step.")

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.split.forward(x)
        transformed = parts
        logabsdet = x.new_zeros(x.shape[0])
        for coupling in self.couplings:
            transformed, det = coupling.forward(transformed, condition=condition)
            # reverse parts
            transformed = transformed[::-1]
            logabsdet += det

        joined = self.split.inverse(transformed)

        return joined, logabsdet

    def inverse(self, joined: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = self.split.forward(joined)

        parts = transformed
        logabsdet = joined.new_zeros(joined.shape[0])
        for coupling in reversed(self.couplings):
            parts = parts[::-1]
            parts, det = coupling.inverse(parts, condition=condition)
            logabsdet += det

        x = self.split.inverse(parts)

        return x, logabsdet

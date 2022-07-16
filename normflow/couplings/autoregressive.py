import torch
import torch.nn as nn


from .base import Coupling


class AutoRegressiveCoupling(Coupling):
    """ Coupling that applies an AR transform based on a set of parameters determined through a neural network """

    def __init__(self, params_network: nn.Module) -> None:
        super().__init__()
        self.params_network = params_network

    def forward(self,
                parts: tuple[torch.Tensor, ...],
                *,
                condition: torch.Tensor = None,
                ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:

        x1, x2 = parts
        z1 = x1

        params = self.params_network(x1, condition=condition)
        z2, logabsdet = self._forward(params, x2)

        return (z1, z2), logabsdet

    def _forward(self, params: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def inverse(self,
                transformed_parts: tuple[torch.Tensor, ...],
                *,
                condition: torch.Tensor = None,
                ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:

        z1, z2 = transformed_parts
        x1 = z1

        params = self.params_network(x1, condition=condition)
        x2, logabsdet = self._inverse(params, z2)

        return (x1, x2), logabsdet

    def _inverse(self, params: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

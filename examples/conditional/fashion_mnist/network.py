
import torch
import torch.nn as nn

from itertools import chain

from typing import Type


class AutoRegressiveNetwork(nn.Module):
    """ Parameter Network for Autoregressive Couplings """
    def __init__(self,
                 in_features: int,
                 condition_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 activation: Type[nn.Module] = nn.ReLU
                 ) -> None:
        super().__init__()
        in_layer = nn.Linear(in_features=in_features + condition_features, out_features=hidden_features)

        hidden_layers = [
            nn.Linear(in_features=hidden_features, out_features=hidden_features) for _ in range(hidden_layers - 1)
        ]
        activations = [activation() for _ in hidden_layers]
        dropout = [nn.Dropout() for _ in hidden_layers]

        hidden_layers = list(chain.from_iterable(zip(dropout, hidden_layers, activations)))

        out_layer = nn.Linear(in_features=hidden_features, out_features=out_features)

        self.network = nn.Sequential(
            in_layer,
            activation(),
            *hidden_layers,
            out_layer,
        )

    def forward(self, x1: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        if condition is not None:
            x = torch.cat((x1, condition), dim=-1)
        else:
            x = x1

        return self.network.forward(x)

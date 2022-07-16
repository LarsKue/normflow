import torch
import torch.nn as nn


from .base import Transform


class CompositeTransform(Transform):
    """
    A transform that is a composite of many sub-transforms
    Output is computed as follows:
    z = f_n(f_n-1(...(f_1(x))...))

    See also: https://en.wikipedia.org/wiki/Function_composition
    """
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        logabsdet = x.new_zeros(x.shape[0])

        for transform in self.transforms:
            z, det = transform.forward(z, condition=condition)
            logabsdet += det

        return z, logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        logabsdet = z.new_zeros(z.shape[0])

        for transform in reversed(self.transforms):
            x, det = transform.inverse(x, condition=condition)
            logabsdet += det

        return x, logabsdet

import torch


from .autoregressive import AutoRegressiveCoupling


class AdditiveCoupling(AutoRegressiveCoupling):
    """
    A special case of the affine coupling where s = 1 and log det J = 0
    """

    def _forward(self, params: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift = params

        z2 = x2 + shift

        logabsdet = x2.new_zeros(x2.shape[0])

        return z2, logabsdet

    def _inverse(self, params: torch.Tensor, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift = params

        x2 = z2 - shift

        logabsdet = z2.new_zeros(z2.shape[0])

        return x2, logabsdet

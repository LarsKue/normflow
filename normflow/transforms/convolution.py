
import torch
import torch.nn as nn
from torch.nn.functional import conv2d


from .base import Transform


class ConvolutionTransform1x1(Transform):
    """
    Invertible 1x1 Convolution that generalizes a learned channel-permutation

    Introduced by arXiv:1807.03039
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        # initialize w as a random orthogonal matrix
        shape = (channels, channels)
        q, r = torch.linalg.qr(torch.randn(*shape))
        # weight matrix shape is (out_channels, in_channels, k1, k2)
        # where k1, k2 are the kernel sizes
        # for us, k1 = k2 = 1 and in_channels = out_channels
        self.weight = nn.Parameter(q.reshape(*shape, 1, 1))

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape

        z = conv2d(x, self.weight)
        # computing det W directly is O(c^3) so acceptable
        logabsdet = torch.log(torch.abs(torch.det(torch.squeeze(self.weight))))
        logabsdet = height * width * logabsdet

        return z, logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = z.shape
        w_inv = torch.linalg.inv(self.weight)

        x = conv2d(z, w_inv)

        logabsdet = -torch.log(torch.abs(torch.det(torch.squeeze(self.weight))))
        logabsdet = height * width * logabsdet

        return x, logabsdet

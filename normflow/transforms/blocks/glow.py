
import torch.nn as nn

from normflow.couplings import AffineCoupling
from normflow.permutations import ReversePermutation
from normflow.splits import EvenSplit
from normflow.transforms import ActNorm, CompositeTransform, ConvolutionTransform1x1, SqueezeTransform
from .base import Block
from .scp import SCPBlock


class GlowBlock(CompositeTransform, Block):
    """
    One Inner Step of the GLOW Architecture.
    Encompasses
    1. Normalizing the Input with ActNorm
    2. Applying an Invertible 1x1 Convolution
    3. Transforming the Convolution with an Affine SCP Block
    """
    def __init__(self, channels: int, affine_params_network: nn.Module) -> None:
        actnorm = ActNorm(shape=(1, channels, 1, 1))
        conv = ConvolutionTransform1x1(channels)
        affine = SCPBlock(
            # split along the channel dimension in two halves
            split=EvenSplit(dim=1),
            # apply affine coupling
            coupling=AffineCoupling(affine_params_network, dim=1),
            # reverse order of channels
            permutation=ReversePermutation(dim=1),
        )

        super().__init__(
            actnorm,
            conv,
            affine
        )


import torch

import nflows.utils as utils

from .base import Distribution


class StandardNormal(Distribution):
    """ Multivariate normal distribution with zero mean and unit covariance """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def sample(self, shape=torch.Size()) -> torch.Tensor:
        shape = (*shape, *self.shape)
        return torch.randn(shape)

    def log_prob(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        # since mean and covariance are known, the log likelihood
        # is simply the log of the probability density function
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Likelihood_function
        # in this case we can drop:
        # ln(|Sigma|) = 0 since Sigma = 1
        # Sigma^-1 = 1
        # mu = 0
        # so it becomes
        k = torch.prod(torch.tensor(self.shape).to(x.device))
        xtx = utils.sum_except_batch(x ** 2)
        return -0.5 * (xtx + k * torch.log(torch.tensor(2 * torch.pi)).to(x.device))

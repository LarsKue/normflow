import torch
from torch.distributions import Distribution


from normflow.common import Invertible
from normflow.distributions import Distribution as PrimitiveDistribution
from normflow.transforms import Transform
from normflow import utils


class Flow(Invertible):
    """
    Base class for Normalizing Flows, consisting of a Transform and a latent Distribution
    """
    def __init__(self, transform: Transform, distribution: Distribution | PrimitiveDistribution):
        super().__init__()
        self.transform = transform
        self.distribution = distribution

    def forward(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """ Return the forward transform and the log likelihood for the latent samples """
        z, logabsdet = self.transform.forward(x, condition=condition)

        # TODO: Conditional Log Prob? Maybe use a custom Distribution Base Class
        # log_prob = self.distribution.log_prob(z, condition=condition)
        # ignore the condition for now (all priors equally likely)
        log_prob = self.distribution.log_prob(z)

        if log_prob.dim() > 1:
            # product of probabilities is the total probability
            # in the log this turns into a sum
            log_prob = utils.sum_except_batch(log_prob)

        log_prob = log_prob.to(logabsdet.device)

        return z, log_prob + logabsdet

    def inverse(self, z: torch.Tensor, *, condition: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """ Return the inverse transform """
        # TODO: Conditional Log Prob? Maybe use a custom Distribution Base Class
        # log_prob = self.distribution.log_prob(z, condition=condition)
        # ignore the condition for now (all priors equally likely)
        log_prob = self.distribution.log_prob(z)

        if log_prob.dim() > 1:
            log_prob = utils.sum_except_batch(log_prob)

        x, logabsdet = self.transform.inverse(z, condition=condition)

        log_prob = log_prob.to(logabsdet.device)

        return x, log_prob - logabsdet

    def __call__(self, xz: torch.Tensor, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            return self.inverse(xz)

        return self.forward(xz)

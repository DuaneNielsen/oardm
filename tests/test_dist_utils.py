from typing import Optional, Any, Dict

import torch
import distributions_old
from matplotlib import pyplot as plt
from torch import distributions as dst

from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import Distribution

def test_dist():
    image = (torch.rand(2, 3, 28, 28) * 255).to(torch.uint8)


def test_sample():
    x = (torch.rand(2, 3, 28, 28) * 255).long().float()/255
    params = torch.randn(2, 64, 28, 28)

    dist = distributions_old.DiscretizedLogisticLikelihood(ch_in=64, color_channels=3, n_bins=256)
    params = dist.distr_params(params)
    y = dist.sample(params)
    log_prob = dist.log_likelihood(x, params)
    plt.imshow(y[0, 0].detach())
    plt.show()


def test_transformed_dist():

    range = torch.linspace(0.01, 1.)
    base_distribution = dst.Uniform(0, 1)

    logistic = TransformedDistribution(
        base_distribution,
        [dst.SigmoidTransform().inv, dst.AffineTransform(loc=0.5, scale=1.0)]
    )

    lognormal = TransformedDistribution(
        dst.Normal(loc=0.5, scale=1.0),
        [dst.ExpTransform()]
    )

    dist = dst.LogNormal(loc=0.5, scale=1.0)
    logprobs = dist.log_prob(range)
    plt.plot(range, logistic.log_prob(range).exp(), color='red')
    plt.plot(range, logprobs.exp(), color='blue')
    plt.plot(range, lognormal.log_prob(range).exp(), color='green')
    plt.show()


def test_discretized_logistic():
    pass

import torch
import distributions as d


def test_logprob():
    params = torch.ones(2, 30, 4, 5)
    x = torch.ones(2, 3, 4, 5)
    logprob = d.discretized_mix_logistic_loss(x, params)


def test_sample():
    params = torch.ones(2, 30, 4, 5)
    x = d.sample_from_discretized_mix_logistic(params)
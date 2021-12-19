import torch
from torch.nn.functional import softplus
from torch.distributions import OneHotCategorical


def unpack_params(params, nr_mix):
    pi_logits = params[:, :, :, :nr_mix]
    remainder = params[:, :, :, nr_mix:].reshape(params.shape[0:3] + (3, nr_mix * 3))
    means, scale, coeffs = torch.chunk(remainder, chunks=3, dim=-1)
    return pi_logits, means, scale, coeffs


def condition_rgb_means(x, means, coeffs):
    """
    computes means given rgb pixel values of x and the co-effs
    """
    coeffs = torch.tanh(coeffs)
    r, g, b = x.chunk(3, dim=3)
    mr, mg, mb = means.chunk(3, dim=3)
    c1, c2, c3 = coeffs.chunk(3, dim=3)

    # compute means with respect to means only for visualization purposes
    with torch.no_grad():
        mmg = mg + c1 * mr
        mmb = mb + c2 * mr + c3 * mg

    # means with respect to input image
    mg = mg + c1 * r
    mb = mb + c2 * r + c3 * g

    return torch.cat([mr, mg, mb], dim=3), torch.cat([mr.detach(), mmg, mmb], dim=3)


def discretize(x, means, scale, gridsize):
    centered_x = x - means
    inv_stdv = softplus(scale)
    plus_in = inv_stdv * (centered_x + gridsize)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - gridsize)
    cdf_min = torch.sigmoid(min_in)

    # log probability for rightmost grid value, tail of distribution
    log_cdf_plus = plus_in - softplus(plus_in)
    # log probability for leftmost grid value, tail of distribution
    log_one_minus_cdf_min = - softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    return log_cdf_plus, log_one_minus_cdf_min, cdf_delta


def redistribute_tails(x, log_cdf_plus, log_one_minus_cdf_min, cdf_delta):
    is_last_bin = x > 0.9999
    is_first_bin = x < -0.9999

    log_cdf_delta = torch.log(torch.clamp(cdf_delta, min=1e-12))

    # Here the tails of the distribution are assigned, if applicable.
    log_probs = is_last_bin * log_one_minus_cdf_min + ~is_last_bin * log_cdf_delta
    log_probs = is_first_bin * log_cdf_plus + ~is_first_bin * log_probs

    return log_probs


def discretized_mix_logistic_rgb(x, params, gridsize):

    # convert to N, H, W, C format
    x = x.permute(0, 2, 3, 1)
    params = params.permute(0, 2, 3, 1)

    assert len(x.shape) == 4
    N, H, W, C = x.shape
    assert C == 3
    assert len(params.shape) == 4

    nr_mix = params.shape[3] // 10
    pi_logits, means, scales, coeffs = unpack_params(params, nr_mix)

    x = x.reshape(*x.shape, 1)
    x = x.repeat_interleave(nr_mix, dim=-1)

    means, dist_means = condition_rgb_means(x, means, coeffs)

    log_cdf_plus, log_one_minus_cdf_min, cdf_delta = discretize(x, means, scales, gridsize)
    log_probs = redistribute_tails(x, log_cdf_plus, log_one_minus_cdf_min, cdf_delta)

    assert log_probs.shape == (N, H, W, C, nr_mix)

    log_pi = torch.log_softmax(pi_logits, dim=-1)
    log_probs = torch.logsumexp(log_probs + log_pi[..., None, :], dim=-1)

    assert log_probs.shape == (N, H, W, C)

    log_probs = log_probs.sum(dim=-1)

    assert log_probs.shape == (N, H, W)

    return log_probs, torch.sum(pi_logits.unsqueeze(-2).exp() * dist_means, dim=-1).permute(0, 3, 1, 2)


def select_logistic(selection, means, log_scales, coeffs):
    means = torch.sum(means * selection, dim=4)
    log_scales = torch.sum(log_scales * selection, dim=4)
    coeffs = torch.sum(coeffs * selection, dim=4)
    return means, log_scales, coeffs


def standard_logistic(u, means, log_scales):
    standard_logistic = torch.log(u) - torch.log(1. - u)
    scale = 1. / softplus(log_scales)
    x = means + scale * standard_logistic
    return x


def color_subpixels(x, coeffs):
    x0, x1, x2 = x.chunk(3, dim=3)
    c0, c1, c2 = coeffs.chunk(3, dim=3)

    x0 = x0.clamp(min=-1, max=1.)
    x1 = x1 + c0 * x0
    x1 = x1.clamp(min=-1, max=1.)
    x2 = x2 + c1 * x0 + c2 * x1
    x2 = x2.clamp(min=-1, max=1.)

    return torch.cat([x0, x1, x2], dim=3)


def sample_from_discretized_mix_logistic_rgb(params):
    params = params.permute(0, 2, 3, 1)
    nr_mix = params.shape[3] // 10
    pi_logits, means, log_scales, coeffs = unpack_params(params, nr_mix)
    coeffs = torch.tanh(coeffs)

    selection = OneHotCategorical(logits=pi_logits).sample().unsqueeze(-2)
    means, log_scales, coeffs = select_logistic(selection, means, log_scales, coeffs)

    u = torch.rand(means.shape, device=params.device).clamp(min=1e-5, max=1. - 1e-5)

    x = standard_logistic(u, means, log_scales)
    sample_x = color_subpixels(x, coeffs)

    return sample_x.permute(0, 3, 1, 2)

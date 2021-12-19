import torch
import distributions_old as d
import jax.numpy as jnp
import jax
import numpy as np
from jax2torch import jax2torch
from distribution_utils_jax import discretized_mix_logistic_rgb, sample_from_discretized_mix_logistic_rgb
import distribution_utils_jax as dujax
import distribution_utils as du


def test_logprob():
    params = torch.ones((2, 30, 4, 5))
    x = torch.ones((2, 3, 4, 5))
    x = x * 2. - 1.

    logprob, means = d.discretized_mix_logistic_loss(x, params)

    x = jnp.ones((2, 4, 5, 3))
    x = x * 2. - 1.
    params = jnp.ones((2, 4, 5, 30))
    gridsize = 1. / (256 - 1.)

    log_probs = discretized_mix_logistic_rgb(x, params, gridsize)

    assert (np.allclose(logprob.numpy(), log_probs, atol=1e-4))


def test_logprob_wrapped():
    torch_discretized_mix_logistic_rgb = jax2torch(discretized_mix_logistic_rgb)

    x = torch.ones((2, 3, 4, 5))
    params = torch.ones((2, 30, 4, 5))
    x = x * 2. - 1.
    gridsize = 1. / (256 - 1.)
    torch_log_probs = torch_discretized_mix_logistic_rgb(x, params, gridsize)

    x = jnp.ones((2, 3, 4, 5))
    x = x * 2. - 1.
    params = jnp.ones((2, 30, 4, 5))
    gridsize = 1. / (256 - 1.)

    log_probs = discretized_mix_logistic_rgb(x, params, gridsize)

    assert (np.allclose(torch_log_probs.numpy(), log_probs, atol=1e-4))


def test_sample():
    params = torch.ones(2, 30, 4, 5)
    x = d.sample_from_discretized_mix_logistic(params)


def test_wrapped_sample():
    torch_sample_from_discretized_mix_logistic_rgb = jax2torch(sample_from_discretized_mix_logistic_rgb)
    params = torch.ones((2, 30, 4, 5))
    nr_mix = torch.tensor([3], dtype=torch.int32)
    seed = torch.tensor([0], dtype=torch.int32)
    sample = torch_sample_from_discretized_mix_logistic_rgb(seed, params, nr_mix)
    from matplotlib import pyplot as plt
    plt.imshow(sample[0])
    plt.show()


def test_dmlrgb_refactor():

    def baseline_discretized_mix_logistic_rgb(x, params, gridsize):
        """ """
        x = jnp.transpose(x, axes=(0, 2, 3, 1))
        params = jnp.transpose(params, axes=(0, 2, 3, 1))

        """Computes discretized mix logistic for 3 channel images."""
        assert len(x.shape) == 4
        batchsize, height, width, channels = x.shape
        assert channels == 3
        assert len(params.shape) == 4

        # 10 = [1 (mixtures) + 3 (means) + 3 (log_scales) + 3 (coeffs)].
        nr_mix = params.shape[3] // 10
        pi_logits = params[:, :, :, :nr_mix]  # mixture coefficients.
        remaining_params = params[:, :, :, nr_mix:].reshape(*x.shape, nr_mix * 3)
        means = remaining_params[:, :, :, :, :nr_mix]
        pre_act_scale = remaining_params[:, :, :, :, nr_mix:2 * nr_mix]

        # Coeffs are used to autoregressively model the _mean_ parameter of the
        # distribution using the Red (variable x0) and Green (variable x1) channels.
        # For Green (x1) and Blue (variable x2).
        # There are 3 coeff channels, one for (x1 | x0) and two for (x2 | x0, x1)
        coeffs = jax.nn.tanh(remaining_params[:, :, :, :, 2 * nr_mix:])

        x = x.reshape(*x.shape, 1)
        x = x.repeat(nr_mix, axis=-1)

        m1 = means[:, :, :, 0:1, :]
        m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
              * x[:, :, :, 0, :]).reshape(batchsize, height, width, 1, nr_mix)

        m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
              coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).reshape(
            batchsize, height, width, 1, nr_mix)

        means = jnp.concatenate([m1, m2, m3], axis=3)

        centered_x = x - means
        inv_stdv = jax.nn.softplus(pre_act_scale)
        plus_in = inv_stdv * (centered_x + gridsize)
        cdf_plus = jax.nn.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - gridsize)
        cdf_min = jax.nn.sigmoid(min_in)

        # log probability for rightmost grid value, tail of distribution
        log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
        # log probability for leftmost grid value, tail of distribution
        log_one_minus_cdf_min = -jax.nn.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases

        is_last_bin = (x > 0.9999).astype(jnp.float32)
        is_first_bin = (x < -0.9999).astype(jnp.float32)

        log_cdf_delta = jnp.log(jnp.clip(cdf_delta, a_min=1e-12))

        # Here the tails of the distribution are assigned, if applicable.
        log_probs = is_last_bin * log_one_minus_cdf_min + (
                1. - is_last_bin) * log_cdf_delta

        log_probs = is_first_bin * log_cdf_plus + (
                1. - is_first_bin) * log_probs

        assert log_probs.shape == (batchsize, height, width, channels, nr_mix)

        log_pi = jax.nn.log_softmax(pi_logits, axis=-1)

        log_probs = jax.nn.logsumexp(log_probs + log_pi[Ellipsis, None, :], axis=-1)

        assert log_probs.shape == (batchsize, height, width, channels)

        log_probs = log_probs.sum(-1)

        assert log_probs.shape == (batchsize, height, width)

        return log_probs

    x = jnp.ones((2, 3, 4, 5))
    x = x * 2. - 1.
    params = jnp.ones((2, 30, 4, 5))
    gridsize = 1. / (256 - 1.)

    baseline_log_probs = baseline_discretized_mix_logistic_rgb(x, params, gridsize)
    log_probs = discretized_mix_logistic_rgb(x, params, gridsize)

    assert jnp.allclose(baseline_log_probs, log_probs)


def test_unpack_params():
    params = np.random.uniform(size=(2, 4, 5, 30))
    pi_logits, means, scale, coeffs = dujax.unpack_params(params, nr_mix=3, x_shape=(2, 4, 5, 3))
    t_params = torch.from_numpy(params)
    t_pi_logits, t_means, t_scale, t_coeffs = du.unpack_params(t_params, 3)

    assert t_pi_logits.shape == pi_logits.shape
    assert t_means.shape == means.shape
    assert t_scale.shape == scale.shape
    assert t_coeffs.shape == coeffs.shape

    assert np.allclose(t_pi_logits.numpy(), pi_logits)
    assert np.allclose(t_means.numpy(), means)
    assert np.allclose(t_scale.numpy(), scale)
    assert np.allclose(t_coeffs.numpy(), coeffs)


def test_condition_rgb_means():
    nr_mix = 10
    x = np.random.uniform(size=(2, 4, 5, 3))
    x = x.reshape(*x.shape, 1)
    x = x.repeat(nr_mix, axis=-1)
    means = np.random.uniform(size=(2, 4, 5, 3, nr_mix))
    coeffs = np.random.uniform(size=(2, 4, 5, 3, nr_mix))

    cond_means = dujax.condition_rgb_means(x, means, coeffs, nr_mix)

    x = torch.from_numpy(x)
    means = torch.from_numpy(means)
    coeffs = torch.from_numpy(coeffs)

    t_cond_means = du.condition_rgb_means(x, means, coeffs)

    assert cond_means.shape == t_cond_means.shape
    assert np.allclose(cond_means, t_cond_means.numpy())


def test_discretize():
    nr_mix = 2
    x = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    x = x.reshape(*x.shape, 1)
    x = x.repeat(nr_mix, axis=-1)
    means = np.random.uniform(size=(2, 4, 5, 3, nr_mix)).astype(np.float32)
    scales = np.random.uniform(size=(2, 4, 5, 3, nr_mix)).astype(np.float32)
    gridsize = 1. / (256 - 1.)

    log_cdf_plus, log_one_minus_cdf_min, cdf_delta = dujax.discretize(x, means, scales, gridsize)

    x = torch.from_numpy(x)
    means = torch.from_numpy(means)
    scales = torch.from_numpy(scales)

    t_log_cdf_plus, t_log_one_minus_cdf_min, t_cdf_delta = du.discretize(x, means, scales, gridsize)

    assert t_log_cdf_plus.shape == log_cdf_plus.shape
    assert np.allclose(t_log_cdf_plus.numpy(), log_cdf_plus)

    assert t_log_one_minus_cdf_min.shape == log_one_minus_cdf_min.shape
    assert np.allclose(t_log_one_minus_cdf_min.numpy(), log_one_minus_cdf_min)

    assert t_cdf_delta.shape == cdf_delta.shape

    # it seems torch sigmoid and jax sigmoid give slightly different results
    assert np.allclose(t_cdf_delta.numpy(), cdf_delta, atol=1e-6)


def test_redistribute_tails():
    nr_mix = 2
    x = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    x = x.reshape(*x.shape, 1)
    x = x.repeat(nr_mix, axis=-1)
    eps = np.finfo(np.float32).eps
    log_cdf_plus = np.log(np.random.uniform(size=(2, 4, 5, 3, 2)).astype(np.float32) + eps)
    log_one_minus_cdf_min = np.log(np.random.uniform(size=(2, 4, 5, 3, 2)).astype(np.float32) + eps)
    cdf_delta = np.random.uniform(size=(2, 4, 5, 3, 2)).astype(np.float32)

    log_probs = dujax.redistribute_tails(x, log_cdf_plus, log_one_minus_cdf_min, cdf_delta)

    x = torch.from_numpy(x)
    log_cdf_plus = torch.from_numpy(log_cdf_plus)
    log_one_minus_cdf_min = torch.from_numpy(log_one_minus_cdf_min)
    cdf_delta = torch.from_numpy(cdf_delta)

    t_logprobs = du.redistribute_tails(x, log_cdf_plus, log_one_minus_cdf_min, cdf_delta)

    assert t_logprobs.shape == log_probs.shape
    assert np.allclose(t_logprobs.numpy(), log_probs)


def test_discretized_mix_logistic_rgb():

    x = np.random.uniform(size=(2, 3, 4, 5)).astype(np.float32)
    params = np.random.uniform(size=(2, 30, 4, 5)).astype(np.float32)
    gridsize = 1. / (256 - 1.)

    log_probs = dujax.discretized_mix_logistic_rgb(x, params, gridsize)

    x = torch.from_numpy(x)
    params = torch.from_numpy(params)

    t_log_probs, means = du.discretized_mix_logistic_rgb(x, params, gridsize)

    assert t_log_probs.shape == log_probs.shape
    assert np.allclose(t_log_probs.numpy(), log_probs)
    assert means.shape == (2, 3, 4, 5)

def test_sample_discrete_mix_logistic_rgb():
    params = np.random.uniform(size=(2, 30, 4, 5)).astype(np.float32)
    params = torch.from_numpy(params)

    x = du.sample_from_discretized_mix_logistic(params)
    assert x.shape == (2, 3, 4, 5)
    from matplotlib import pyplot as plt
    def to_image(x):
        x = x + 1
        x = x * 255 / 2
        return x.numpy().astype(np.uint8)
    plt.imshow(to_image(x[0].permute(1, 2, 0)))
    plt.show()
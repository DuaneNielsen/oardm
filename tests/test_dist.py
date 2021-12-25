import torch
import jax.numpy as jnp
import jax
import numpy as np
from jax2torch import jax2torch
from distribution_utils_jax import discretized_mix_logistic_rgb, sample_from_discretized_mix_logistic_rgb
import distribution_utils_jax as dujax
import distribution_utils as du


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

    t_cond_means, dist_means = du.condition_rgb_means(x, means, coeffs)

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


def test_discretized_mix_logistic_rgb_for_nan():

    x = np.random.uniform(size=(2, 3, 4, 5)).astype(np.float32) * 2. - 1.
    mn = np.finfo(np.float32).min
    params = np.full((2, 30, 4, 5), mn)
    gridsize = 1. / (256 - 1.)

    log_probs = dujax.discretized_mix_logistic_rgb(x, params, gridsize)

    x = torch.from_numpy(x)
    params = torch.from_numpy(params)

    t_log_probs, means = du.discretized_mix_logistic_rgb(x, params, gridsize)

    assert t_log_probs.shape == log_probs.shape
    assert np.allclose(t_log_probs.numpy(), log_probs)
    assert means.shape == (2, 3, 4, 5)
    assert ~t_log_probs.isnan().any()
    assert ~means.isnan().any()

    mx = np.finfo(np.float32).max / 1e4
    params = np.full((2, 30, 4, 5), mx)
    params = torch.from_numpy(params)
    t_log_probs, means = du.discretized_mix_logistic_rgb(x, params, gridsize)

    assert ~t_log_probs.isnan().any()
    assert ~means.isnan().any()

    params = np.zeros((2, 30, 4, 5), mx)
    params = torch.from_numpy(params)
    t_log_probs, means = du.discretized_mix_logistic_rgb(x, params, gridsize)

    assert ~t_log_probs.isnan().any()
    assert ~means.isnan().any()


def test_refactor_sample_from_discretized_mix_logistic_rgb():
    def sample_from_discretized_mix_logistic_rgb(seed, params, nr_mix):
        params = jnp.transpose(params, (0, 2, 3, 1))
        # had to hack the signature to get jax to take integers!

        rng = jax.random.PRNGKey(seed)

        """Sample from discretized mix logistic distribution."""
        xshape = params.shape[:-1] + (3,)
        batchsize, height, width, _ = xshape

        # unpack parameters
        pi_logits = params[:, :, :, :nr_mix]
        remaining_params = params[:, :, :, nr_mix:].reshape(*xshape, nr_mix * 3)

        # sample mixture indicator from softmax
        rng1, rng2 = jax.random.split(rng)
        mixture_idcs = dujax.sample_categorical(rng1, pi_logits)

        onehot_values = dujax.onehot(mixture_idcs, nr_mix)

        assert onehot_values.shape == (batchsize, height, width, nr_mix)

        selection = onehot_values.reshape(xshape[:-1] + (1, nr_mix))

        # select logistic parameters
        means = jnp.sum(remaining_params[:, :, :, :, :nr_mix] * selection, axis=4)
        pre_act_scales = jnp.sum(
            remaining_params[:, :, :, :, nr_mix:2 * nr_mix] * selection, axis=4)

        coeffs = jnp.sum(jax.nn.tanh(
            remaining_params[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * selection, axis=4)

        u = jax.random.uniform(rng2, means.shape, minval=1e-5, maxval=1. - 1e-5)

        standard_logistic = jnp.log(u) - jnp.log(1. - u)
        scale = 1. / jax.nn.softplus(pre_act_scales)
        x = means + scale * standard_logistic

        x0 = jnp.clip(x[:, :, :, 0], a_min=-1., a_max=1.)
        # TODO(emielh) although this is typically how it is implemented, technically
        # one should first round x0 to the grid before using it. It does not matter
        # too much since it is only used linearly.
        x1 = jnp.clip(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, a_min=-1., a_max=1.)
        x2 = jnp.clip(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1,
            a_min=-1., a_max=1.)

        sample_x = jnp.concatenate(
            [x0[:, :, :, None], x1[:, :, :, None], x2[:, :, :, None]], axis=3)
        sample_x = jnp.transpose(sample_x, (0, 3, 1, 2))

        return sample_x

    params = np.random.uniform(size=(2, 30, 4, 5)).astype(np.float32)
    seed, nr_mix = 0, 3
    x = sample_from_discretized_mix_logistic_rgb(seed, params, nr_mix)
    x_ = dujax.sample_from_discretized_mix_logistic_rgb(seed, params, nr_mix)
    assert np.allclose(x, x_)


def test_standard_logistic():
    u = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    means = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    log_scales = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    x = dujax.standard_logistic(u, means, log_scales)

    u = torch.from_numpy(u)
    means = torch.from_numpy(means)
    log_scales = torch.from_numpy(log_scales)
    t_x = du.standard_logistic(u, means, log_scales)

    assert np.allclose(t_x.numpy(), x, atol=1e-6)


def test_color_subpixels():
    x = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)
    coeffs = np.random.uniform(size=(2, 4, 5, 3)).astype(np.float32)

    sample = dujax.color_subpixels(x, coeffs)

    x = torch.from_numpy(x)
    coeffs = torch.from_numpy(coeffs)

    t_sample = du.color_subpixels(x, coeffs)

    assert sample.shape == t_sample.shape
    assert np.allclose(sample, t_sample.numpy(), atol=1e-6)


def test_select_logistic():
    nr_mix = 3
    selection = np.random.uniform(size=(2, 4, 5, 1, nr_mix)).astype((np.float32))
    means = np.random.uniform(size=(2, 4, 5, 3, nr_mix)).astype((np.float32))
    log_scales = np.random.uniform(size=(2, 4, 5, 3, nr_mix)).astype((np.float32))
    coeffs = np.random.uniform(size=(2, 4, 5, 3, nr_mix)).astype((np.float32))

    j_means, j_scales, j_coeffs = dujax.select_logistic(selection, means, log_scales, coeffs)

    selection = torch.from_numpy(selection)
    means = torch.from_numpy(means)
    log_scales = torch.from_numpy(log_scales)
    coeffs = torch.from_numpy(coeffs)

    t_means, t_scales, t_coeffs = du.select_logistic(selection, means, log_scales, coeffs)

    assert np.allclose(j_means, t_means.numpy())
    assert np.allclose(j_scales, t_scales.numpy())
    assert np.allclose(j_coeffs, t_coeffs.numpy())


def test_sample_discrete_mix_logistic_rgb():
    params = np.random.uniform(size=(2, 30, 4, 5)).astype(np.float32)
    params = torch.from_numpy(params)

    x = du.sample_from_discretized_mix_logistic_rgb(params)

    assert x.shape == (2, 3, 4, 5)
    from matplotlib import pyplot as plt

    def to_image(x):
        x = x + 1
        x = x * 255 / 2
        return x.numpy().astype(np.uint8)
    # plt.imshow(to_image(x[0].permute(1, 2, 0)))
    # plt.show()
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors


def make_shift_and_scale(params):
    return tfb.Shift(params[..., 0])(tfb.Scale(log_scale=params[..., 1]))

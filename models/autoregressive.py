import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.module import compact
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
import distrax
from tensorflow_probability.substrates import jax as tfp

from typing import Any, List
import dataclasses

Array = Any
tfb = tfp.bijectors


class MaskedDense(nn.Dense):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      mask: mask to apply to the weights.
    """

    mask: Array = None

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """

        kernel = self.param("kernel", self.kernel_init, (jnp.shape(inputs)[-1], self.features), self.param_dtype)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        kernel = self.mask * kernel

        y = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MADE(nn.Module):
    n_params: Any = 2
    n_context: Any = 0
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [32, 32])
    activation: str = "tanh"

    @compact
    def __call__(self, y: Array, context=None):
        if context is not None:
            # Stack with context on the left so that the parameters are autoregressively conditioned on it with left-to-right ordering
            y = jnp.hstack([context, y])

        broadcast_dims = y.shape[:-1]

        masks = tfb.masked_autoregressive._make_dense_autoregressive_masks(params=2, event_size=self.n_params + self.n_context, hidden_units=self.hidden_dims, input_order="left-to-right")  # 2 parameters for scele and shift factors

        for mask in masks[:-1]:
            y = MaskedDense(features=mask.shape[-1], mask=mask)(y)
            y = getattr(jax.nn, self.activation)(y)
        y = MaskedDense(features=masks[-1].shape[-1], mask=masks[-1])(y)

        # Unravel the inputs and parameters
        params = y.reshape(broadcast_dims + (self.n_params + self.n_context, 2))

        # Only take the values corresponding to the parameters of interest for scale and shift; ignore context outputs
        params = params[..., self.n_context :, :]

        return params


class MAF(distrax.Bijector):
    def __init__(self, bijector_fn, unroll_loop=False):
        super().__init__(event_ndims_in=1)

        self.autoregressive_fn = bijector_fn
        self.unroll_loop = unroll_loop

    def forward_and_log_det(self, x, context):
        event_ndims = x.shape[-1]

        if self.unroll_loop:
            y = jnp.zeros_like(x)
            log_det = None

            for _ in range(event_ndims):
                params = self.autoregressive_fn(y, context)
                shift, log_scale = params[..., 0], params[..., 1]
                y, log_det = distrax.ScalarAffine(shift=shift, log_scale=log_scale).forward_and_log_det(x)

        # TODO: Rewrite with Flax primitives rather than jax.lax; these cannot be mixed
        else:

            def update_fn(i, y_and_log_det):
                y, log_det = y_and_log_det
                params = self.autoregressive_fn(y)
                shift, log_scale = params[..., 0], params[..., 1]
                y, log_det = distrax.ScalarAffine(shift=shift, log_scale=log_scale).forward_and_log_det(x)
                return y, log_det

            y, log_det = jax.lax.fori_loop(0, event_ndims, update_fn, (jnp.zeros_like(x), jnp.zeros_like(x)))

        return y, log_det.sum(-1)

    def inverse_and_log_det(self, y, context):
        params = self.autoregressive_fn(y, context)
        shift, log_scale = params[..., 0], params[..., 1]
        x, log_det = distrax.ScalarAffine(shift=shift, log_scale=log_scale).inverse_and_log_det(y)

        return x, log_det.sum(-1)

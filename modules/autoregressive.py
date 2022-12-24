import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import numpy as np
from flax.linen.module import compact
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
import distrax
from typing import Any, List
import dataclasses

Array = Any

tfd = tfp.distributions
tfb = tfp.bijectors


class MaskedDense(nn.Dense):
    """A linear transformation applied over the last dimension of the input.

    Attributes:
      use_context: whether to condition the layer on a context.
      mask: mask to apply to the weights.
    """

    use_context: bool = False
    mask: Array = None

    @compact
    def __call__(self, inputs: Array, context=None) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        if self.use_context and context is not None:
            assert inputs.shape[0] == context.shape[0]  # Batch dim should match
            inputs = jnp.hstack([inputs, context])

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
    bijector_fn: Any
    n_params: Any = 2
    n_context: Any = 0
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [32, 32])
    activation: str = "tanh"

    @compact
    def __call__(self, y: Array, context=None) -> distrax.Bijector:

        n_inputs = y.shape[-1]
        broadcast_dims = y.shape[:-1]

        masks = tfb.masked_autoregressive._make_dense_autoregressive_masks(params=self.n_params, event_size=n_inputs, hidden_units=self.hidden_dims, input_order="left-to-right")

        for idx, mask in enumerate(masks[:-1]):
            y = MaskedDense(features=mask.shape[-1], mask=mask)(y, context=context if idx == 0 else None)
            y = getattr(jax.nn, self.activation)(y)
        y = MaskedDense(features=masks[-1].shape[-1], mask=masks[-1])(y)

        # Unravel the inputs and parameters
        params = y.reshape(broadcast_dims + (n_inputs, self.n_params))

        return self.bijector_fn(params)

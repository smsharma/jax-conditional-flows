from typing import Any, List, Optional
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import flax.linen as nn
from flax.linen.module import compact
import distrax

from models.bijectors import InverseConditional, ChainConditional, TransformedConditional, MaskedCouplingConditional

tfd = tfp.distributions
tfb = tfp.bijectors

Array = Any


class Conditioner(nn.Module):
    event_shape: List[int]
    context_shape: List[int]
    hidden_dims: List[int]
    num_bijector_params: int
    activation: str = "tanh"

    @compact
    def __call__(self, x: Array, context=None):

        # Infer batch dims
        batch_shape = x.shape[: -len(self.event_shape)]
        batch_shape_context = context.shape[: -len(self.context_shape)]
        assert batch_shape == batch_shape_context

        # Flatten event dims
        x = x.reshape(*batch_shape, -1)
        context = context.reshape(*batch_shape, -1)

        x = jnp.hstack([context, x])

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = getattr(jax.nn, self.activation)(x)
        x = nn.Dense(np.prod(self.event_shape) * self.num_bijector_params, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)

        x = x.reshape(*batch_shape, *(tuple(self.event_shape) + (self.num_bijector_params,)))

        return x


class NeuralSplineFlow(nn.Module):
    """Bases on the implementation in the Distrax repo, https://github.com/deepmind/distrax/blob/master/examples/flow.py"""

    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    n_bins: int = 8
    event_shape: Optional[List[int]] = None
    context_shape: Optional[List[int]] = None

    def setup(self):
        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

        # If event shapes are not provided, assume single event and context dimensions
        event_shape = (self.n_dim,) if self.event_shape is None else self.event_shape
        context_shape = (self.n_context,) if self.context_shape is None else self.context_shape

        # Alternating binary mask
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters
        num_bijector_params = 3 * self.n_bins + 1

        self.conditioner = [Conditioner(event_shape=event_shape, context_shape=context_shape, hidden_dims=self.hidden_dims, num_bijector_params=num_bijector_params, activation=self.activation, name="conditioner_{}".format(i)) for i in range(self.n_transforms)]

        bijectors = []
        for i in range(self.n_transforms):
            bijectors.append(MaskedCouplingConditional(mask=mask, bijector=bijector_fn, conditioner=self.conditioner[i]))
            mask = jnp.logical_not(mask)  # Flip the mask after each layer

        self.bijector = InverseConditional(ChainConditional(bijectors))
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(event_shape), jnp.ones(event_shape))

        self.flow = TransformedConditional(self.base_dist, self.bijector)

    def __call__(self, x: Array, context: Array = None) -> Array:
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples, rng, context: Array = None) -> Array:
        return self.flow.sample(seed=rng, sample_shape=(num_samples,), context=context)

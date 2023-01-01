from typing import Any, List
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
    use_context_embedding: bool = False

    @compact
    def __call__(self, x: Array, context=None):

        x = x.reshape(-1, *self.event_shape)
        context = context.reshape(-1, *self.context_shape)

        # Context embedding using a small neural network
        if self.use_context_embedding:
            context = nn.Dense(int(4 * self.context_shape[-1]))(context)
            context = getattr(jax.nn, self.activation)(context)
            context = nn.Dense(self.context_shape[-1])(context)

        x = jnp.hstack([context, x])

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = getattr(jax.nn, self.activation)(x)
        x = nn.Dense(np.prod(self.event_shape) * self.num_bijector_params, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)

        x = x.reshape(-1, *(tuple(self.event_shape) + (self.num_bijector_params,)))

        return x


class NeuralSplineFlow(nn.Module):
    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    n_bins: int = 8
    use_context_embedding: bool = False

    def setup(self):
        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

        event_shape = (self.n_dim,)
        context_shape = (self.n_context,)

        # Alternating binary mask.
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters.
        num_bijector_params = 3 * self.n_bins + 1

        self.conditioner = [Conditioner(event_shape=event_shape, context_shape=context_shape, hidden_dims=self.hidden_dims, num_bijector_params=num_bijector_params, activation=self.activation, use_context_embedding=self.use_context_embedding, name="conditioner_{}".format(i)) for i in range(self.n_transforms)]

        bijectors = []
        for i in range(self.n_transforms):
            bijectors.append(MaskedCouplingConditional(mask=mask, bijector=bijector_fn, conditioner=self.conditioner[i]))
            # Flip the mask after each layer.
            mask = jnp.logical_not(mask)

        self.bijector = InverseConditional(ChainConditional(bijectors))

    def make_flow_model(self):

        flow = self.bijector
        base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.n_dim), jnp.ones(self.n_dim))
        return flow, base_dist

    def __call__(self, x: Array, context: Array = None) -> Array:
        flow, base_dist = self.make_flow_model()
        return TransformedConditional(base_dist, flow).log_prob(x, context=context)

    def sample(self, num_samples, rng, context: Array = None) -> Array:
        flow, base_dist = self.make_flow_model()
        return TransformedConditional(base_dist, flow).sample(seed=rng, sample_shape=(num_samples,), context=context)

from typing import Any, List
import dataclasses

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import flax.linen as nn
import distrax

from models.bijectors import InverseConditional, ChainConditional, TransformedConditional, Permute
from models.autoregressive import MAF, MADE

tfd = tfp.distributions
tfb = tfp.bijectors

Array = Any


class MaskedAutoregressiveFlow(nn.Module):
    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    unroll_loop: bool = True
    use_random_permutations: bool = True
    use_context_embedding: bool = False
    rng_key: Array = jax.random.PRNGKey(42)
    inverse: bool = False

    def setup(self):

        self.made = [MADE(n_params=self.n_dim, n_context=self.n_context, activation=self.activation, hidden_dims=self.hidden_dims, use_context_embedding=self.use_context_embedding, name="made_{}".format(i)) for i in range(self.n_transforms)]

        bijectors = []
        key = self.rng_key
        for i in range(self.n_transforms):
            if self.use_random_permutations:
                permutation = jax.random.choice(key, jnp.arange(self.n_dim), shape=(self.n_dim,), replace=False)
                key, _ = jax.random.split(key)
            else:
                permutation = list(reversed(range(self.n_dim)))
            bijectors.append(Permute(permutation))
            bijectors.append(MAF(bijector_fn=self.made[i], unroll_loop=self.unroll_loop))

        if self.inverse:
            self.bijector = ChainConditional(bijectors)
        else:
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

from typing import Any, List
import dataclasses

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

from models.bijectors import InverseConditional, ChainConditional, TransformedConditional, Permute
from models.autoregressive import MAF, MADE

Array = Any


class MaskedAutoregressiveFlow(nn.Module):
    # Note: Does not currently allow for general event shapes

    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    unroll_loop: bool = True
    use_random_permutations: bool = True
    rng_key: Array = jax.random.PRNGKey(42)
    inverse: bool = False

    def setup(self):

        self.made = [MADE(n_params=self.n_dim, n_context=self.n_context, activation=self.activation, hidden_dims=self.hidden_dims, name="made_{}".format(i)) for i in range(self.n_transforms)]

        bijectors = []
        key = self.rng_key
        for i in range(self.n_transforms):

            # Permutation
            if self.use_random_permutations:
                permutation = jax.random.choice(key, jnp.arange(self.n_dim), shape=(self.n_dim,), replace=False)
                key, _ = jax.random.split(key)
            else:
                permutation = list(reversed(range(self.n_dim)))
            bijectors.append(Permute(permutation))

            bijector_af = MAF(bijector_fn=self.made[i], unroll_loop=self.unroll_loop)
            if self.inverse:
                bijector_af = InverseConditional(bijector_af)  # Flip forward and reverse directions for IAF
            bijectors.append(bijector_af)

        self.bijector = InverseConditional(ChainConditional(bijectors))  # Forward direction goes from target to base distribution
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(self.n_dim), jnp.ones(self.n_dim))

        self.flow = TransformedConditional(self.base_dist, self.bijector)

    def __call__(self, x: Array, context: Array = None) -> Array:
        return self.flow.log_prob(x, context=context)

    def sample(self, num_samples: int, rng: Array, context: Array = None) -> Array:
        return self.flow.sample(seed=rng, sample_shape=(num_samples,), context=context)

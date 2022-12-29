from typing import Any, Tuple

import jax
import jax.numpy as jnp

import distrax
from distrax._src.bijectors.chain import Chain
from distrax._src.bijectors.inverse import Inverse
from distrax._src.distributions.transformed import Transformed


Array = Any
PRNGKey = Array


class ConditionalTransformed(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def sample(self, seed: PRNGKey, sample_shape, z: Array) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, z)
        return y

    def log_prob(self, value: Array, z: Array) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(value, z)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(self, seed: PRNGKey, sample_shape: int, z: Array) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, z)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y


class ConditionalInverse(Inverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, z: Array) -> Array:
        return self._bijector.inverse(x, z)

    def inverse(self, y: Array, z: Array) -> Array:
        return self._bijector.forward(y, z)

    def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
        return self._bijector.inverse_and_log_det(x, z)

    def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
        return self._bijector.forward_and_log_det(y, z)


class ConditionalChain(Chain):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, z: Array) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, z)
        return x

    def inverse(self, y: Array, z: Array) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y, z)
        return y

    def forward_and_log_det(self, x: Array, z: Array) -> Tuple[Array, Array]:
        x, log_det = self._bijectors[-1].forward_and_log_det(x, z)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, z)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, z: Array) -> Tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, z)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, z)
            log_det += ld
        return y, log_det


class ConditionalPermute(distrax.Bijector):
    def __init__(self, permutation, axis=-1):

        super().__init__(event_ndims_in=1)

        self.permutation = jnp.array(permutation)
        self.axis = axis

    def permute_along_axis(self, x, permutation, axis=-1):
        x = jnp.moveaxis(x, axis, 0)
        x = x[permutation, ...]
        x = jnp.moveaxis(x, 0, axis)
        return x

    def forward_and_log_det(self, x, z):
        y = self.permute_along_axis(x, self.permutation, axis=self.axis)
        return y, jnp.zeros(x.shape[: -self.event_ndims_in])

    def inverse_and_log_det(self, y, z):
        inv_permutation = jnp.zeros_like(self.permutation)
        inv_permutation = inv_permutation.at[self.permutation].set(jnp.arange(len(self.permutation)))
        x = self.permute_along_axis(y, inv_permutation)
        return x, jnp.zeros(y.shape[: -self.event_ndims_in])

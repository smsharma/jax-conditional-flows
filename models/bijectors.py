# Bijectors with base from Distrax, additionally allowing for a conditioning context

from typing import Any, List, Tuple, Optional

import math

import jax
import jax.numpy as jnp

import distrax
from distrax._src.bijectors.chain import Chain
from distrax._src.bijectors.inverse import Inverse
from distrax._src.distributions.transformed import Transformed
from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.utils import math

Array = Any
PRNGKey = Array


class TransformedConditional(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def sample(self, seed: PRNGKey, sample_shape: List[int], context: Optional[Array] = None) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, context)
        return y

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(x, context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(self, seed: PRNGKey, sample_shape: List[int], context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y


class InverseConditional(Inverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.inverse(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self._bijector.forward(y, context)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.inverse_and_log_det(x, context)

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        return self._bijector.forward_and_log_det(y, context)


class ChainConditional(Chain):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x, context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y, context)
        return y

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        x, log_det = self._bijectors[-1].forward_and_log_det(x, context)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, context)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class Permute(distrax.Bijector):
    def __init__(self, permutation: Array, axis: int = -1):

        super().__init__(event_ndims_in=1)

        self.permutation = jnp.array(permutation)
        self.axis = axis

    def permute_along_axis(self, x: Array, permutation: Array, axis: int = -1) -> Array:
        x = jnp.moveaxis(x, axis, 0)
        x = x[permutation, ...]
        x = jnp.moveaxis(x, 0, axis)
        return x

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        y = self.permute_along_axis(x, self.permutation, axis=self.axis)
        return y, jnp.zeros(x.shape[: -self.event_ndims_in])

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        inv_permutation = jnp.zeros_like(self.permutation)
        inv_permutation = inv_permutation.at[self.permutation].set(jnp.arange(len(self.permutation)))
        x = self.permute_along_axis(y, inv_permutation)
        return x, jnp.zeros(y.shape[: -self.event_ndims_in])


class MaskedCouplingConditional(MaskedCoupling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_and_log_det(self, x: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_forward_input_shape(x)
        masked_x = jnp.where(self._event_mask, x, 0.0)
        params = self._conditioner(masked_x, context)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x, y0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return y, logdet

    def inverse_and_log_det(self, y: Array, context: Optional[Array] = None) -> Tuple[Array, Array]:
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        params = self._conditioner(masked_y, context)
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(jnp.where(self._mask, 0.0, log_d), self._event_ndims - self._inner_event_ndims)
        return x, logdet

# Conditional normalizing flows in Jax

Implementation of some common normalizing flow models allowing for a conditioning context using [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Distrax](https://github.com/deepmind/distrax). The following are currently implemented:
- Masked/Inverse Autoregressive Flows (MAF/IAF; [Papamakarios et al, 2017](https://arxiv.org/abs/1705.07057) and [Kingma et al, 2016](https://arxiv.org/abs/1606.04934))
- Neural Spline Flows (NSF; [Durkan et al, 2019](https://arxiv.org/abs/1906.04032))

## Examples
- See [notebooks/example.ipynb](notebooks/example.ipynb) for a simple usage example.
- See [notebooks/sbi.ipynb](notebooks/sbi.ipynb) for an example application for neural simulation-based inference (conditional posterior estimation).

## Basic usage

```python
import jax
from models.maf import MaskedAutoregressiveFlow
from models.nsf import NeuralSplineFlow

n_dim = 2  # Feature dim
n_context = 1  # Context dim

## Define flow model
# model = MaskedAutoregressiveFlow(n_dim=n_dim, n_context=n_context, hidden_dims=[128,128], n_transforms=12, activation="tanh", use_random_permutations=False)
model = NeuralSplineFlow(n_dim=n_dim, n_context=n_context, hidden_dims=[128,128], n_transforms=8, activation="gelu", n_bins=4)

## Initialize model and params
key = jax.random.PRNGKey(42)
x_test = jax.random.uniform(key=key, shape=(64, n_dim))
context = jax.random.uniform(key=key, shape=(64, n_context))
params = model.init(key, x_test, context)

## Log-prob and sampling
log_prob = model.apply(params, x_test, jnp.ones((x_test.shape[0], n_context)))
samples = model.apply(params, n_samples, key, jnp.ones((n_samples, n_context)), method=model.sample)
```
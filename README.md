# Conditional normalizing flows in Jax

Implementation of some common normalizing flow models allowing for a conditioning context using [Jax](https://github.com/google/jax), [Flax](https://github.com/google/flax), and [Distrax](https://github.com/deepmind/distrax). The following are currently implemented:
- Masked/Inverse Autoregressive Flows (MAF/IAF; [Papamakarios et al, 2017](https://arxiv.org/abs/1705.07057) and [Kingma et al, 2016](https://arxiv.org/abs/1606.04934))
- Neural Spline Flows (NSF; [Durkan et al, 2019](https://arxiv.org/abs/1906.04032))

See [this notebook](notebooks/example.ipynb) for a simple usage example.
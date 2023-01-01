# Conditional normalizing flows in Jax

A small library implementing some common normalizing flows with context conditioning using Jax, Flax, and Distrax. The following are currently implemented:
- Masked/Inverse Autoregressive Flows (MAF/IAF; [Papamakarios et al, 2017](https://arxiv.org/abs/1705.07057) and [Kingma et al, 2016](https://arxiv.org/abs/1606.04934))
- Neural Spline Flows ([Durkan et al, 2019](https://arxiv.org/abs/1906.04032))
# Conditional normalizing flows in Jax

A small library implementing some common normalizing flows using Jax, Flax, and Distrax. The following are currently implemented:
- Masked/Inverse Autoregressive Flows (MAFs/IAF; [Papamakarios et al, 2017](https://arxiv.org/abs/1705.07057) and [Kingma et al, 2016](https://arxiv.org/abs/1606.04934))
- Glow ([Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039))
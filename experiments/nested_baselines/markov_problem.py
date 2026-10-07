"""The chain model on the Euler--Maruyama latent, in the form NSVI / NPSGLD expect.

Latent w = (x_0, v_0, ..., v_T) with x_i = x_{i-1} + h v_i, so model, latent and
data are those of the chain_ss leg and only the inference algorithm differs.
The physics enters through the energy

    E[w, lam] = h/2 sum_i r_i^T Sigma_r^{-1} r_i + 1/2 |(z_0 - mu_0) / sd_0|^2,
    r_i = A_lam(z_{i-1}) ((v_i - v_{i-1}) / h - a_lam(z_{i-1})),

and lam = exp(log prior_median + prior_sd * u) with u ~ N(0, I).
"""
import math
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class Problem:
    d_w: int
    d_theta: int
    d_x0: int
    log_likelihood_fn: Callable
    energy_fn: Callable
    to_physical: Callable
    init_mean: tuple


def build(system, warmup_iters=8000, warmup_lr=3e-3):
    d, h, T = system.d, system.h, system.T
    n_sub = system.n_sub
    sigma_r = system.sigma_r_true
    y = jnp.asarray(system.y)
    obs_idx = jnp.asarray(np.arange(system.K) * n_sub)
    t_steps = system.t_obs[0] + h * jnp.arange(T)
    log_median = jnp.log(system.prior_median)
    prior_sd = system.prior_sd
    P = len(system.param_names)

    def to_physical(u):
        return jnp.exp(log_median + prior_sd * u)

    def unpack(w):
        x0 = w[:d]
        v = w[d:].reshape(T + 1, d)
        x = x0[None, :] + h * jnp.cumsum(jnp.concatenate([jnp.zeros((1, d)), v[1:]], axis=0), axis=0)
        return x, v

    def log_likelihood_fn(w, u):
        x, _ = unpack(w)
        r = (x[obs_idx] - y) / system.sigma_obs
        return jnp.sum(-0.5 * (r ** 2 + LOG_2PI + 2.0 * math.log(system.sigma_obs)))

    def energy_fn(w, u, _x0):
        x, v = unpack(w)
        lam = to_physical(u)
        acc = (v[1:] - v[:-1]) / h
        r = jax.vmap(system.residual_fn, in_axes=(0, 0, 0, 0, None))(x[:-1], v[:-1], acc, t_steps, lam)
        z0 = jnp.concatenate([x[0], v[0]])
        return (0.5 * h * jnp.sum(r ** 2) / sigma_r ** 2
                + 0.5 * jnp.sum(((z0 - system.init_mean) / system.init_sd) ** 2))

    # warm start: noise-free rollout at the prior median, then a MAP polish of w
    lam0 = system.prior_median
    z = system.init_mean
    vs = [z[d:]]
    for i in range(T):
        z = system.em_step(z, float(t_steps[i]), lam0, h, None, noise=False)
        vs.append(z[d:])
    w = jnp.concatenate([system.init_mean[:d], jnp.concatenate(vs)])
    if warmup_iters:
        import optax
        u0 = jnp.zeros(P)
        objective = lambda w: -log_likelihood_fn(w, u0) + energy_fn(w, u0, None)
        opt = optax.chain(optax.clip_by_global_norm(1e4),
                          optax.adam(optax.cosine_decay_schedule(warmup_lr, warmup_iters)))
        state = opt.init(w)

        @jax.jit
        def step(w, state):
            upd, state = opt.update(jax.grad(objective)(w), state, w)
            return optax.apply_updates(w, upd), state

        for _ in range(warmup_iters):
            w, state = step(w, state)
    return Problem(d_w=d + (T + 1) * d, d_theta=P, d_x0=0, log_likelihood_fn=log_likelihood_fn,
                   energy_fn=energy_fn, to_physical=to_physical,
                   init_mean=tuple(np.concatenate([np.asarray(w), np.zeros(P)]).tolist()))


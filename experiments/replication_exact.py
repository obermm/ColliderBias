#!/usr/bin/env python
"""Exact posteriors of the Fig. 2 configuration over datasets, without sampling.

Oscillator, acceleration residual, sigma_r fixed.  In the identified
coordinates (log c/m, log k/m) three posteriors are evaluated on a grid, with
the paper's log-normal prior pushed forward to these coordinates:

    chain (state space)       exact Kalman evidence of the Euler--Maruyama latent
    collider (linear)         B-PINN on a linear cosine basis x = Phi theta,
                              theta ~ N(0, s0^2 I), normalized Gaussian residual at
                              N trapezoid collocation points; theta integrated out
    chain (linear)            the same latent with log Z_r - log Z_lambda restored exactly

Each dataset is a new SDE path and new observation noise (numpy seed = dataset).
DATA_SUBS sets the step of the data simulator in steps per observation interval:
64 is the replication table (2x finer than the inference grid), 1024 the check
that the chain does not profit from a matched discretization (32x finer).
Writes results/replication/exact.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import colliderbias  # noqa: E402,F401  (double precision)
from colliderbias.paths import RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "replication")
C_M, K_M, SIGMA_R, HORIZON, K_OBS, SIGMA_OBS, X0 = 1.2, 4.0, 0.30, 24.0, 40, 0.02, 1.0
N_SUB = 32                  # Euler--Maruyama steps per observation interval of the inference
N_COLLOC, N_BASIS, BASIS_SD = 128, 65, 0.3
PRIOR_SD = dict(m=0.4, c=0.4, k=0.25)
N_GRID = 120
SEEDS = tuple(range(10))
DATA_SUBS = (64, 1024)

T_OBS = np.linspace(0.0, HORIZON, K_OBS)
DT = T_OBS[1] - T_OBS[0]


def simulate(seed, data_sub):
    """Fine Euler--Maruyama path of x'' = -c/m x' - k/m x + sigma_r xi, observed with noise."""
    rng = np.random.default_rng(seed)
    h = DT / data_sub
    x, v, xs = X0, 0.0, [X0]
    for _ in range(K_OBS - 1):
        for _ in range(data_sub):
            v = v + h * (-C_M * v - K_M * x) + SIGMA_R * np.sqrt(h) * rng.standard_normal()
            x = x + h * v
        xs.append(x)
    return np.array(xs) + SIGMA_OBS * rng.standard_normal(K_OBS)


def kalman(y, g, w2, s):
    """log p(y | c/m = g, k/m = w2, sigma_r = s) of the chain on the Euler--Maruyama latent."""
    h = DT / N_SUB
    a, b = 1 - h * g, -h * w2
    F = jnp.array([[1 + h * b, h * a], [b, a]])
    G = jnp.array([h, 1.0])
    Q = (s ** 2 * h) * jnp.outer(G, G)
    H = jnp.array([1.0, 0.0])

    def update(mu, P, yk):
        S = H @ P @ H + SIGMA_OBS ** 2
        gain = P @ H / S
        r = yk - H @ mu
        return mu + gain * r, P - jnp.outer(gain, H @ P), -0.5 * (jnp.log(2 * jnp.pi * S) + r ** 2 / S)

    def sub(c, _):
        Fn, Qn = c
        return (F @ Fn, F @ Qn @ F.T + Q), None

    (Fn, Qn), _ = jax.lax.scan(sub, (jnp.eye(2), jnp.zeros((2, 2))), None, length=N_SUB)
    mu, P, ll = update(jnp.array([X0, 0.0]), jnp.diag(jnp.array([0.05 ** 2, 0.3 ** 2])), y[0])

    def step(c, yk):
        mu, P, ll = c
        mu, P, l = update(Fn @ mu, Fn @ P @ Fn.T + Qn, yk)
        return (mu, P, ll + l), None

    (_, _, ll), _ = jax.lax.scan(step, (mu, P, ll), y[1:])
    return ll


def linear_surrogate(y):
    """(collider log likelihood, log Z_lambda - log Z_r) on the cosine basis, as a function of (g, w2, s)."""
    tq = np.linspace(0.0, HORIZON, N_COLLOC)
    wq = np.full(N_COLLOC, tq[1] - tq[0])
    wq[0] *= 0.5
    wq[-1] *= 0.5
    k = np.arange(N_BASIS) * np.pi / HORIZON
    phi_obs = np.cos(np.outer(T_OBS, k))
    B0 = jnp.asarray(np.cos(np.outer(tq, k)))
    B1 = jnp.asarray(-np.sin(np.outer(tq, k)) * k)
    B2 = jnp.asarray(-np.cos(np.outer(tq, k)) * k ** 2)
    prec = jnp.eye(N_BASIS) / BASIS_SD ** 2
    OtO = jnp.asarray(phi_obs.T @ phi_obs / SIGMA_OBS ** 2)
    Oty = jnp.asarray(phi_obs.T @ y / SIGMA_OBS ** 2)
    sw = jnp.asarray(np.sqrt(wq))
    yj = jnp.asarray(y)
    log_det_prec = N_BASIS * np.log(1 / BASIS_SD ** 2)

    def logs(g, w2, s):
        R = (B2 + g * B1 + w2 * B0) * sw[:, None] / s          # whitened residual rows
        M = R.T @ R
        La = jnp.linalg.cholesky(prec + M + OtO)
        u = jax.scipy.linalg.solve_triangular(La, Oty, lower=True)
        log_int = (0.5 * log_det_prec - jnp.sum(jnp.log(jnp.diag(La))) + 0.5 * u @ u
                   - 0.5 * jnp.sum(yj ** 2) / SIGMA_OBS ** 2 - 0.5 * K_OBS * jnp.log(2 * jnp.pi * SIGMA_OBS ** 2))
        log_Z_r = jnp.sum(jnp.log(jnp.sqrt(2 * jnp.pi) * s / sw))
        log_Z_lam = 0.5 * log_det_prec - jnp.sum(jnp.log(jnp.diag(jnp.linalg.cholesky(prec + M))))
        return jnp.stack([log_int - log_Z_r, log_Z_lam - log_Z_r])

    return logs


def posteriors(y):
    lg = np.log(C_M) + np.linspace(-3.0, 1.0, N_GRID)
    lw = np.log(K_M) + np.linspace(-1.5, 0.6, N_GRID)
    grid = np.stack(np.meshgrid(lg, lw, indexing="ij"), -1).reshape(-1, 2)
    ex = jnp.exp(jnp.asarray(grid))
    yj = jnp.asarray(y)
    lin = linear_surrogate(y)
    kfv = np.asarray(jax.jit(jax.vmap(lambda z: kalman(yj, z[0], z[1], SIGMA_R)))(ex))
    lin_v = np.asarray(jax.jit(jax.vmap(lambda z: lin(z[0], z[1], SIGMA_R)))(ex))
    collider, tilt = lin_v[:, 0], lin_v[:, 1]
    # the log-normal prior on (m, c, k), pushed forward to (log c/m, log k/m)
    sd = PRIOR_SD
    cov = np.array([[sd["c"] ** 2 + sd["m"] ** 2, sd["m"] ** 2],
                    [sd["m"] ** 2, sd["k"] ** 2 + sd["m"] ** 2]])
    truth = np.log([C_M, K_M])
    d = grid - truth
    log_prior = -0.5 * np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
    res = {}
    for name, ll in (("chain_ss", kfv), ("collider_linear", collider), ("chain_linear", collider - tilt)):
        lp = log_prior + ll
        w = np.exp(lp - lp.max())
        w /= w.sum()
        mean = w @ grid
        W = w.reshape(N_GRID, N_GRID)
        edge = float(W[0].sum() + W[-1].sum() + W[:, 0].sum() + W[:, -1].sum())
        # central 90% interval of each marginal
        covered = []
        for i, (other, axis) in enumerate(((1, lg), (0, lw))):
            lo, hi = np.interp([0.05, 0.95], np.cumsum(W.sum(axis=other)), axis)
            covered.append(bool(lo <= truth[i] <= hi))
        res[name] = dict(mean=mean.tolist(), sd=np.sqrt(w @ (grid - mean) ** 2).tolist(),
                         covered=covered, edge=edge)
    return res


def main():
    rows = []
    for data_sub in DATA_SUBS:
        for seed in SEEDS:
            res = posteriors(simulate(seed, data_sub))
            rows.append(dict(seed=seed, data_sub=data_sub, res=res))
            truth = np.log([C_M, K_M])
            print("data step 1/%-4d seed %d  " % (data_sub, seed) + "  ".join(
                "%s %s" % (name, " ".join("%+.3f" % (r["mean"][i] - truth[i]) for i in range(2)))
                for name, r in res.items()), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exact.json"), "w") as f:
        json.dump(dict(truth=dict(c_m=C_M, k_m=K_M), n_sub=N_SUB, rows=rows), f)
    print("saved", os.path.join(OUT, "exact.json"))


if __name__ == "__main__":
    main()

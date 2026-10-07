#!/usr/bin/env python
"""The dissipation term of a global surrogate on the two conservative systems.

The tilt of a global surrogate contains -1/2 int sum_k |Re s_k| dt, with s_k the
eigenvalues of the drift Jacobian Df_lambda, whereas the chain's divergence
term int div f dt is a bounded boundary term for both systems.  Evaluated along
the noise-free trajectory at the ground truth:

    pendulum   time average of sum_k |Re s_k| (hyperbolic pairs of the chaotic orbit),
               the mean divergence, invariance along (m1, m2) -> alpha (m1, m2),
               and the slope in log l1 at a fixed trajectory
    particle   int div f dt (closed-form orbit: the drive returns the particle to rest),
               int |div f| dt, Delta_diss = -1/2 int |div f| dt, and its slope in
               log c at a fixed trajectory

Prints the numbers only.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import configs as C  # noqa: E402

N_STEPS = {"pendulum": 60000, "particle": 200000}
EPS = 0.05          # half-width of the finite differences in log l1
EPS_C = 0.01        # and in log c


def noise_free_path(system, lam, n):
    """Euler path of the drift (no noise) on n steps over the horizon: (n + 1, 2d) states and times."""
    h = system.horizon / n
    ts = jnp.arange(n + 1) * h

    def step(z, t):
        z = system.em_step(z, t, lam, h, None, noise=False)
        return z, z

    _, zs = jax.lax.scan(step, system.init_mean, ts[:-1])
    return jnp.vstack([system.init_mean[None], zs]), ts


def drift_jacobians(system, Z, ts, lam):
    d = system.d
    f = lambda z, t: jnp.concatenate([z[d:], system.accel(z[:d], z[d:], t, lam)])
    return np.asarray(jax.vmap(jax.jacfwd(f))(Z, ts))


def sum_abs_re(J):
    return np.abs(np.linalg.eigvals(J).real).sum(axis=1)


def pendulum():
    system = C.make_system("pendulum-accel")
    lam = system.lam_true
    Z, ts = noise_free_path(system, lam, N_STEPS["pendulum"])
    J = drift_jacobians(system, Z, ts, lam)
    avg = sum_abs_re(J).mean()
    print("pendulum   <sum_k |Re s_k|> = %.4f over [0, %g]  ->  T/2 <.> = %.2f nats;  mean div f = %.2g"
          % (avg, system.horizon, system.horizon / 2 * avg, np.trace(J, axis1=1, axis2=2).mean()))
    for alpha in (0.5, 2.0):
        lam_a = lam * jnp.array([alpha, alpha, 1.0, 1.0])
        Za, tsa = noise_free_path(system, lam_a, N_STEPS["pendulum"])
        print("           along the ray, alpha = %.1f: <sum_k |Re s_k|> = %.6f  (alpha = 1: %.6f)"
              % (alpha, sum_abs_re(drift_jacobians(system, Za, tsa, lam_a)).mean(), avg))
    vals = [sum_abs_re(drift_jacobians(system, Z, ts, lam * jnp.array([1.0, 1.0, np.exp(e), 1.0]))).mean()
            for e in (-EPS, EPS)]
    print("           log l1 +- %.2f at the fixed trajectory: <.> in [%.5f, %.5f], "
          "|dDelta/dlog l1| = %.3f nats" % (EPS, min(vals), max(vals),
                                             abs(system.horizon / 2 * (vals[1] - vals[0]) / (2 * EPS))))


def particle():
    system = C.make_system("particle-accel")
    lam = system.lam_true
    ts = jnp.linspace(0.0, system.horizon, N_STEPS["particle"] + 1)
    x, v = system.skeleton(lam, ts)
    h = float(ts[1] - ts[0])

    def div_f(lam_):
        # one degree of freedom: the only nonzero eigenvalue of Df is d a / d v = div f
        dv = jax.vmap(lambda xi, vi, t: jax.grad(lambda u: system.accel(xi, u, t, lam_)[0])(vi))
        return np.asarray(dv(x, v, ts))[:, 0]

    trapz = lambda g: float(h * (g.sum() - 0.5 * (g[0] + g[-1])))
    div = div_f(lam)
    abs_int = trapz(np.abs(div))
    slope = [trapz(np.abs(div_f(lam.at[2].set(lam[2] * np.exp(e))))) for e in (-EPS_C, EPS_C)]
    print("particle   int div f dt = %.2g;  int |div f| dt = %.2f  ->  Delta_diss = %.2f nats;  "
          "dDelta_diss/dlog c = %+.1f nats at the fixed trajectory"
          % (trapz(div), abs_int, -0.5 * abs_int, -0.5 * (slope[1] - slope[0]) / (2 * EPS_C)))


def main():
    pendulum()
    particle()


if __name__ == "__main__":
    main()

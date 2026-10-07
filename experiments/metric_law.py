#!/usr/bin/env python
"""Residual-metric law on the oscillator (appendix figure) and the filter-vs-Kalman check.

Along the drift-invariant ray (m, c, k) -> alpha (m, c, k), force residual, the
chain log posterior is exact (Kalman) and the collider differs from it by
sum_i delta_i = T (log h - log(alpha m)), so

    Delta(alpha) - Delta(1) = -T log alpha        (d = 1)

at every step count T.  Also compares the particle filter with the exact
evidence at three points on the ray.  Writes results/laws/metric_law.npz.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import jax  # noqa: E402
import numpy as np  # noqa: E402

import configs as C  # noqa: E402
from colliderbias.paths import RESULTS  # noqa: E402
from colliderbias.statespace import make_log_evidence  # noqa: E402

OUT = os.path.join(RESULTS, "laws")
ALPHAS = np.exp(np.linspace(-0.36, 0.36, 41))
N_SUBS = (8, 16, 32, 64)


def main():
    system = C.make_system("oscillator-force")
    chain = np.zeros((len(N_SUBS), len(ALPHAS)))
    collider = np.zeros_like(chain)
    T = np.zeros(len(N_SUBS))
    for j, n_sub in enumerate(N_SUBS):
        T[j] = (system.K - 1) * n_sub
        h = system.dt_obs / n_sub
        for i, alpha in enumerate(ALPHAS):
            lam = system.along_ray(float(alpha))
            chain[j, i] = float(system.log_evidence_kalman(lam, n_sub)) + float(system.log_prior(lam))
            log_det_A = float(system.log_det_A(None, None, lam))
            collider[j, i] = chain[j, i] + T[j] * (system.d * np.log(h) - log_det_A)
    i1 = len(ALPHAS) // 2
    tilt = (collider - collider[:, i1:i1 + 1]) - (chain - chain[:, i1:i1 + 1])
    pred = -T[:, None] * np.log(ALPHAS)[None, :]
    for j in range(len(N_SUBS)):
        print("T = %5d   tilt span %7.1f nats   chain span %5.2f nats   max |tilt + T log alpha| %.1e"
              % (T[j], np.ptp(tilt[j]), np.ptp(chain[j]), np.max(np.abs(tilt[j] - pred[j]))))

    log_ev = make_log_evidence(system)
    keys = jax.random.split(jax.random.PRNGKey(7), 5)
    pf = []
    for alpha in (0.85, 1.0, 1.2):
        lam = system.along_ray(alpha)
        exact = float(system.log_evidence_kalman(lam))
        est = [float(log_ev(lam, k, system.n_sub, 1000, False)) for k in keys]
        pf.append((alpha, exact, np.mean(est), np.std(est)))
        print("PF vs Kalman at alpha = %.2f: %+.2f nats (MC sd %.2f)"
              % (alpha, np.mean(est) - exact, np.std(est)))

    os.makedirs(OUT, exist_ok=True)
    pf = np.array(pf)
    np.savez(os.path.join(OUT, "metric_law.npz"), alpha=ALPHAS, n_sub=np.array(N_SUBS), T=T,
             log_post_chain=chain, log_post_collider=collider, tilt=tilt, predicted=pred,
             pf_alpha=pf[:, 0], pf_exact=pf[:, 1], pf_mean=pf[:, 2], pf_sd=pf[:, 3])
    print("saved", os.path.join(OUT, "metric_law.npz"))


if __name__ == "__main__":
    main()

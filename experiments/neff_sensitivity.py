#!/usr/bin/env python
"""How n_eff = tr[G (G + P)^-1] depends on where the curvature is evaluated (oscillator B-PINN).

Chain (surrogate) measures n_eff once, at the last draw theta of the matching
B-PINN run and at lam_true.  Here it is re-evaluated for the oscillator's force
residual, the one cell whose correction depends on n_eff,

    at the last theta of both oscillator B-PINN runs (lam_true),
    at the chain's posterior mean, and over 20 draws of the chain, the B-PINN and the prior,
    at a MAP fit of the network at the chain's posterior mean.

Reads results/runs/oscillator-{force,accel}/{bpinn,chain_ss}.npz.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import configs as C  # noqa: E402
from colliderbias.bpinn import BPINN, Surrogate  # noqa: E402
from colliderbias.paths import RUNS  # noqa: E402
from colliderbias.runs import load_run  # noqa: E402

CELLS = ("oscillator-force",)
THETA_CELLS = CELLS + ("oscillator-accel",)
N_DRAWS = 20


def spread(values):
    return "%.1f-%.1f (median %.1f)" % (np.min(values), np.max(values), np.median(values))


def main():
    thetas = {}
    for cell in THETA_CELLS:
        r = load_run(os.path.join(RUNS, cell, "bpinn.npz"))
        if r is not None and r.theta_last is not None:
            thetas[cell] = jnp.asarray(r.theta_last)
    for cell in CELLS:
        chain = load_run(os.path.join(RUNS, cell, "chain_ss.npz"))
        bpinn = load_run(os.path.join(RUNS, cell, "bpinn.npz"))
        if chain is None or cell not in thetas:
            print("%s: needs chain_ss and bpinn (with theta_last)" % cell)
            continue
        system = C.make_system(cell)
        _, _, net_kw = C.cell_settings(cell)
        model = BPINN(system, Surrogate(system, **net_kw))
        theta = thetas[cell]
        n_eff = lambda th, lam: model.measure_n_eff(th, jnp.asarray(lam))
        pick = lambda draws: draws[np.linspace(0, len(draws) - 1, N_DRAWS).astype(int)]
        rng = np.random.default_rng(0)
        prior = np.exp(np.log(np.asarray(system.prior_median))
                       + np.asarray(system.prior_sd) * rng.standard_normal((N_DRAWS, len(system.param_names))))
        mean = chain.draws.mean(0)
        kw = C.BPINN["oscillator"]
        theta_map = model.fit_map(jnp.asarray(mean), jax.random.PRNGKey(0), kw["map_steps"],
                                  init=kw["map_init"], n_chunks=kw["map_chunks"])
        print("%s  (Nd = %d)" % (cell, model.net.N * system.d))
        print("   at its own B-PINN state, lam_true   %.1f" % n_eff(theta, system.lam_true))
        print("   last theta of each B-PINN run      "
              + "  ".join("%s %.1f" % (c, n_eff(t, system.lam_true)) for c, t in thetas.items()))
        print("   at the chain's posterior mean       %.1f" % n_eff(theta, mean))
        print("   over draws of the chain             %s" % spread([n_eff(theta, p) for p in pick(chain.draws)]))
        if bpinn is not None:
            print("   over draws of the B-PINN            %s" % spread([n_eff(theta, p) for p in pick(bpinn.draws)]))
        print("   over draws of the prior             %s" % spread([n_eff(theta, p) for p in prior]))
        print("   at a MAP fit at the chain mean      %.1f" % n_eff(theta_map, mean), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run one leg of one cell and store it as results/runs/<cell>/<leg>.npz.

    python experiments/run.py oscillator-accel chain_ss
    python experiments/run.py oscillator-accel bpinn --quick
    python experiments/run.py oscillator-force chain_ss --seed 1   # -> chain_ss-seed1.npz
    python experiments/run.py oscillator-accel-d3 bpinn             # data seed 3
    python experiments/run.py --list              # every run behind the paper
    python experiments/run.py --list --groups replication

chain_surrogate reads theta from the bpinn run of the same cell to measure
n_eff, so run bpinn first (or pass --n-eff).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import jax  # noqa: E402
import numpy as np  # noqa: E402

import configs as C  # noqa: E402
from colliderbias import guards  # noqa: E402
from colliderbias.bpinn import BPINN, Surrogate  # noqa: E402
from colliderbias.paths import RUNS  # noqa: E402
from colliderbias.runs import load_run, save_run  # noqa: E402
from colliderbias.statespace import run_kalman_rwm, run_path_nuts, run_pmmh  # noqa: E402

RESULTS = RUNS


def run_state_space(system, leg, n_particles, pmmh, check):
    collider = leg.startswith("collider")
    if check:
        guards.ray(system)
        if collider:
            guards.tilt(system, n_particles)
        else:
            if system.name == "oscillator":
                guards.kalman(system, n_particles)
            if system.name == "particle":
                guards.skeleton(system)
            guards.filter_noise(system, n_particles)
            guards.step_convergence(system, n_particles)
    t0 = time.time()
    chain, note = run_pmmh(system, collider, n_particles, **pmmh)
    return chain, C.PMMH_BURN, 1, note, {"wall_s": np.array(time.time() - t0)}


def run_surrogate(system, leg, cell, net_kw, nuts, bpinn_kw, results, n_eff, check):
    net = Surrogate(system, **net_kw)
    correct = leg == "chain_surrogate"
    model = BPINN(system, net, init_prior=bpinn_kw["init_prior"], barrier=bpinn_kw["barrier"],
                  correct_log_Z=correct)
    if correct and n_eff is None and system.residual == "force":
        twin = load_run(os.path.join(results, cell, "bpinn.npz"))
        if twin is None or twin.theta_last is None:
            raise SystemExit("chain_surrogate needs %s/bpinn.npz (or --n-eff)" % cell)
        n_eff = model.measure_n_eff(jax.numpy.asarray(twin.theta_last))
    model.n_eff = n_eff
    if correct:
        print("   n_eff = %s  (Nd = %d, dim theta = %d)"
              % (n_eff, net.N * system.d, net.n_weights), flush=True)
    if check:
        guards.normalized_residual(model)
    run_kw = {k: bpinn_kw[k] for k in ("map_init", "map_steps", "map_chunks",
                                       "map_key_offset", "mcmc_key_offset")}
    chain, thetas, note = model.run_nuts(**nuts, **run_kw)
    if check:
        guards.surrogate_fit(model, jax.numpy.asarray(thetas[-1]))
        if model.barrier:
            idx = np.linspace(0, len(chain) - 1, 20).astype(int)
            guards.barrier_inactive(model, [jax.numpy.asarray(thetas[i]) for i in idx],
                                    [jax.numpy.asarray(chain[i]) for i in idx])
    extra = dict(theta_last=thetas[-1])
    if correct:
        extra["n_eff"] = np.array(np.nan if n_eff is None else n_eff)
        note += "; n_eff %s" % n_eff
    return chain, 0.0, nuts["n_chains"], note, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", nargs="?")
    ap.add_argument("leg", nargs="?", choices=C.LEGS + C.PATH_NUTS_LEGS + C.KALMAN_LEGS)
    ap.add_argument("--seed", type=int, default=None,
                    help="sampler seed (default: that of the paper's main runs)")
    ap.add_argument("--list", action="store_true", help="print every run of the paper")
    ap.add_argument("--legs", default=None, help="with --list: comma-separated legs to keep")
    ap.add_argument("--groups", default=None,
                    help="with --list: comma-separated groups of %s" % (C.GROUPS,))
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for smoke tests")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--n-sub", type=int, default=None)
    ap.add_argument("--n-eff", type=float, default=None)
    ap.add_argument("--no-guards", action="store_true")
    a = ap.parse_args()

    if a.list:
        keep = a.legs.split(",") if a.legs else None
        groups = a.groups.split(",") if a.groups else C.GROUPS
        for cell, leg, seed in C.paper_runs(groups):
            if keep is None or leg in keep:
                print(cell, leg, "" if seed is None else "--seed %d" % seed)
        return 0
    if not (a.cell and a.leg):
        ap.error("give a cell and a leg, or --list")

    name, _, net_kw = C.cell_settings(a.cell)
    override = {"n_sub": a.n_sub} if a.n_sub else {}
    system = C.make_system(a.cell, **override)
    check = not a.no_guards
    print("=" * 78)
    print("%s / %s   on %s" % (a.cell, a.leg, jax.default_backend()))
    print("  " + system.describe())
    print("=" * 78, flush=True)

    config = dict(cell=a.cell, leg=a.leg, quick=a.quick, n_sub=system.n_sub,
                  data_seed=system.data_seed)
    seed = {} if a.seed is None else {"seed": a.seed}
    t0 = time.time()
    if a.leg in ("chain_ss", "collider_ss"):
        n_particles = C.QUICK_PARTICLES if a.quick else C.N_PARTICLES[name]
        pmmh = dict(C.QUICK_PMMH if a.quick else C.PMMH, **seed)
        config.update(pmmh, n_particles=n_particles)
        chain, burn, n_chains, note, extra = run_state_space(system, a.leg, n_particles, pmmh, check)
    elif a.leg in C.PATH_NUTS_LEGS:
        kw = dict(C.QUICK_PATH_NUTS if a.quick else C.PATH_NUTS, **seed)
        config.update(kw)
        chain, note = run_path_nuts(system, a.leg.startswith("collider"), **kw)
        burn, n_chains, extra = 0.0, 1, {}
    elif a.leg in C.KALMAN_LEGS:
        if name != "oscillator":
            ap.error("%s needs the oscillator's exact Kalman evidence" % a.leg)
        kw = dict(C.KALMAN, n_iter=400, n_adapt=100) if a.quick else dict(C.KALMAN)
        config.update(kw)
        chain, note = run_kalman_rwm(system, a.leg.startswith("collider"), **kw)
        burn, n_chains, extra = 0.0, kw["n_chains"], {}
    else:
        nuts = dict(C.QUICK_NUTS if a.quick else C.NUTS, **seed)
        bpinn_kw = dict(C.BPINN[name])
        if a.quick:
            bpinn_kw["map_steps"] = C.QUICK_MAP_STEPS
        config.update(nuts, **bpinn_kw, **net_kw)
        chain, burn, n_chains, note, extra = run_surrogate(
            system, a.leg, a.cell, net_kw, nuts, bpinn_kw, a.results, a.n_eff, check)

    extra.setdefault("wall_s", np.array(time.time() - t0))   # sampler only for PMMH
    save_run(os.path.join(a.results, a.cell, C.leg_file(a.leg, a.seed) + ".npz"), system, chain,
             burn, note, config, n_chains=n_chains, **extra)
    post = chain[int(burn * len(chain)):]
    for j, nm in enumerate(system.param_names):
        print("   %-8s mean %10.4g  sd %10.4g  (truth %g)"
              % (nm, post[:, j].mean(), post[:, j].std(), float(system.lam_true[j])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

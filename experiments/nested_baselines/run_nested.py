#!/usr/bin/env python
"""NSVI or NPSGLD (Hao & Bilionis) on the chain model's Euler--Maruyama latent.

    python experiments/nested_baselines/run_nested.py --cell oscillator-force --method nsvi
    python experiments/nested_baselines/run_nested.py --sweep 3     # line 3 of sweeps.txt

Uses the engines of niff-replication (https://github.com/BoltMaxwell/niff-replication,
commit dab4403), which must be importable.  Writes results/nested/<tag>.npz/.json.
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments"), HERE]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from niff.npsgld import NPSGLDConfig, run_npsgld  # noqa: E402
from niff.nsvi import NSVIConfig, draw_nsvi_samples, run_nsvi  # noqa: E402

import configs as C  # noqa: E402
import markov_problem  # noqa: E402
from colliderbias.diagnostics import split_rhat_ess  # noqa: E402
from colliderbias.paths import RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "nested")


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=None, help="run line N of sweeps.txt")
    ap.add_argument("--cell", default="oscillator-force")
    ap.add_argument("--method", choices=["nsvi", "npsgld"], default="nsvi")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-iters", type=int, default=8000)
    ap.add_argument("--warmup-lr", type=float, default=3e-3)
    # NSVI
    ap.add_argument("--iterations", type=int, default=50_000)
    ap.add_argument("--inner-iterations", type=int, default=8)
    ap.add_argument("--outer-lr", type=float, default=2e-3)
    ap.add_argument("--inner-lr", type=float, default=3e-3)
    ap.add_argument("--n-aux-samples", type=int, default=8)
    ap.add_argument("--partition-anneal", type=int, default=0)
    ap.add_argument("--theta-guide", choices=["diag", "full_rank"], default="full_rank")
    ap.add_argument("--init-std", type=float, default=3e-6)
    ap.add_argument("--inner-init-std", type=float, default=3e-6)
    ap.add_argument("--nsvi-draws", type=int, default=20_000)
    # NPSGLD
    ap.add_argument("--npsgld-iterations", type=int, default=200_000)
    ap.add_argument("--npsgld-chains", type=int, default=4)
    ap.add_argument("--npsgld-thinning", type=int, default=10)
    ap.add_argument("--step-size", type=float, default=2e-6)
    ap.add_argument("--step-size-final", type=float, default=None)
    ap.add_argument("--aux-step-size", type=float, default=2e-6)
    ap.add_argument("--aux-iterations", type=int, default=5)
    ap.add_argument("--npsgld-init-std", type=float, default=0.05)
    ap.add_argument("--preconditioner", default="rmsprop",
                    choices=["rmsprop", "identity", "diag_fisher", "dense_fisher"])
    a = ap.parse_args(argv)
    if a.sweep is not None:
        with open(os.path.join(HERE, "sweeps.txt")) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")]
        a = ap.parse_args(shlex.split(lines[a.sweep]))
    return a


def main():
    a = parse(sys.argv[1:])
    tag = a.tag or "%s_%s" % (a.cell, a.method)
    system = C.make_system(a.cell)
    print("%s  %s  on %s" % (tag, system.describe(), jax.default_backend()), flush=True)
    t0 = time.time()
    prob = markov_problem.build(system, a.warmup_iters, a.warmup_lr)
    setup_s = time.time() - t0
    key = jax.random.PRNGKey(a.seed)
    t1 = time.time()
    if a.method == "nsvi":
        cfg = NSVIConfig(iterations=a.iterations, inner_iterations=a.inner_iterations,
                         outer_lr=a.outer_lr, inner_lr=a.inner_lr, n_outer_samples=1,
                         n_aux_samples=a.n_aux_samples, theta_guide=a.theta_guide,
                         init_mean=prob.init_mean, init_std=a.init_std,
                         inner_init_std=a.inner_init_std, theta_prior_std=1.0, x0_prior_std=1.0,
                         partition_weight=1.0, partition_anneal_steps=a.partition_anneal,
                         min_std=1e-9, grad_clip=1e3, log_every=max(1, a.iterations // 20))
        res = run_nsvi(key, d_w=prob.d_w, d_theta=prob.d_theta, d_x0=prob.d_x0,
                       log_likelihood_fn=prob.log_likelihood_fn, energy_fn=prob.energy_fn, config=cfg)
        jax.block_until_ready(res.params["theta_mean"])
        engine_s = time.time() - t1
        u = draw_nsvi_samples(jax.random.PRNGKey(a.seed + 1), res.params, n_samples=a.nsvi_draws,
                              d_w=prob.d_w, d_theta=prob.d_theta, d_x0=prob.d_x0,
                              config=cfg)["theta_samples"]
        chain_id = None
        hist = np.asarray(res.history["mean"])[:, prob.d_w:prob.d_w + prob.d_theta]
        trace = np.asarray(jax.vmap(prob.to_physical)(jnp.asarray(hist)))
    else:
        cfg = NPSGLDConfig(iterations=a.npsgld_iterations, chains=a.npsgld_chains,
                           burn_in=a.npsgld_iterations // 2, thinning=a.npsgld_thinning,
                           step_size=a.step_size, step_size_final=a.step_size_final,
                           aux_step_size=a.aux_step_size, aux_iterations=a.aux_iterations,
                           theta_prior_std=1.0, x0_prior_std=1.0, init_mean=prob.init_mean,
                           init_std=a.npsgld_init_std, preconditioner=a.preconditioner,
                           include_riemannian_correction=False,
                           trace_every=max(1, a.npsgld_iterations // 20))
        res = run_npsgld(key, d_w=prob.d_w, d_theta=prob.d_theta, d_x0=prob.d_x0,
                         log_likelihood_fn=prob.log_likelihood_fn, energy_fn=prob.energy_fn, config=cfg)
        engine_s = time.time() - t1
        u = res.samples["theta_samples"]
        chain_id = np.asarray(res.samples["chain_id"])
        trace = np.zeros((0, prob.d_theta))

    lam = np.asarray(jax.vmap(prob.to_physical)(jnp.asarray(u)))
    keep = np.isfinite(lam).all(axis=1)
    lam = lam[keep]
    summary = dict(tag=tag, cell=a.cell, method=a.method, argv=" ".join(sys.argv[1:]),
                   setup_s=setup_s, engine_s=engine_s, n_draws=int(len(lam)),
                   n_nonfinite=int((~keep).sum()),
                   param_names=system.param_names, mean=lam.mean(0).tolist(), sd=lam.std(0).tolist())
    if chain_id is not None:
        chain_id = chain_id[keep]
        n_ch = int(chain_id.max()) + 1
        m = min(int((chain_id == c).sum()) for c in range(n_ch))
        stacked = np.concatenate([lam[chain_id == c][:m] for c in range(n_ch)])
        rhat, ess = split_rhat_ess(stacked, n_ch)
        summary.update(rhat=rhat.tolist(), ess=ess.tolist(), ess_per_s=float(np.nanmin(ess) / engine_s))
    print(json.dumps(summary, indent=1))
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, tag + ".npz"), lam=lam, u=np.asarray(u), guide_trace=trace,
             chain_id=np.array([]) if chain_id is None else chain_id)
    with open(os.path.join(OUT, tag + ".json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()

"""Every run in the paper, as (cell, leg).

A cell fixes system, residual metric and dataset (sigma_r is fixed at its true
value throughout); a leg fixes model, latent and sampler:

    chain_ss         chain model,    state-space latent, particle filter + PMMH
    collider_ss      collider model, state-space latent, particle filter + PMMH
    bpinn            collider model, Fourier-feature MLP, NUTS
    chain_surrogate  chain model on the B-PINN's latent (oscillator only), NUTS
    chain_ss_nuts, collider_ss_nuts
                     state-space latent under joint NUTS (estimator check)
    chain_kalman, collider_kalman
                     state-space latent with the exact Kalman likelihood (oscillator),
                     adaptive random-walk Metropolis

A cell name may carry suffixes: -T<horizon> (the horizon ladder) and -d<seed>
(another dataset), e.g. oscillator-accel-d3.  A run with a sampler seed
other than the default is stored as <leg>-seed<k>.
"""
from colliderbias.systems import SYSTEMS

SYSTEM = {
    "oscillator": dict(horizon=24.0, K=40, n_sub=32, data_seed=2),
    "pendulum": dict(horizon=12.0, K=20, n_sub=65, data_seed=0),
    "particle": dict(horizon=12.0, K=20, n_sub=640, data_seed=0),
}
N_PARTICLES = {"oscillator": 1000, "pendulum": 1000, "particle": 2000}
SURROGATE = {
    "oscillator": dict(n_fourier=16, n_colloc=128),
    "pendulum": dict(n_fourier=16, n_colloc=360),
    "particle": dict(n_fourier=16, n_colloc=300),
}
# MAP warm start and PRNG conventions of the published runs.
BPINN = {
    "oscillator": dict(init_prior=False, barrier=False, map_init="small", map_steps=12000,
                       map_chunks=6, map_key_offset=100, mcmc_key_offset=0),
    "pendulum": dict(init_prior=True, barrier=False, map_init="lecun", map_steps=20000,
                     map_chunks=8, map_key_offset=100, mcmc_key_offset=0),
    "particle": dict(init_prior=True, barrier=True, map_init="lecun", map_steps=20000,
                     map_chunks=8, map_key_offset=0, mcmc_key_offset=1),
}
PMMH = dict(n_iter=24000, n_adapt=4000, seed=0)
PMMH_BURN = 0.3
NUTS = dict(n_chains=4, n_warmup=3000, n_samples=6000, max_tree_depth=11, seed=0)
PATH_NUTS = dict(n_warmup=400, n_samples=1600, max_tree_depth=8, seed=0)

QUICK_PMMH = dict(n_iter=60, n_adapt=30, seed=0)
QUICK_NUTS = dict(n_chains=1, n_warmup=20, n_samples=20, max_tree_depth=4, seed=0)
QUICK_PATH_NUTS = dict(n_warmup=10, n_samples=10, max_tree_depth=4, seed=0)
QUICK_MAP_STEPS = 240
QUICK_PARTICLES = 100

# Exact Kalman-likelihood sampler for the oscillator's state space (chain_kalman,
# collider_kalman): 4 chains, the first quarter of each adapts the proposal.
KALMAN = dict(n_chains=4, n_iter=40000, n_adapt=10000)

# Fig. 3: the oscillator at three horizons with the step size and the
# surrogate's bandwidth held fixed (T = 24 is the main oscillator cell).
LADDER = {12.0: dict(K=20, n_fourier=8, n_colloc=64),
          48.0: dict(K=79, n_fourier=32, n_colloc=256)}

# Replication over datasets (Fig. 2 configuration); data seed 2 is the main cell.
REPLICATION_CELL = "oscillator-accel"
REPLICATION_SEEDS = tuple(range(10))
# Extra PMMH chains (sampler seeds 1-3; seed 0 is the main run) for the
# references of the nested-inference comparison.
MULTICHAIN_CELLS = ("oscillator-force", "oscillator-accel", "pendulum-force")
MULTICHAIN_SEEDS = (1, 2, 3)

LEGS = ("chain_ss", "collider_ss", "bpinn", "chain_surrogate")
PATH_NUTS_LEGS = ("chain_ss_nuts", "collider_ss_nuts")
KALMAN_LEGS = ("chain_kalman", "collider_kalman")


def cell_name(system, residual, horizon=None, data_seed=None):
    name = "%s-%s" % (system, residual)
    if horizon is not None:
        name += "-T%d" % horizon
    if data_seed is not None and data_seed != SYSTEM[system]["data_seed"]:
        name += "-d%d" % data_seed
    return name


def leg_file(leg, seed=None):
    return leg if seed is None else "%s-seed%d" % (leg, seed)


def parse_cell(name):
    """(system, residual, horizon or None, data_seed or None)."""
    parts = name.split("-")
    system, residual = parts[:2]
    horizon = data_seed = None
    for p in parts[2:]:
        if p[0] == "T":
            horizon = float(p[1:])
        elif p[0] == "d":
            data_seed = int(p[1:])
        else:
            raise ValueError("unknown cell suffix %r in %s" % (p, name))
    return system, residual, horizon, data_seed


def cell_settings(name):
    """System kwargs and surrogate kwargs of one cell."""
    system, residual, horizon, data_seed = parse_cell(name)
    sys_kw = dict(SYSTEM[system], residual=residual)
    if data_seed is not None:
        sys_kw["data_seed"] = data_seed
    net_kw = dict(SURROGATE[system])
    if horizon is not None:
        lad = LADDER[horizon]
        sys_kw.update(horizon=horizon, K=lad["K"])
        net_kw.update(n_fourier=lad["n_fourier"], n_colloc=lad["n_colloc"])
    return system, sys_kw, net_kw


def make_system(name, **override):
    system, sys_kw, _ = cell_settings(name)
    sys_kw.update(override)
    return SYSTEMS[system](**sys_kw)


GROUPS = ("grid", "ladder", "kalman", "estimator", "multichain", "replication")


def paper_runs(groups=GROUPS):
    """All (cell, leg, sampler seed or None) behind the paper, in dependency order.

        grid         3 systems x 2 residuals (main and appendix tables, Fig. 2)
        ladder       oscillator horizons 12 and 48 (Fig. 3)
        kalman       exact state-space posteriors of the oscillator, force residual
        estimator    joint NUTS on the state-space latent (pendulum, particle)
        multichain   PMMH sampler seeds 1-3 of the nested-inference references
        replication  Fig. 2 configuration on data seeds 0-9 (seed 2 is the grid cell)
    """
    runs = []
    if "grid" in groups:
        for system in SYSTEM:
            for residual in ("accel", "force"):
                cell = cell_name(system, residual)
                runs += [(cell, "chain_ss", None), (cell, "collider_ss", None), (cell, "bpinn", None)]
                if system == "oscillator":
                    runs.append((cell, "chain_surrogate", None))
    if "ladder" in groups:
        for horizon in LADDER:
            cell = cell_name("oscillator", "accel", horizon)
            runs += [(cell, leg, None) for leg in LEGS]
    if "kalman" in groups:
        runs += [("oscillator-force", leg, None) for leg in KALMAN_LEGS]
    if "estimator" in groups:
        for system in ("pendulum", "particle"):
            cell = cell_name(system, "force")
            runs += [(cell, leg, None) for leg in PATH_NUTS_LEGS]
    if "multichain" in groups:
        runs += [(cell, "chain_ss", seed) for cell in MULTICHAIN_CELLS for seed in MULTICHAIN_SEEDS]
    if "replication" in groups:
        system, residual, _, _ = parse_cell(REPLICATION_CELL)
        for seed in REPLICATION_SEEDS:
            if seed != SYSTEM[system]["data_seed"]:
                cell = cell_name(system, residual, data_seed=seed)
                runs += [(cell, "chain_ss", None), (cell, "bpinn", None)]
    return runs

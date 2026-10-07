"""Storage of one inference run: chain, burn-in, truth, prior and configuration."""
import json
import os

import numpy as np


def save_run(path, system, chain, burn, note, config, n_chains=1, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, chain=np.asarray(chain), burn=np.array(burn), n_chains=np.array(n_chains),
        param_names=np.array(system.param_names), lam_true=np.asarray(system.lam_true),
        prior_median=np.asarray(system.prior_median), prior_sd=np.asarray(system.prior_sd),
        note=np.array(note), config=np.array(json.dumps(config)), **extra)
    print("saved", path, flush=True)


class Run:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.path = path
        self.chain = np.asarray(d["chain"], float)
        self.burn = float(d["burn"])
        self.n_chains = int(d["n_chains"])
        self.param_names = [str(s) for s in d["param_names"]]
        self.lam_true = np.asarray(d["lam_true"], float)
        self.prior_sd = np.asarray(d["prior_sd"], float)
        self.note = str(d["note"])
        self.config = json.loads(str(d["config"]))
        self.theta_last = np.asarray(d["theta_last"]) if "theta_last" in d.files else None
        self.wall_s = float(d["wall_s"]) if "wall_s" in d.files else float("nan")

    @property
    def draws(self):
        return self.chain[int(self.burn * len(self.chain)):]

    def __getitem__(self, name):
        return self.draws[:, self.param_names.index(name)]


def load_run(path):
    return Run(path) if os.path.exists(path) else None

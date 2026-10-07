#!/usr/bin/env python
"""Replication of the Fig. 2 configuration over datasets (appendix table), to results/tables.

    top     Chain (state space, PMMH) and the B-PINN (NUTS) on data seeds 0-9
            (`python experiments/run.py --list --groups replication`; seed 2 is the
            main cell).  Datasets on which the B-PINN has not converged are left out.
    bottom  exact posteriors on a linear surrogate, results/replication/exact.json
            (experiments/replication_exact.py)

Displacement: posterior mean of the log quantity minus its log truth, averaged
over datasets, +- its standard error; sd: mean posterior sd; cov.: datasets
whose central 90% credible interval contains the truth.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

from colliderbias.diagnostics import RHAT_OK, split_rhat_ess  # noqa: E402
from colliderbias.paths import RESULTS, RUNS  # noqa: E402
from colliderbias.runs import load_run  # noqa: E402
import configs as C  # noqa: E402

OUT = os.path.join(RESULTS, "tables")
QUANTITIES = (("log(c/m)", lambda r: r["c"] / r["m"]), ("log(k/m)", lambda r: r["k"] / r["m"]))
LEGS = (("chain_ss", "Chain (state space)"), ("bpinn", "Collider B-PINN (MLP)"))
EXACT = (("chain_ss", "Chain (state space)"), ("chain_linear", "Chain (linear surrogate)"),
         ("collider_linear", "Collider (linear surrogate)"))


def sampled():
    """{leg: {seed: (displacements, sds, covered, max R-hat)}}, one entry per quantity."""
    system, residual, _, _ = C.parse_cell(C.REPLICATION_CELL)
    out = {leg: {} for leg, _ in LEGS}
    for seed in C.REPLICATION_SEEDS:
        cell = C.cell_name(system, residual, data_seed=seed)
        for leg, _ in LEGS:
            r = load_run(os.path.join(RUNS, cell, leg + ".npz"))
            if r is None:
                continue
            truth = dict(zip(r.param_names, r.lam_true))
            disp, sd, cov, rhat = [], [], [], 0.0
            for _, f in QUANTITIES:
                u, u0 = np.log(f(r)), np.log(f(truth))
                lo, hi = np.quantile(u, [0.05, 0.95])
                disp.append(u.mean() - u0)
                sd.append(u.std())
                cov.append(lo <= u0 <= hi)
                if r.n_chains > 1:
                    rhat = max(rhat, float(split_rhat_ess(f(r)[:, None], r.n_chains)[0][0]))
            out[leg][seed] = (np.array(disp), np.array(sd), np.array(cov), rhat)
    return out


def exact():
    path = os.path.join(RESULTS, "replication", "exact.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    t = d["truth"]
    truth = np.log([t["c_m"], t["k_m"]])
    out = {"n_sub": d["n_sub"]}
    for row in d["rows"]:
        for leg, r in row["res"].items():
            out.setdefault(row["data_sub"], {}).setdefault(leg, {})[row["seed"]] = (
                np.array(r["mean"]) - truth, np.array(r["sd"]), np.array(r["covered"]), r["edge"])
    return out


def row(label, per_seed, bold=False):
    D = np.array([v[0] for v in per_seed.values()])
    S = np.array([v[1] for v in per_seed.values()])
    cov = np.array([v[2] for v in per_seed.values()]).sum(0)
    n = len(D)
    se = D.std(0, ddof=1) / np.sqrt(n)
    cells = []
    for i in range(len(QUANTITIES)):
        b = bold and abs(D[:, i].mean()) > 2 * se[i]
        disp = "%+.2f \\pm %.2f" % (D[:, i].mean(), se[i])
        cells += ["$\\mathbf{%s}$" % disp if b else "$%s$" % disp, "$%.2f$" % S[:, i].mean(),
                  ("$\\mathbf{%d/%d}$" if b else "$%d/%d$") % (cov[i], n)]
    return "%s & %s \\\\" % (label, " & ".join(cells))


def main():
    lines, log = [], []
    S = sampled()
    converged = sorted(s for s, v in S["bpinn"].items() if v[3] <= RHAT_OK and s in S["chain_ss"])
    excluded = sorted(set(S["bpinn"]) - set(converged))
    if converged:
        lines.append(r"\multicolumn{7}{@{}l}{\textit{Sampled posteriors, %d datasets}} \\" % len(converged))
        for leg, label in LEGS:
            lines.append(row(label, {s: S[leg][s] for s in converged}, bold=leg == "bpinn"))
        diff = np.array([S["bpinn"][s][0] - S["chain_ss"][s][0] for s in converged])
        log.append("B-PINN minus chain per dataset: " + ", ".join(
            "%s %+.2f +- %.2f" % (q, diff[:, i].mean(), diff[:, i].std(ddof=1) / np.sqrt(len(diff)))
            for i, (q, _) in enumerate(QUANTITIES)))
        all_b = {s: v[0][0] for s, v in S["bpinn"].items()}
        main_seed = C.SYSTEM["oscillator"]["data_seed"]
        log.append("B-PINN displacement of log(c/m): main dataset %+.2f, median over converged %+.2f"
                   % (all_b.get(main_seed, np.nan), np.median([all_b[s] for s in converged])))
        for s in excluded:
            log.append("excluded data seed %d: B-PINN max split R-hat %.3f, displacement of log(c/m) %+.2f"
                       % (s, S["bpinn"][s][3], all_b[s]))
        below = [all(S["bpinn"][s][0][i] < 0 for s in converged) for i in range(len(QUANTITIES))]
        log.append("B-PINN below the truth on every dataset: %s" % below)
    E = exact()
    if E is not None:
        lines.append(r"\midrule")
        n_sub = E.pop("n_sub")
        main_sub = min(E)
        lines.append(r"\multicolumn{7}{@{}l}{\textit{Exact posteriors on a linear surrogate, %d datasets}} \\"
                     % len(E[main_sub]["chain_ss"]))
        for leg, label in EXACT:
            lines.append(row(label, E[main_sub][leg], bold=leg == "collider_linear"))
            log.append("exact %-16s largest posterior mass on the grid boundary %.1e"
                       % (leg, max(v[3] for v in E[main_sub][leg].values())))
        for sub in sorted(E)[1:]:
            D = np.array([v[0] for v in E[sub]["chain_ss"].values()])
            cov = np.array([v[2] for v in E[sub]["chain_ss"].values()]).sum(0)
            log.append("data simulated %dx finer than the inference grid: chain (state space) " % (sub // n_sub)
                       + ", ".join("%s %+.2f +- %.2f (90%% cov %d/%d)"
                                   % (q, D[:, i].mean(), D[:, i].std(ddof=1) / np.sqrt(len(D)), cov[i], len(D))
                                   for i, (q, _) in enumerate(QUANTITIES)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "replication.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUT, "replication.tex"))
    print("\n".join(log))


if __name__ == "__main__":
    main()

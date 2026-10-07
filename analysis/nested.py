#!/usr/bin/env python
"""Nested inference (NSVI, NPSGLD) against PF + PMMH (appendix table), to results/tables/nested.tex.

Reads results/nested/<tag>.{npz,json} (experiments/nested_baselines/run_nested.py)
and the references, the chain_ss runs of oscillator-{force,accel} and
pendulum-force, plus their extra sampler seeds (chain_ss-seed<k>).

Every run is scored on the drift-invariant scale (m, or m1 + m2):

    location  posterior mean if R-hat <= 1.05, else the median (dagger)
    width     posterior sd / the reference's
    cost      engine wall clock / ESS, in seconds per effective sample

R-hat and ESS of NPSGLD are over the run's own chains, on the scale itself; for
the reference the ESS is the smallest over the components of lambda.  NSVI draws
i.i.d. from a fitted guide, so R-hat and ESS do not apply to it.  A run has
diverged if it produced non-finite draws, if its location is off by more than a
factor of 100, or if its spread exceeds 100 reference standard deviations.
"""
import glob
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

NESTED = os.path.join(RESULTS, "nested")
OUT = os.path.join(RESULTS, "tables")
CELLS = (("oscillator-force", "Oscillator", "force residuals", ["m"]),
         ("oscillator-accel", "Oscillator", r"accel.\ residuals", ["m"]),
         ("pendulum-force", "Double pendulum", "torque residuals", ["m1", "m2"]))


def rhat_ess_chains(x, chain_id):
    """R-hat and ESS of one scalar series over a run's own chains (not split)."""
    ch = [x[chain_id == c] for c in np.unique(chain_id)]
    n = min(len(c) for c in ch)
    if n < 4:
        return np.nan, np.nan
    ch = np.asarray([c[:n] for c in ch])
    m = len(ch)
    W = ch.var(axis=1, ddof=1).mean()
    B = n * ch.mean(axis=1).var(ddof=1)
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan
    V = (n - 1) / n * W + B / n
    ess = float(min(m * n * V / (B / (m - 1) * m), m * n)) if B > 0 else np.nan
    return float(np.sqrt(V / W)), ess


def reference(cell, scale):
    run = load_run(os.path.join(RUNS, cell, "chain_ss.npz"))
    if run is None:
        return None
    s = sum(run[n] for n in scale)
    ess = float(np.nanmin(split_rhat_ess(run.draws, run.n_chains)[1]))
    truth = sum(dict(zip(run.param_names, run.lam_true))[n] for n in scale)
    return dict(mean=float(s.mean()), sd=float(s.std()), ess=ess, wall=run.wall_s, truth=truth,
                ratio=run["c"] / run["m"] if "c" in run.param_names else None)


def nested_runs(cell, scale, ref):
    rows = []
    for jf in sorted(glob.glob(os.path.join(NESTED, "*.json"))):
        with open(jf) as f:
            meta = json.load(f)
        if meta["cell"] != cell:
            continue
        d = np.load(jf[:-5] + ".npz")
        names = meta["param_names"]
        lam = d["lam"]
        s = sum(lam[:, names.index(n)] for n in scale)
        sd = float(s.std())
        diverged = (meta.get("n_nonfinite", 0) > 0 or np.median(np.abs(s)) > 100.0 * ref["truth"]
                    or sd > 100.0 * ref["sd"])
        if meta["method"] == "nsvi":
            rhat, ess = np.nan, float(len(s))
        else:
            rhat, ess = rhat_ess_chains(s, d["chain_id"])
        ok = meta["method"] == "nsvi" or rhat <= RHAT_OK
        c_m = (lam[:, names.index("c")] / lam[:, names.index("m")]).mean() if "c" in names else np.nan
        rows.append(dict(tag=meta["tag"], method=meta["method"], wall=meta["engine_s"], sd=sd,
                         loc=float(s.mean() if ok and not diverged else np.median(s)), ok=ok and not diverged,
                         diverged=diverged, rhat=rhat, ess=ess, width=sd / ref["sd"], c_m=c_m))
    return rows


def span(values, fmt):
    lo, hi = fmt % min(values), fmt % max(values)
    return "$%s$" % lo if lo == hi else "$%s$--$%s$" % (lo, hi)


def multichain(cell, scale):
    """R-hat over sampler seeds 0-3 and the spread of their means of the scale."""
    runs = [load_run(os.path.join(RUNS, cell, C.leg_file("chain_ss", s) + ".npz"))
            for s in (None,) + C.MULTICHAIN_SEEDS]
    runs = [r for r in runs if r is not None]
    if len(runs) < 2:
        return
    n = min(len(r.draws) for r in runs)
    rhat = split_rhat_ess(np.concatenate([r.draws[:n] for r in runs]), len(runs))[0]
    means = [sum(r[k] for k in scale).mean() for r in runs]
    print("%-24s %d PMMH chains: max split R-hat %.4f, means of the scale %s (spread %.3f)"
          % (cell, len(runs), np.nanmax(rhat), " ".join("%.4f" % m for m in means), np.ptp(means)))


def main():
    lines = []
    for cell, system, label, scale in CELLS:
        multichain(cell, scale)
        ref = reference(cell, scale)
        if ref is None:
            print("%s: no reference run" % cell)
            continue
        cost = ref["wall"] / ref["ess"]
        print("\n%s  reference %.4f +- %.4f  ESS %.0f  %.0f s  ->  %.2f s per effective sample%s"
              % (cell, ref["mean"], ref["sd"], ref["ess"], ref["wall"], cost,
                 "" if ref["ratio"] is None else "   c/m %.3f +- %.3f" % (ref["ratio"].mean(), ref["ratio"].std())))
        rows = nested_runs(cell, scale, ref)
        for r in sorted(rows, key=lambda r: (r["method"], r["tag"])):
            if r["diverged"]:
                print("   %-16s %-7s %8.0f s  diverged" % (r["tag"], r["method"], r["wall"]))
                continue
            print("   %-16s %-7s %8.0f s  location %8.4f%s  R-hat %6s  ESS %6.0f  width %5.2f  %s  c/m %.3f"
                  % (r["tag"], r["method"], r["wall"], r["loc"], "" if r["ok"] else " (median)",
                     "%.2f" % r["rhat"] if r["rhat"] == r["rhat"] else "n/a", r["ess"], r["width"],
                     "" if r["method"] == "nsvi" else "%7.1f s/ESS (%.0fx)" % (r["wall"] / r["ess"],
                                                                            r["wall"] / r["ess"] / cost),
                     r["c_m"]))
        lines.append(r"%s & PF\,+\,PMMH (ours) & $1$ & $%.3f \pm %.3f$ & $1.00$ & $%.0f$ & $1.00$ & $\mathbf{%.2f}$ \\"
                     % (system, ref["mean"], ref["sd"], ref["ess"], cost))
        for method, name in (("nsvi", "NSVI"), ("npsgld", "NPSGLD")):
            sel = [r for r in rows if r["method"] == method]
            if not sel:
                continue
            fine = [r for r in sel if not r["diverged"]]
            n_div = len(sel) - len(fine)
            loc = span([r["loc"] for r in fine], "%.2f") if fine else "---"
            if fine and not all(r["ok"] for r in fine):
                loc += r"\,$\dagger$"
            if n_div:
                loc += ", $%d$ diverged" % n_div
            cells = [label if method == "nsvi" else "", name, "$%d$" % len(sel), loc]
            if method == "nsvi":
                cells += ["---", "---", span([r["width"] for r in fine], "%.2f"), "---"]
            else:
                cells += [span([r["rhat"] for r in fine], "%.1f"), span([r["ess"] for r in fine], "%.0f"),
                          span([r["width"] for r in fine], "%.2f"),
                          span([r["wall"] / r["ess"] for r in fine], "%.0f")]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "nested.tex"), "w") as f:
        f.write("\n".join(lines[:-1]) + "\n")
    print("\nwrote", os.path.join(OUT, "nested.tex"))


if __name__ == "__main__":
    main()

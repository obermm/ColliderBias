#!/usr/bin/env python
"""All result tables of the paper, from results/runs, written to results/tables.

    grid.tex               per-system posterior tables (appendix)
    scorecard.tex          main grid table: shift t along the drift-invariant direction
                           and the identified quantity of each system
    system_settings.tex    per-system settings
    surrogate_chain.tex    what restoring -log Z_lambda on the B-PINN latent removes
    summary.txt            posterior mean, sd and R-hat of every run

and prints the numbers the text quotes next to these tables (exact Kalman
posteriors of the oscillator, joint NUTS against PMMH on the state-space latent).
A run that is not a posterior (R-hat > 1.05, or fewer than 25 distinct draws)
is reported by its median and marked with a dagger; a double dagger marks the
oscillator's state-space collider sampled on the exact Kalman likelihood.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]

from colliderbias import guards  # noqa: E402
from colliderbias.diagnostics import ess_autocorr, is_posterior, split_rhat_ess  # noqa: E402
from colliderbias.paths import RESULTS, RUNS  # noqa: E402
from colliderbias.runs import load_run  # noqa: E402
import configs as C  # noqa: E402
from configs import cell_name  # noqa: E402

OUT = os.path.join(RESULTS, "tables")
SYSTEMS = ("oscillator", "pendulum", "particle")
LEG_LABEL = {"chain_ss": "Chain (state space)", "collider_ss": "Collider (state space)",
             "chain_surrogate": "Chain (Fourier MLP)", "bpinn": "Collider B-PINN (Fourier MLP)"}
SCALE = {"oscillator": ["m"], "pendulum": ["m1", "m2"], "particle": ["m0"]}
# components rescaled along the drift-invariant direction
RAY = {"oscillator": ["m", "c", "k"], "pendulum": ["m1", "m2"], "particle": ["m0", "q"]}
# the identified physical quantity of the main grid table
IDENT = {"oscillator": (r"$c/m$", lambda r: r["c"] / r["m"]),
         "pendulum": (r"$m_2/m_1$", lambda r: r["m2"] / r["m1"]),
         "particle": (r"$c$", lambda r: r["c"])}
# every identified combination (log space), for the chain's worst standardized error
IDENTIFIED = {"oscillator": [lambda r: r["c"] / r["m"], lambda r: r["k"] / r["m"]],
              "pendulum": [lambda r: r["m2"] / r["m1"], lambda r: r["l1"], lambda r: r["l2"]],
              "particle": [lambda r: r["q"] / r["m0"], lambda r: r["c"]]}
DERIVED = {
    "oscillator": [(r"$c/m$", lambda r: r["c"] / r["m"]), (r"$k/m$", lambda r: r["k"] / r["m"])],
    "pendulum": [(r"$m_1{+}m_2$", lambda r: r["m1"] + r["m2"]), (r"$m_2/m_1$", lambda r: r["m2"] / r["m1"])],
    "particle": [(r"$q/m_0$", lambda r: r["q"] / r["m0"])],
}
TEX = {"m": "$m$", "c": "$c$", "k": "$k$", "m1": "$m_1$", "m2": "$m_2$", "l1": r"$\ell_1$",
       "l2": r"$\ell_2$", "m0": "$m_0$", "q": "$q$"}


class Draws(dict):
    """Posterior draws by parameter name, plus truth and verdict."""

    def __init__(self, run):
        super().__init__({nm: run[nm] for nm in run.param_names})
        self.truth = dict(zip(run.param_names, run.lam_true))
        self.prior_sd = dict(zip(run.param_names, run.prior_sd))
        self.n_chains = run.n_chains
        self.matrix = run.draws
        self.ok = is_posterior(run.draws, run.n_chains)
        self.rhat = split_rhat_ess(run.draws, run.n_chains)[0]


def load(system, residual, leg, horizon=None):
    run = load_run(os.path.join(RUNS, cell_name(system, residual, horizon), leg + ".npz"))
    return None if run is None else Draws(run)


def load_collider_ss(system, residual):
    """The state-space collider; on the oscillator's force residual its exact posterior (Kalman)."""
    if system == "oscillator" and residual == "force":
        D = load(system, residual, "collider_kalman")
        if D is not None:
            return D, True
    return load(system, residual, "collider_ss"), False


def ray_shift(D, system):
    """t = precision-weighted mean of log(lam_j / median_j) over the drift-invariant components."""
    w = np.array([D.prior_sd[n] ** -2.0 for n in RAY[system]])
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(D[n] / D.truth[n]) for n in RAY[system]], axis=1)
    return logs @ w / w.sum()


def num(x, sig=4):
    if x == 0 or not np.isfinite(x):
        return "%g" % x
    a = abs(x)
    if 1e-3 <= a < 1e4:
        return "%.*f" % (min(max(0, sig - 1 - int(np.floor(np.log10(a)))), 5), x)
    e = int(np.floor(np.log10(a)))
    return r"%.1f{\cdot}10^{%d}" % (x / 10.0 ** e, e)


def cell_text(v, ok):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "---"
    pos = v[v > 0]
    if pos.size == 0 or np.median(pos) < 1e-300:
        return r"underflow\,$\dagger$"
    if not ok or pos.max() / pos.min() > 1e6:
        return r"$%s$\,$\dagger$" % num(float(np.median(v)))
    return r"$%s \pm %s$" % (num(float(v.mean())), num(float(v.std()), 2))


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text + "\n")
    print("wrote", os.path.join(OUT, name))


# ---- per-system grid tables ------------------------------------------------------
def grid_tables():
    out = []
    for system in SYSTEMS:
        rows, head = [], None
        for residual, rlabel in (("accel", "acceleration"), ("force", "force / torque")):
            first = True
            for leg in ("chain_ss", "collider_ss", "chain_surrogate", "bpinn"):
                exact = False
                if leg == "collider_ss":
                    D, exact = load_collider_ss(system, residual)
                else:
                    D = load(system, residual, leg)
                if D is None:
                    continue
                names = list(D.keys())
                if head is None:
                    head = ([r"\textbf{Residual}", r"\textbf{Model (latent)}"]
                            + ["%s (%s)" % (TEX[n], num(float(D.truth[n]), 3)) for n in names]
                            + ["%s (%s)" % (t, num(float(f(D.truth)), 3)) for t, f in DERIVED[system]])
                mark = r"\,$\ddagger$" if exact else ""
                cells = [cell_text(D[n], D.ok) + mark for n in names]
                cells += [cell_text(f(D), D.ok) + mark for _, f in DERIVED[system]]
                rows.append(" & ".join([rlabel if first else "", LEG_LABEL[leg]] + cells) + r" \\")
                first = False
            rows.append(r"\addlinespace")
        if head is None:
            continue
        out += [r"%% %s" % system, r"\begin{tabular}{@{}ll%s@{}}" % ("r" * (len(head) - 2)),
                r"\toprule", " & ".join(head) + r" \\", r"\midrule"] + rows[:-1] + [
                r"\bottomrule", r"\end{tabular}", ""]
    write("grid.tex", "\n".join(out))


# ---- main grid table ------------------------------------------------------------
def scorecard():
    """t and the identified quantity (posterior mean / truth).  Bold: off by more than a factor of two."""
    def ratio(v):
        if not np.isfinite(v):
            return "overflow"
        if 0.1 <= abs(v) < 10:
            text = "%.2f" % v
        elif 10 <= abs(v) < 1e3:
            text = "%.0f" % v
        else:
            e = int(np.floor(np.log10(abs(v))))
            text = r"%.1f{\cdot}10^{%d}" % (v / 10.0 ** e, e)
        return (r"$\mathbf{%s}$" if not 0.5 <= v <= 2.0 else "$%s$") % text

    lines, log = [], []
    for residual, rlabel in (("accel", "Acceleration residuals"), ("force", "Force / torque residuals")):
        lines.append(r"\multicolumn{7}{@{}l}{\textit{\textbf{%s}}} \\" % rlabel)
        for leg in ("chain_ss", "collider_ss", "chain_surrogate", "bpinn"):
            cells, present = [], False
            for system in SYSTEMS:
                exact = False
                if leg == "collider_ss":
                    D, exact = load_collider_ss(system, residual)
                else:
                    D = load(system, residual, leg)
                if D is None:
                    cells += ["---"] * 2
                    continue
                present = True
                t = ray_shift(D, system)
                stat = np.mean if D.ok else np.median
                mark = (r"^{\ddagger}" if exact else "") + ("" if D.ok or exact else r"^{\dagger}")
                tv = float(stat(t))
                tt = "%+.2f%s" % (tv, mark)
                cells.append(r"$\mathbf{%s}$" % tt if abs(tv) > np.log(2) else "$%s$" % tt)
                cells.append(ratio(float(stat(IDENT[system][1](D)) / IDENT[system][1](D.truth))))
                mass = sum(D[n] for n in SCALE[system]) / sum(D.truth[n] for n in SCALE[system])
                log.append("%-10s %-5s %-15s t %+8.2f  sd %.3f  (%.1f sd)  mass scale / truth %.3g%s"
                           % (system, residual, leg, tv, float(np.std(t)), abs(tv) / float(np.std(t)),
                              float(stat(mass)), "  exact Kalman" if exact else ""))
            if present:
                lines.append(r"%-34s & %s \\" % (LEG_LABEL[leg], " & ".join(cells)))
        lines.append(r"\midrule")
    write("scorecard.tex", "\n".join(lines[:-1]))

    # the chain's worst standardized error over every identified combination
    worst = []
    for system in SYSTEMS:
        for residual in ("accel", "force"):
            D = load(system, residual, "chain_ss")
            if D is None:
                continue
            for f in IDENTIFIED[system]:
                u = np.log(f(D))
                worst.append(abs(u.mean() - np.log(f(D.truth))) / u.std())
    print("\n".join(log))
    if worst:
        print("chain (state space) recovers every identified quantity within %.2f posterior sd" % max(worst))


# ---- surrogate chain -------------------------------------------------------------
def surrogate_chain():
    """Displacement of log m (force) and log(c/m) (acceleration) before and after the correction."""
    lines = []
    for residual, label, q in (("force", "force residuals", "m"), ("accel", "acceleration residuals", "c/m")):
        Dc, Dn = load("oscillator", residual, "bpinn"), load("oscillator", residual, "chain_surrogate")
        if Dc is None or Dn is None:
            continue
        get = (lambda D: D["c"] / D["m"]) if q == "c/m" else (lambda D: D[q])
        truth = Dc.truth["c"] / Dc.truth["m"] if q == "c/m" else Dc.truth[q]
        dc, dn = np.log(get(Dc)).mean() - np.log(truth), np.log(get(Dn)).mean() - np.log(truth)
        lines.append("%s & $\\log %s$ & $%+.3f$ & $%+.3f$ & $%.1f\\%%$ \\\\"
                     % (label, q, dc, dn, 100 * (1 - abs(dn) / abs(dc))))
    write("surrogate_chain.tex", "\n".join(lines))
    legs = {leg: load("oscillator", "accel", leg) for leg in ("chain_surrogate", "bpinn", "chain_ss")}
    if None not in legs.values():
        print("oscillator accel, k/m: " + "  ".join(
            "%s %.2f" % (leg, (D["k"] / D["m"]).mean()) for leg, D in legs.items()))


# ---- per-system settings -----------------------------------------------------------
def settings():
    from colliderbias.bpinn import Surrogate
    cols = []
    for system in SYSTEMS:
        cell = cell_name(system, "force")
        _, _, net_kw = C.cell_settings(cell)
        sysm = C.make_system(cell)
        net = Surrogate(sysm, **net_kw)
        cols.append(dict(
            truth=", ".join("%s" % num(float(v), 3) for v in sysm.lam_phys_true),
            sigma_r="%.2f" % sysm.sigma_r_true, horizon="%g" % sysm.horizon, K="%d" % sysm.K,
            sigma_obs="%g" % sysm.sigma_obs, T="%d" % sysm.T, particles="%d" % C.N_PARTICLES[system],
            arch=net.arch, dim_theta="%d" % net.n_weights, Nd="%d" % (net.N * sysm.d)))
    rows = [("ground truth $\\lambda$", "truth"), ("residual scale $\\sigma_r$ (fixed)", "sigma_r"),
            ("horizon $\\mathcal{T}$", "horizon"), ("observations $K$", "K"),
            ("observation noise $\\sigma_{\\mathrm{obs}}$", "sigma_obs"),
            ("Euler--Maruyama steps $T$", "T"), ("particles", "particles"),
            ("MLP layer widths", "arch"), ("$\\dim\\theta$", "dim_theta"),
            ("residual components $Nd$", "Nd")]
    write("system_settings.tex", "\n".join("%s & %s \\\\" % (lab, " & ".join(c[k] for c in cols))
                                           for lab, k in rows))


# ---- exact posteriors of the oscillator's state space -------------------------------
def kalman_check():
    Dk, Dp = load("oscillator", "force", "chain_kalman"), load("oscillator", "force", "chain_ss")
    if Dk is not None and Dp is not None:
        lk, lp = np.log(Dk["m"]), np.log(Dp["m"])
        print("oscillator force: log m  Kalman chain %.3f +- %.3f  PMMH chain %.3f +- %.3f (R-hat %.4f)"
              % (lk.mean(), lk.std(), lp.mean(), lp.std(), np.nanmax(Dk.rhat)))
    D = load("oscillator", "force", "collider_kalman")
    if D is not None:
        t = ray_shift(D, "oscillator")
        print("oscillator force: exact collider  log m %.1f  t %.2f +- %.2f  c/m %.3g  k/m %.3g  R-hat %.4f"
              % (np.log(D["m"]).mean(), t.mean(), t.std(), (D["c"] / D["m"]).mean(),
                 (D["k"] / D["m"]).mean(), np.nanmax(D.rhat)))


# ---- joint NUTS against PMMH on the state-space latent ---------------------------------
def estimator_check():
    for system in ("pendulum", "particle"):
        for model in ("chain", "collider"):
            pair = [load_run(os.path.join(RUNS, cell_name(system, "force"), "%s_ss%s.npz" % (model, s)))
                    for s in ("", "_nuts")]
            if None in pair:
                continue
            out = []
            for r in pair:
                scale = sum(r[n] for n in SCALE[system])
                ok = is_posterior(r.draws, r.n_chains)
                ess = ess_autocorr(np.log(scale), r.n_chains)
                out.append("%s %.4g (%s) ESS %.0f  %.1f s per effective sample, %.1f h"
                           % ("mean" if ok else "median", scale.mean() if ok else np.median(scale),
                              "posterior" if ok else "not a posterior", ess, r.wall_s / ess, r.wall_s / 3600))
            print("%-9s force %-8s scale  PMMH: %s\n%27s NUTS: %s" % (system, model, out[0], "", out[1]))


def null_checks():
    for system in SYSTEMS:
        runs = [load_run(os.path.join(RUNS, cell_name(system, "accel"), leg + ".npz"))
                for leg in ("chain_ss", "collider_ss")]
        if None not in runs:
            print("%-24s" % cell_name(system, "accel"), end="")
            guards.null(runs[0].chain, runs[1].chain)


def summary():
    lines = []
    for system in SYSTEMS:
        for residual in ("force", "accel"):
            for leg in ("chain_ss", "collider_ss", "chain_surrogate", "bpinn"):
                D = load(system, residual, leg)
                if D is None:
                    continue
                r = np.nanmax(D.rhat) if np.isfinite(D.rhat).any() else np.nan
                lines.append("%-34s %6d  Rhat %-6s %s" % (
                    cell_name(system, residual) + "/" + leg, len(D.matrix),
                    "%.3f" % r if np.isfinite(r) else "n/a",
                    "  ".join("%s=%.4g+/-%.3g" % (n, D[n].mean(), D[n].std()) for n in D)))
    write("summary.txt", "\n".join(lines))


def main():
    grid_tables()
    scorecard()
    settings()
    surrogate_chain()
    kalman_check()
    estimator_check()
    null_checks()
    summary()


if __name__ == "__main__":
    main()

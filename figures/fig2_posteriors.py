#!/usr/bin/env python
"""Fig. 2: oscillator posteriors of the identified quantities c/m and k/m.

Acceleration residual; Chain (state space) against the B-PINN.
Reads results/runs/oscillator-accel/{chain_ss,bpinn}.npz.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from colliderbias.paths import RESULTS, RUNS  # noqa: E402
from colliderbias.runs import load_run  # noqa: E402
from style import (C_BPINN, C_CHAIN, C_PRIOR, C_TRUTH, TEXTWIDTH, draw_leg,  # noqa: E402
                   finish_density_axis, log_grid, lognormal_unitpeak, use_paper_style)

CELL = os.path.join(RUNS, "oscillator-accel")
OUT = os.path.join(RESULTS, "figures", "fig2_oscillator_posteriors")
LEGS = (("chain_ss", C_CHAIN, "-", "Chain Model (Ours)"),
        ("bpinn", C_BPINN, (0, (1.4, 1.2)), "B-PINN"))


def span(values, truth, prior_sd, pad=0.35):
    lo = [np.log(truth) - 2.2 * prior_sd] + [np.quantile(np.log(v), 0.001) - pad for v in values]
    hi = [np.log(truth) + 2.2 * prior_sd] + [np.quantile(np.log(v), 0.999) + pad for v in values]
    return float(np.exp(min(lo))), float(np.exp(max(hi)))


def main():
    use_paper_style()
    runs = {leg: load_run(os.path.join(CELL, leg + ".npz")) for leg, *_ in LEGS}
    ref = runs["chain_ss"]
    truth = dict(zip(ref.param_names, ref.lam_true))
    psd = dict(zip(ref.param_names, ref.prior_sd))
    panels = [
        (lambda r: r["c"] / r["m"], truth["c"] / truth["m"], np.hypot(psd["c"], psd["m"]),
         r"Damping Rate $c/m$", (0.5, 1, 2)),
        (lambda r: r["k"] / r["m"], truth["k"] / truth["m"], np.hypot(psd["k"], psd["m"]),
         r"Stiffness Ratio $k/m$", (2, 3, 4, 6, 10)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXTWIDTH, 1.75))
    fig.subplots_adjust(left=0.015, right=0.99, top=0.97, bottom=0.36, wspace=0.08)
    for ax, (q, tv, sd, label, ticks) in zip(axes, panels):
        lo, hi = span([q(r) for r in runs.values()], tv, sd)
        gu = log_grid(lo, hi)
        ax.fill_between(np.exp(gu), 0, lognormal_unitpeak(np.exp(gu), tv, sd), color=C_PRIOR,
                        alpha=0.35, lw=0, zorder=0)
        for leg, col, ls, _ in LEGS:
            draw_leg(ax, q(runs[leg]), gu, col, ls=ls, lw=1.3, fill=(leg == "chain_ss"))
        finish_density_axis(ax, lo, hi, truth=tv)
        ax.set_xticks([t for t in ticks if lo <= t <= hi])
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _p: "%g" % v))
        ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        ax.set_xlabel(label, labelpad=1)
        print("  %-26s truth %.3f  " % (label, tv) + "  ".join(
            "%s median %.3f" % (leg, np.median(q(runs[leg]))) for leg, *_ in LEGS))
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.3, label=lab) for _, c, ls, lab in LEGS]
    handles.append(plt.Line2D([], [], color=C_TRUTH, ls=(0, (4, 2)), lw=0.9, label="Ground Truth"))
    handles.append(matplotlib.patches.Patch(facecolor=C_PRIOR, alpha=0.35, edgecolor="none", label="Prior"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.6, columnspacing=2.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT + "." + ext, dpi=220, bbox_inches="tight")
    print("saved", OUT + ".pdf")


if __name__ == "__main__":
    main()

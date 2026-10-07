#!/usr/bin/env python
"""Fig. 3: longer records amplify the dissipation bias (oscillator, acceleration residual).

Displacement of log(c/m) from the truth at horizons 12, 24, 48, with +-1 posterior
sd, from results/runs/oscillator-accel[-T12|-T48] (legs chain_ss, collider_ss,
chain_surrogate, bpinn).  Also prints the numbers of the text, and the
leading-order prediction: the chain posterior reweighted by exp(Delta_diss).

Colours by model and latent: BLUE the chain model, DARK RED the collider on the
causal state-space latent, ORANGE the collider on a global surrogate.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

from colliderbias.paths import RESULTS, RUNS  # noqa: E402
from colliderbias.runs import load_run  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

from style import (C_BPINN, C_CHAIN, C_COLLIDER, C_SURROGATE_CHAIN, COLWIDTH,  # noqa: E402
                   use_paper_style)

OUT = os.path.join(RESULTS, "figures", "fig3_dissipation_horizon")
HORIZON_CELLS = {12: "oscillator-accel-T12", 24: "oscillator-accel", 48: "oscillator-accel-T48"}
# Model (latent), exactly as in the main grid table, so every curve maps onto a
# row of it.  The fourth leg, collider_ss, is bit-identical to chain_ss under the
# acceleration residual and is drawn as a ring on its markers, not as a line.
LADDER_LEGS = (("chain_ss", C_CHAIN, "o", "-", 1.000, "Chain (state space)"),
               ("chain_surrogate", C_SURROGATE_CHAIN, "s", (0, (3.2, 1.4)), 1.035,
                "Chain (Fourier MLP)"),
               ("bpinn", C_BPINN, "D", "-", 0.966, "Collider B-PINN (Fourier MLP)"))


def ladder():
    """Mean and sd of log(c/m) - log(c/m)_true per horizon and leg."""
    out = {leg: ([], []) for leg in [l[0] for l in LADDER_LEGS] + ["collider_ss"]}
    for cell in HORIZON_CELLS.values():
        for leg in out:
            r = load_run(os.path.join(RUNS, cell, leg + ".npz"))
            u = np.log(r["c"] / r["m"]) - np.log(r.lam_true[1] / r.lam_true[0])
            out[leg][0].append(u.mean())
            out[leg][1].append(u.std())
    return np.array(list(HORIZON_CELLS), float), {k: np.array(v) for k, v in out.items()}


def main():
    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(COLWIDTH, 2.25))
    # THE CLAIM UNDER TEST is that a longer record does not rescue a B-PINN.
    # Every leg carries its own +-1 posterior sd as a bar: the chain's width
    # shrinks with the record while the B-PINN's does not, which is what turns a
    # fixed displacement into a growing standardized error.
    horizons, stats = ladder()
    # zero IS the truth here, by the definition of "displacement", so it needs a
    # reference rule but not a label.
    ax.axhline(0.0, color="#B0B0B0", lw=0.7, zorder=1)
    handles = {}
    for leg, col, mk, ls, dodge, _lab in LADDER_LEGS:
        mu, sd = stats[leg]
        x = horizons * dodge                      # dodge: the axis is log
        ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=col, elinewidth=0.8,
                    capsize=1.7, capthick=0.8, alpha=0.9, zorder=2)
        handles[leg], = ax.plot(x, mu, linestyle=ls, color=col, lw=1.2, marker=mk,
                                ms=3.6, mew=0.6, mec="#333333", zorder=3)
    # the collider on the SAME latent is bit-identical: draw it as an open ring
    # ON the chain's markers, so the coincidence is seen, not asserted.
    ring, = ax.plot(horizons, stats["collider_ss"][0], linestyle="none", marker="o",
                    ms=7.2, mfc="none", mec=C_COLLIDER, mew=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(9.9, 57)
    ax.set_ylim(-1.02, 0.86)
    ax.set_xticks([12, 24, 48])
    ax.set_xlabel(r"observation length $\mathcal{T}$", labelpad=1)
    ax.set_ylabel(r"displacement of $\log(c/m)$", labelpad=1)
    # ndivide=1, not None: one section means both handles get the FULL handlebox
    # and are drawn on top of each other, so the ring sits around the dot exactly
    # as it does in the panel.  With None they are laid out side by side.
    ax.legend([(handles["chain_ss"], ring), handles["chain_surrogate"], handles["bpinn"]],
              [r"Chain $=$ Collider (state space)"] + [l[-1] for l in LADDER_LEGS[1:]],
              handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
              loc="upper left", fontsize=5.9, bbox_to_anchor=(-0.012, 1.025),
              labelspacing=0.30, handlelength=2.0, borderpad=0.25, framealpha=0.0)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _p: "%g" % v))
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    # no title: the caption's bold heading states the claim
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print("saved", OUT + ".pdf")

    for k, hz in enumerate(horizons):
        b, s = stats["bpinn"][0][k], stats["bpinn"][1][k]
        print("  horizon %2d  B-PINN %+.3f (%.2f sd, sd %.3f)  chain (surrogate) %+.3f  "
              "chain (state space) %+.3f (sd %.3f)"
              % (hz, b, b / s, s, stats["chain_surrogate"][0][k], stats["chain_ss"][0][k],
                 stats["chain_ss"][1][k]))
    # leading-order prediction: the chain posterior of u = log(c/m), taken as Gaussian,
    # reweighted by exp(Delta_diss) = exp(-e^u horizon / 2)
    for k, hz in enumerate(horizons):
        r = load_run(os.path.join(RUNS, HORIZON_CELLS[int(hz)], "chain_ss.npz"))
        u = np.log(r["c"] / r["m"])
        g = u.mean() + u.std() * np.linspace(-8, 8, 20001)
        w = np.exp(-0.5 * ((g - u.mean()) / u.std()) ** 2 - np.exp(g) * hz / 2)
        print("  horizon %2d  shift relative to the chain: predicted by exp(Delta_diss) %+.2f, B-PINN %+.2f"
              % (hz, (g * w).sum() / w.sum() - u.mean(), stats["bpinn"][0][k] - stats["chain_ss"][0][k]))


if __name__ == "__main__":
    main()

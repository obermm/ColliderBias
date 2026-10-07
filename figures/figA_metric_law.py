#!/usr/bin/env python
"""Appendix figure: the exact residual-metric bias on the oscillator's state-space latent.

Delta(alpha) - Delta(1) along the drift-invariant ray (m, c, k) -> alpha (m, c, k),
force residual, against the closed form -T log alpha, for T = 312, 1248, 2496
steps, and the chain's log posterior.  Reads results/laws/metric_law.npz
(experiments/metric_law.py).
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

from colliderbias.paths import RESULTS  # noqa: E402

from style import C_CHAIN, C_PRED, COLWIDTH, use_paper_style  # noqa: E402

LAWS = os.path.join(RESULTS, "laws")
OUT = os.path.join(RESULTS, "figures", "figA_metric_law")
TITLE = r"Metric bias, $\Theta(T)$: grows with the grid"
RUNGS = ((0, "#E09A8C", 1.0), (2, "#C4553C", 1.5), (3, "#7A1C10", 1.0))   # T = 312, 1248, 2496


def main():
    use_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=(COLWIDTH, 2.30))
    d = np.load(os.path.join(LAWS, "metric_law.npz"))
    alpha, T, chain = d["alpha"], d["T"], d["log_post_chain"]
    i1 = len(alpha) // 2
    for n, (j, col, lw) in enumerate(RUNGS):
        ax.plot(alpha, -T[j] * np.log(alpha), color=C_PRED, lw=2.4, alpha=0.28, zorder=1,
                label=r"$-T\log\alpha$ (closed form)" if n == 0 else None)
        ax.plot(alpha, d["tilt"][j], color=col, lw=lw, zorder=3, label=r"$T=%d$ steps" % T[j])
    ax.plot(alpha, chain[2] - chain[2, i1], color=C_CHAIN, lw=1.6, zorder=4)
    ax.annotate(r"$\log p_{\mathrm{Chain}}$: %.1f nats" % np.ptp(chain[2]), xy=(1.30, 0.0),
                xytext=(1.03, 330), fontsize=6.0, color=C_CHAIN,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_CHAIN, shrinkA=1, shrinkB=1))
    ax.axhline(0, color="#999999", lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(alpha[0], alpha[-1])
    ax.set_ylim(-1000, 1000)
    ax.set_xticks([0.7, 0.85, 1.0, 1.2, 1.4])
    ax.set_xlabel(r"parameter scaling $\alpha$", labelpad=1)
    ax.set_ylabel(r"collider's reward $\Delta$  [nats]", labelpad=1)
    ax.legend(loc="lower left", fontsize=6.0, bbox_to_anchor=(-0.01, -0.03))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _p: "%g" % v))
    ax.xaxis.set_minor_formatter(mtick.NullFormatter())
    ax.set_title(TITLE, fontsize=7.6, pad=6, loc="left", x=0.0, linespacing=1.35)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT + ".png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print("saved", OUT + ".pdf")
    for j in range(len(T)):
        print("  T = %5d   max |tilt + T log alpha| = %.2e nats   tilt span %.0f nats"
              % (T[j], np.max(np.abs(d["tilt"][j] - d["predicted"][j])), np.ptp(d["tilt"][j])))
    print("  chain log posterior spans %.1f nats" % np.ptp(chain[2]))


if __name__ == "__main__":
    main()

"""Typography and drawing primitives shared by the figures (drawn at final print size)."""
import matplotlib as mpl
import numpy as np

C_CHAIN = "#0B4F8A"
C_COLLIDER = "#7A1C10"
C_BPINN = "#EDA43A"
C_SURROGATE_CHAIN = "#3B7EA8"
C_TRUTH = "#111111"
C_PRIOR = "#B8B8B8"
C_PRED = "#444444"
TEXTWIDTH = 6.75
COLWIDTH = 3.25


def use_paper_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4, "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.25, "legend.borderaxespad": 0.2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
        "savefig.dpi": 400, "pdf.fonttype": 42,
    })


def density(vals, grid_u, min_sd=0.0, min_unique=8):
    """Unit-peak Gaussian KDE of log(vals) on grid_u = log x; None if the leg has no drawable width."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 8 or np.unique(v).size < min_unique:
        return None
    u = np.log(v)
    sd = float(u.std())
    if not np.isfinite(sd) or sd <= max(min_sd, 1e-12):
        return None
    bw = 1.06 * sd * v.size ** (-0.2)
    d = np.exp(-0.5 * ((grid_u[:, None] - u[None, :]) / bw) ** 2).sum(axis=1)
    return d / d.max() if d.max() > 0 else None


def draw_leg(ax, vals, grid_u, color, ls="-", lw=1.1, fill=False, z=2):
    """A posterior on a log axis; a bare vertical line at the median if it has no width."""
    d = density(vals, grid_u, min_sd=2.0 * (grid_u[-1] - grid_u[0]) / len(grid_u))
    if d is None:
        v = np.asarray(vals, float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            ax.axvline(float(np.median(v)), color=color, ls=ls, lw=1.3, zorder=z + 1)
        return False
    grid = np.exp(grid_u)
    ax.plot(grid, d, color=color, ls=ls, lw=lw, zorder=z)
    if fill:
        ax.fill_between(grid, 0, d, color=color, alpha=0.16, lw=0, zorder=z - 1)
    return True


def lognormal_unitpeak(grid, median, sd):
    return np.exp(-0.5 * ((np.log(grid) - np.log(median)) / sd) ** 2)


def finish_density_axis(ax, lo, hi, truth=None):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if truth is not None:
        ax.axvline(truth, color=C_TRUTH, ls=(0, (4, 2)), lw=0.9, zorder=6)


def log_grid(lo, hi, n=900):
    return np.linspace(np.log(lo), np.log(hi), n)

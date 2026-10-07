"""Convergence diagnostics on log lam.

split_rhat_ess  split-R-hat, and an ESS from the between-half variance (cheap, coarse;
                used for the tables and the reference cost of the nested comparison)
ess_autocorr    ESS from the autocorrelation (numpyro), for costs per effective sample
"""
import numpy as np

RHAT_OK = 1.05


def split_rhat_ess(draws, n_chains=1):
    x = np.asarray(draws, float)
    n_tot, P = x.shape
    n_chains = max(1, int(n_chains))
    if n_tot % n_chains:
        n_chains = 1
    m = (n_tot // n_chains) // 2
    if m < 4:
        return np.full(P, np.nan), np.full(P, np.nan)
    c = x.reshape(n_chains, -1, P)[:, :2 * m, :].reshape(2 * n_chains, m, P)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(np.where(c > 0, c, np.nan))
    if not np.isfinite(c).all():
        return np.full(P, np.nan), np.full(P, np.nan)
    W = c.var(axis=1, ddof=1).mean(axis=0)
    B = m * c.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (m - 1) / m * W + B / m
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(np.where(W > 0, var_hat / W, np.nan))
        ess = np.where(B > 0, c.shape[0] * m * var_hat / (B / (m - 1) * m), np.nan)
    return rhat, np.minimum(ess, c.shape[0] * m)


def is_posterior(draws, n_chains=1, min_unique=25):
    """False for chains that have not merged or walks that barely moved."""
    rhat, _ = split_rhat_ess(draws, n_chains)
    if not np.isfinite(rhat).any() or np.nanmax(rhat) > RHAT_OK:
        return False
    return np.unique(np.asarray(draws)[:, 0]).size >= min_unique


def ess_autocorr(values, n_chains=1):
    """Autocorrelation ESS of one scalar series, chains stacked along the first axis."""
    from numpyro.diagnostics import effective_sample_size
    x = np.asarray(values, float)
    n_chains = max(1, int(n_chains))
    if x.size % n_chains:
        n_chains = 1
    return float(effective_sample_size(x.reshape(n_chains, -1)))

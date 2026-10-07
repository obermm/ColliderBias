"""Inference on the causal state-space latent z_{0:T}.

The chain model's transition kernel K_lambda is normalized, so a bootstrap
particle filter that samples from it gives an unbiased estimate of p(y | lam),
and PMMH on top of it targets p(lam | y) exactly.  The collider model on the
same latent differs only by the per-step factor

    delta_i = log Z_lambda(z_{i-1}) - log Z_r,i = d log h - log|det A_lambda(z_{i-1})|,

whose sum is the tilt Delta = log Z_lambda - log Z_r on this latent.  It enters
the same filter as an extra potential on the particle weights.
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS, init_to_value


def _resample(logw, key):
    n = logw.shape[0]
    w = jnp.exp(logw - logsumexp(logw))
    u = (jnp.arange(n) + jax.random.uniform(key)) / n
    return jnp.clip(jnp.searchsorted(jnp.cumsum(w), u), 0, n - 1)


def make_log_evidence(system):
    """log p_hat(y | lam) for the chain (collider=False) or collider model."""
    d = system.d
    y = system.y
    t_obs = system.t_obs

    def log_lik_obs(Z, yk):
        return -0.5 * jnp.sum(((Z[:, :d] - yk) / system.sigma_obs) ** 2, axis=1)

    def propagate(Z, t0, lam, key, n_sub, collider):
        h = system.dt_obs / n_sub
        per_particle = collider and not system.A_state_free

        def sub(carry, arg):
            Z, delta = carry
            j, k = arg
            if per_particle:
                delta = delta - jax.vmap(lambda z: system.log_det_A(z[:d], z[d:], lam))(Z)
            Z = system.step_particles(Z, t0 + j * h, lam, h, k)
            return (Z, delta), None

        (Z, delta), _ = jax.lax.scan(sub, (Z, jnp.zeros(Z.shape[0])),
                                     (jnp.arange(n_sub), jax.random.split(key, n_sub)))
        return Z, (delta if per_particle else None)

    @partial(jax.jit, static_argnums=(2, 3, 4))
    def log_evidence(lam, key, n_sub, n_particles, collider):
        key, k0 = jax.random.split(key)
        Z = system.init_mean + system.init_sd * jax.random.normal(k0, (n_particles, 2 * d))
        logw = log_lik_obs(Z, y[0])
        ll = logsumexp(logw) - jnp.log(n_particles)
        key, kr = jax.random.split(key)
        Z = Z[_resample(logw, kr)]

        def step(carry, k_obs):
            Z, ll, key = carry
            key, kp, kr = jax.random.split(key, 3)
            Z, delta = propagate(Z, t_obs[k_obs - 1], lam, kp, n_sub, collider)
            logw = log_lik_obs(Z, y[k_obs])
            if delta is not None:
                logw = logw + delta
            ll = ll + logsumexp(logw) - jnp.log(n_particles)
            Z = Z[_resample(logw, kr)]
            return (Z, ll, key), None

        (_, ll, _), _ = jax.lax.scan(step, (Z, ll, key), jnp.arange(1, system.K))
        if collider:
            h = system.dt_obs / n_sub
            T = (system.K - 1) * n_sub
            log_det = system.log_det_A(None, None, lam) if system.A_state_free else 0.0
            ll = ll + T * (d * jnp.log(h) - log_det)
        return ll - system.K * d * jnp.log(jnp.sqrt(2 * jnp.pi) * system.sigma_obs)

    return log_evidence


def run_pmmh(system, collider, n_particles, n_iter=24000, n_adapt=4000, seed=0,
             n_sub=None, step0=0.02, target_acc=0.234, log_every=None):
    """Adaptive particle-marginal Metropolis--Hastings in log lam, started at the truth.

    Gaussian random walk preconditioned by the running covariance with a
    Robbins--Monro global scale; both are frozen after n_adapt iterations and
    only the frozen phase is returned.
    """
    n_sub = n_sub or system.n_sub
    log_evidence = make_log_evidence(system)
    P = len(system.param_names)
    tag = "collider" if collider else "chain"

    def log_target(lam, key):
        return log_evidence(lam, key, n_sub, n_particles, collider) + system.log_prior(lam)

    key = jax.random.PRNGKey(seed)
    u = jnp.log(system.lam_true)
    key, k0 = jax.random.split(key)
    lt = log_target(jnp.exp(u), k0)

    L = np.linalg.cholesky(np.eye(P) * step0 ** 2)
    scale = 1.0
    u_mean, u_cov, n_seen = np.asarray(u), np.zeros((P, P)), 0
    chain = np.zeros((n_iter - n_adapt, P))
    n_acc = n_acc_frozen = n_bad = 0
    log_every = log_every or max(1, n_iter // 12)
    t0 = time.time()
    for it in range(n_iter):
        key, kprop, kfilt, kacc = jax.random.split(key, 4)
        u_new = u + scale * (L @ jax.random.normal(kprop, (P,)))
        lt_new = log_target(jnp.exp(u_new), kfilt)
        log_acc = (lt_new + jnp.sum(u_new)) - (lt + jnp.sum(u))
        finite = bool(jnp.isfinite(log_acc))
        n_bad += not finite
        accepted = finite and bool(jnp.log(jax.random.uniform(kacc)) < log_acc)
        if accepted:
            u, lt = u_new, lt_new
        n_acc += accepted
        if it >= n_adapt:
            chain[it - n_adapt] = np.asarray(jnp.exp(u))
            n_acc_frozen += accepted
        else:
            scale *= float(np.exp((accepted - target_acc) / (it + 1) ** 0.6))
            ui = np.asarray(u)
            n_seen += 1
            du = ui - u_mean
            u_mean = u_mean + du / n_seen
            u_cov = u_cov + np.outer(du, ui - u_mean)
            if n_seen > 4 * P and it % max(25, n_adapt // 8) == 0:
                L = np.linalg.cholesky(u_cov / (n_seen - 1) * 2.38 ** 2 / P + 1e-8 * np.eye(P))
                scale = 1.0
        if it % log_every == 0:
            print("  pmmh[%s] %6d/%d %s acc=%.2f log_target=%.1f %.0fs"
                  % (tag, it, n_iter, "adapt" if it < n_adapt else "frozen",
                     n_acc / (it + 1), float(lt), time.time() - t0), flush=True)
    n_frozen = n_iter - n_adapt
    note = ("PMMH %d iter (%d adapt), %d particles, n_sub %d; acceptance %.3f; %d accepted"
            " moves in %d frozen draws; %d non-finite proposals"
            % (n_iter, n_adapt, n_particles, n_sub, n_acc_frozen / max(n_frozen, 1),
               n_acc_frozen, n_frozen, n_bad))
    if n_acc_frozen < 50:
        note += "; fewer than 50 moves: not a posterior"
    print("  pmmh[%s] %s" % (tag, note), flush=True)
    return chain, note


# ---- joint NUTS over (lam, z_0, xi) on the same latent -----------------------
def run_path_nuts(system, collider, n_warmup=400, n_samples=1600, max_tree_depth=8,
                  seed=0, n_sub=None):
    """The same two models, non-centred in the driving noise xi, sampled by NUTS.

    The path is the deterministic image of (z_0, xi, lam), and the standard
    normal prior on xi does not depend on lam, so the chain model needs no
    normalizer here and the collider adds sum_i delta_i.
    """
    n_sub = n_sub or system.n_sub
    d, h = system.d, system.dt_obs / n_sub
    T = (system.K - 1) * n_sub

    def rollout(lam, z0, xi):
        def step(carry, xi_i):
            z, i = carry
            x, v = z[:d], z[d:]
            delta = d * jnp.log(h) - system.log_det_A(x, v, lam)
            t = system.t_obs[0] + i * h
            kick = system.sigma_r_true * jnp.sqrt(h) * xi_i
            kick = jnp.linalg.solve(system.A(x, v, lam), kick)
            v = v + system.accel(x, v, t, lam) * h + kick
            z = jnp.concatenate([x + v * h, v])
            return (z, i + 1), (z[:d], delta)

        _, (xs, deltas) = jax.lax.scan(step, (z0, 0), xi)
        return xs, deltas

    def model():
        lam = numpyro.sample("lam", dist.LogNormal(jnp.log(system.prior_median),
                                                   system.prior_sd).to_event(1))
        z0 = numpyro.sample("z0", dist.Normal(system.init_mean, system.init_sd).to_event(1))
        xi = numpyro.sample("xi", dist.Normal(0.0, 1.0).expand([T, d]).to_event(2))
        xs, deltas = rollout(lam, z0, xi)
        x_obs = jnp.vstack([z0[None, :d], xs[n_sub - 1::n_sub]])
        numpyro.sample("y", dist.Normal(x_obs, system.sigma_obs).to_event(2), obs=system.y)
        if collider:
            numpyro.factor("tilt", jnp.sum(deltas))

    init = init_to_value(values={"lam": system.lam_true, "z0": system.init_mean,
                                 "xi": jnp.zeros((T, d))})
    mcmc = MCMC(NUTS(model, init_strategy=init, max_tree_depth=max_tree_depth),
                num_warmup=n_warmup, num_samples=n_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    note = "path NUTS %d + %d, tree %d, n_sub %d" % (n_warmup, n_samples, max_tree_depth, n_sub)
    return np.asarray(mcmc.get_samples()["lam"]), note


# ---- exact likelihood on the same latent (linear-Gaussian systems) ---------------
def run_kalman_rwm(system, collider, n_chains=4, n_iter=40000, n_adapt=10000, log_every=None):
    """Adaptive random-walk Metropolis in log lam on the exact Kalman evidence.

    For the collider, PMMH cannot reach the posterior under the force residual:
    its mode lies far along the drift-invariant ray, where the particle filter
    breaks down.  The Kalman evidence of the chain model plus the tilt
    -T log|det A_lambda| is exact there.  The collider chains start at the
    maximum of the log posterior along the ray (scanned over t in [-50, 0]).
    Chain c uses numpy seed c; the proposal covariance adapts during the first
    n_adapt iterations and only the frozen phase is returned, chains stacked.
    """
    P = len(system.param_names)
    T = system.T
    mu0 = np.asarray(jnp.log(system.prior_median))
    sd0 = np.asarray(system.prior_sd)

    @jax.jit
    def log_post(u):
        lam = jnp.exp(u)
        v = system.log_evidence_kalman(lam) + jnp.sum(-0.5 * ((u - mu0) / sd0) ** 2)
        if collider:
            v = v - T * system.log_det_A(None, None, lam)
        return jnp.where(jnp.isfinite(v), v, -jnp.inf)

    u_true = mu0.copy()
    if collider:
        ray = np.zeros(P)
        ray[list(system.scale_idx)] = 1.0
        ts = np.linspace(-50.0, 0.0, 101)
        vals = [float(log_post(jnp.asarray(u_true + t * ray))) for t in ts]
        u_start = u_true + ts[int(np.nanargmax(vals))] * ray
        print("  kalman[collider] start at t = %.1f along the ray" % ts[int(np.nanargmax(vals))])
    else:
        u_start = u_true
    log_every = log_every or max(1, n_iter // 4)
    t0 = time.time()
    chains, accs = [], []
    for c in range(n_chains):
        rng = np.random.default_rng(c)
        u = u_start + 0.05 * rng.standard_normal(P)
        lp = float(log_post(jnp.asarray(u)))
        cov = np.eye(P) * 0.01
        out = np.zeros((n_iter - n_adapt, P))
        hist, acc = [], 0
        for it in range(n_iter):
            prop = u + rng.multivariate_normal(np.zeros(P), cov)
            lp_new = float(log_post(jnp.asarray(prop)))
            if np.log(rng.uniform()) < lp_new - lp:
                u, lp = prop, lp_new
                acc += it >= n_adapt
            if it < n_adapt:
                hist.append(u.copy())
                if it > 500 and it % 250 == 0:
                    cov = np.cov(np.array(hist[it // 2:]).T) * 2.38 ** 2 / P + 1e-10 * np.eye(P)
            else:
                out[it - n_adapt] = u
            if it % log_every == 0:
                print("  kalman[%s] chain %d %6d/%d log_post=%.1f %.0fs"
                      % ("collider" if collider else "chain", c, it, n_iter, lp, time.time() - t0),
                      flush=True)
        chains.append(np.exp(out))
        accs.append(acc / (n_iter - n_adapt))
    note = ("Kalman RWM, %d chains x (%d adapt + %d), acceptance %s"
            % (n_chains, n_adapt, n_iter - n_adapt, " ".join("%.2f" % a for a in accs)))
    return np.concatenate(chains), note

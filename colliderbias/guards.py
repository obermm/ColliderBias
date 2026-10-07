"""Sanity checks that a run means what it claims.  Each returns (passed, message)."""
import jax
import jax.numpy as jnp
import numpy as np

from .statespace import make_log_evidence


def _report(name, ok, msg):
    print("   [%s] %-14s %s" % ("ok" if ok else "FAIL", name, msg), flush=True)
    return bool(ok), msg


def kalman(system, n_particles, points=(0.8, 1.0, 1.25), n_rep=4, seed=11):
    """Particle-filter evidence of the chain model against the exact Kalman evidence (oscillator)."""
    log_ev = make_log_evidence(system)
    worst = 0.0
    for alpha in points:
        lam = system.along_ray(alpha)
        exact = float(system.log_evidence_kalman(lam))
        est = [float(log_ev(lam, jax.random.PRNGKey(seed + 7 * i), system.n_sub, n_particles, False))
               for i in range(n_rep)]
        worst = max(worst, abs(exact - np.mean(est)) / max(1.0, 3 * np.std(est)))
    return _report("kalman", worst <= 1.0,
                   "PF within %.2f x max(1, 3 sd) of the exact evidence" % worst)


def null(chain_ss, collider_ss):
    """Under A_lambda = I the two state-space posteriors must be bit-identical."""
    diff = float(np.max(np.abs(np.asarray(chain_ss) - np.asarray(collider_ss))))
    return _report("null", diff == 0.0, "max |chain - collider| = %.3g" % diff)


def ray(system, alphas=(0.5, 2.0), n=64, seed=17):
    """Drift invariance along the ray and d log|det A| / d log alpha = d (force)."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(n, system.d)) * 0.3 + np.asarray(system.x0_mean))
    v = jnp.asarray(rng.normal(size=(n, system.d)) * 0.1)
    t = jnp.asarray(rng.uniform(0, system.horizon, size=n))
    lam1 = system.lam_true
    a1 = jax.vmap(lambda xi, vi, ti: system.accel(xi, vi, ti, lam1))(x, v, t)
    ld1 = jax.vmap(lambda xi, vi: system.log_det_A(xi, vi, lam1))(x, v)
    worst_a, worst_ld = 0.0, 0.0
    for alpha in alphas:
        lam = system.along_ray(alpha)
        a = jax.vmap(lambda xi, vi, ti: system.accel(xi, vi, ti, lam))(x, v, t)
        ld = jax.vmap(lambda xi, vi: system.log_det_A(xi, vi, lam))(x, v)
        want = system.d * np.log(alpha) if system.residual == "force" else 0.0
        worst_a = max(worst_a, float(jnp.max(jnp.abs(a - a1) / (jnp.abs(a1) + 1e-30))))
        worst_ld = max(worst_ld, float(jnp.max(jnp.abs(ld - ld1 - want))))
    return _report("ray", worst_a < 1e-10 and worst_ld < 1e-10,
                   "drift rel. change %.1e, log|det A| off by %.1e" % (worst_a, worst_ld))


def tilt(system, n_particles, alphas=(0.8, 1.25), seed=23):
    """Delta(alpha) - Delta(1) against -T d log alpha (exact where A is state-free)."""
    log_ev = make_log_evidence(system)
    key = jax.random.PRNGKey(seed)

    def delta(alpha):
        lam = system.along_ray(alpha)
        return (float(log_ev(lam, key, system.n_sub, n_particles, True))
                - float(log_ev(lam, key, system.n_sub, n_particles, False)))

    d1 = delta(1.0)
    worst = 0.0
    for alpha in alphas:
        want = -system.T * system.d * np.log(alpha) if system.residual == "force" else 0.0
        worst = max(worst, abs(delta(alpha) - d1 - want))
    if system.A_state_free:
        return _report("tilt", worst < 1e-6 * max(system.T, 1),
                       "Delta along the ray matches -T d log alpha to %.2e nats" % worst)
    return _report("tilt", True, "state-dependent A: deviation from -T d log alpha %.1f nats"
                   " (path-dependent, reported only)" % worst)


def step_convergence(system, n_particles, alphas=(1.0, 2.0), n_rep=3, seed=909):
    """Chain evidence profile along the ray at n_sub against 2 n_sub."""
    log_ev = make_log_evidence(system)
    prof = {}
    for f in (1, 2):
        ns = system.n_sub * f
        rows = [[float(log_ev(system.along_ray(a), jax.random.PRNGKey(seed + i), ns, n_particles, False))
                 for i in range(n_rep)] for a in alphas]
        prof[f] = np.array(rows)
    slope = {f: prof[f][1].mean() - prof[f][0].mean() for f in (1, 2)}
    noise = np.sqrt(sum(prof[f][i].var() / n_rep for f in (1, 2) for i in (0, 1)))
    diff = abs(slope[1] - slope[2])
    return _report("step", diff <= max(1.0, 3 * noise),
                   "profile change n_sub -> 2 n_sub: %.2f nats (noise %.2f)" % (diff, noise))


def filter_noise(system, n_particles, collider=False, n_rep=8, seed=707):
    """sd of log p_hat at the truth; PMMH mixes well for sd ~ 1-1.7."""
    log_ev = make_log_evidence(system)
    v = [float(log_ev(system.lam_true, jax.random.PRNGKey(seed + i), system.n_sub, n_particles,
                      collider)) for i in range(n_rep)]
    sd = float(np.std(v))
    return _report("filter_noise", sd < 2.0, "sd(log p_hat) = %.2f at the truth" % sd)


def normalized_residual(bpinn, seed=3):
    """The residual likelihood is a normalized Gaussian: compare with an explicit density."""
    s, net = bpinn.system, bpinn.net
    theta = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (net.n_weights,))
    worst = 0.0
    for alpha in (0.7, 1.0, 1.6):
        lam = s.along_ray(alpha)
        r, _ = net.residual(theta, lam)
        sd = s.sigma_r_true / jnp.sqrt(net.w_c)[:, None]
        explicit = jnp.sum(-0.5 * (r / sd) ** 2 - jnp.log(sd * jnp.sqrt(2 * jnp.pi)))
        worst = max(worst, abs(float(bpinn.log_lik_residual(r, lam)) - float(explicit)))
    return _report("normalized", worst < 1e-6, "|model - explicit| = %.1e" % worst)


def surrogate_fit(bpinn, theta):
    """Surrogate at a posterior draw: data chi^2 against K d, and RMSE to the latent path."""
    s, net = bpinn.system, bpinn.net
    chi2 = float(jnp.sum(((net.x(theta, s.t_obs) - s.y) / s.sigma_obs) ** 2))
    idx = np.linspace(0, len(s.t_path) - 1, 400).astype(int)
    rmse = float(jnp.sqrt(jnp.mean((net.x(theta, s.t_path[idx]) - s.x_path[idx]) ** 2)))
    n = s.K * s.d
    ok = chi2 < 4.0 * n and rmse < 5.0 * s.sigma_obs
    return _report("fit", ok, "chi^2 %.1f vs K d = %d, path RMSE %.2g sigma_obs"
                   % (chi2, n, rmse / s.sigma_obs))


def barrier_inactive(bpinn, thetas, lams):
    """The particle's subluminal barrier must contribute nothing on posterior draws."""
    vals = [float(bpinn.system.barrier(bpinn.net.residual(th, lam)[1], lam))
            for th, lam in zip(thetas, lams)]
    return _report("barrier", max(vals) == 0.0, "max barrier term %.2g" % max(vals))


def skeleton(system, factors=(1, 4, 16)):
    """Noise-free Euler integrator against the closed-form solution (particle)."""
    errs = []
    for f in factors:
        h = system.dt_sim * factors[-1] / f
        n = int(round(system.horizon / h)) + 1
        ts = h * jnp.arange(n)

        def step(z, t):
            z = system.em_step(z, t, system.lam_true, h, None, noise=False)
            return z, z

        _, zs = jax.lax.scan(step, system.init_mean, ts[:-1])
        x = jnp.vstack([system.x0_mean[None, :], zs[:, :1]])
        x_ref, _ = system.skeleton(system.lam_true, ts)
        errs.append(float(jnp.max(jnp.abs(x - x_ref))))
    return _report("skeleton", errs[-1] < 0.1 * system.sigma_obs,
                   "max |x - x_exact| = %s" % ", ".join("%.1e" % e for e in errs))

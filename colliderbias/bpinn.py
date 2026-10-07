"""B-PINN (collider) on a Fourier-feature MLP surrogate x_theta(t), sampled by NUTS.

Posterior:  p(lam) p(theta) p(r | lam, theta) p(y | theta), with the
residual likelihood a normalized Gaussian over N trapezoid collocation points,
r_n ~ N(0, Sigma_r / w_n), sigma_r fixed.  With correct_log_Z=True the two-term
approximation of the collider bias,

    log Z_r - log Z_lam ~= (n_eff / Nd) sum_n log|det A_lam| - 1/2 int div f dt,

is added, which turns the collider into the chain model on the same latent
("Chain (surrogate)").  n_eff is measured once at frozen theta and lam_true.
The correction needs a constant A_lambda and div f, i.e. the oscillator.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value


class Surrogate:
    """x_theta(t): input features -> tanh MLP -> (d,), with a flat weight vector theta."""

    def __init__(self, system, n_fourier=16, hidden=(24, 24), n_colloc=128):
        self.system = system
        self.n_fourier = n_fourier
        n_in = system.features(jnp.zeros(1), n_fourier).shape[1]
        self.dims = [n_in] + list(hidden) + [system.d]
        self.shapes = list(zip(self.dims[:-1], self.dims[1:]))
        self.n_weights = sum(a * b + b for a, b in self.shapes)
        self.arch = "-".join(map(str, self.dims))
        std = []
        for a, b in self.shapes:
            std += [np.full(a * b, 1.0 / np.sqrt(a)), np.ones(b)]
        self.theta_std = jnp.asarray(np.concatenate(std))       # P = diag(theta_std^-2)
        self.t_c = jnp.linspace(0.0, system.horizon, n_colloc)
        self.w_c = jnp.full(n_colloc, float(self.t_c[1] - self.t_c[0])).at[0].mul(0.5).at[-1].mul(0.5)
        self.N = n_colloc

    def layers(self, theta):
        out, i = [], 0
        for a, b in self.shapes:
            W = theta[i:i + a * b].reshape(a, b)
            i += a * b
            out.append((W, theta[i:i + b]))
            i += b
        return out

    def x(self, theta, ts):
        h = self.system.features(ts, self.n_fourier)
        layers = self.layers(theta)
        for W, b in layers[:-1]:
            h = jnp.tanh(h @ W + b)
        W, b = layers[-1]
        return h @ W + b

    def xva(self, theta, ts):
        """(x, xd, xdd) at times ts, each (n, d), by nested forward-mode derivatives."""
        ones = jnp.ones_like(ts)
        f1 = lambda tt: jax.jvp(lambda s: self.x(theta, s), (tt,), (ones,))
        (x, v), (_, a) = jax.jvp(f1, (ts,), (ones,))
        return x, v, a

    def residual(self, theta, lam):
        """r_n = A_lam (xdd - a_lam) at the collocation points, (N, d), and xd there."""
        x, v, a = self.xva(theta, self.t_c)
        r = jax.vmap(self.system.residual_fn, in_axes=(0, 0, 0, 0, None))(x, v, a, self.t_c, lam)
        return r, v

    def init_theta(self, key, scheme):
        """'small': 0.1 N(0, 1) / sqrt(n_in) everywhere; 'lecun': W ~ N(0, 1/fan_in), b ~ 0.1 N(0, 1)."""
        if scheme == "small":
            return 0.1 * jax.random.normal(key, (self.n_weights,)) / jnp.sqrt(self.dims[0])
        keys = jax.random.split(key, 2 * len(self.shapes))
        parts = []
        for i, (a, b) in enumerate(self.shapes):
            parts += [((1.0 / np.sqrt(a)) * jax.random.normal(keys[2 * i], (a, b))).ravel(),
                      0.1 * jax.random.normal(keys[2 * i + 1], (b,))]
        return jnp.concatenate(parts)


class BPINN:
    def __init__(self, system, surrogate, init_prior=False, barrier=False,
                 correct_log_Z=False, n_eff=None):
        self.system, self.net = system, surrogate
        self.init_prior = init_prior
        self.barrier = barrier
        self.correct_log_Z = correct_log_Z
        self.n_eff = n_eff

    # ---- log density ----------------------------------------------------------
    def log_lik_residual(self, r, lam):
        scale = self.system.sigma_r_true / jnp.sqrt(self.net.w_c)[:, None]
        return jnp.sum(dist.Normal(0.0, scale).log_prob(r))

    def log_lik_obs(self, theta):
        return jnp.sum(dist.Normal(self.net.x(theta, self.system.t_obs), self.system.sigma_obs)
                       .log_prob(self.system.y))

    def extra_factors(self, theta, v_c, lam):
        """Initial-state prior on the surrogate and the particle's subluminal barrier."""
        s, out = self.system, 0.0
        if self.init_prior:
            x, v, _ = self.net.xva(theta, jnp.zeros(1))
            out = out - 0.5 * (jnp.sum(((x[0] - s.x0_mean) / s.x0_sd) ** 2)
                               + jnp.sum(((v[0] - s.v0_mean) / s.v0_sd) ** 2))
        if self.barrier:
            out = out - s.barrier(v_c, lam)
        return out

    def log_Z_correction(self, lam):
        """log Z_r - log Z_lam (two-term approximation); zero unless correct_log_Z."""
        if not self.correct_log_Z:
            return 0.0
        s = self.system
        out = -0.5 * s.div_f(lam) * s.horizon
        if s.residual == "force":
            out = out + self.n_eff * s.log_det_A(None, None, lam)
        return out

    def log_joint(self, theta, lam):
        lp_theta = jnp.sum(dist.Normal(0.0, self.net.theta_std).log_prob(theta))
        r, v_c = self.net.residual(theta, lam)
        return (self.log_lik_obs(theta) + self.log_lik_residual(r, lam) + lp_theta
                + self.system.log_prior(lam) + self.extra_factors(theta, v_c, lam)
                + self.log_Z_correction(lam))

    def model(self):
        s = self.system
        lam = jnp.stack([numpyro.sample(nm, dist.LogNormal(jnp.log(s.prior_median[i]), s.prior_sd[i]))
                         for i, nm in enumerate(s.param_names)])
        theta = numpyro.sample("theta", dist.Normal(jnp.zeros(self.net.n_weights),
                                                    self.net.theta_std).to_event(1))
        numpyro.sample("y", dist.Normal(self.net.x(theta, s.t_obs), s.sigma_obs), obs=s.y)
        r, v_c = self.net.residual(theta, lam)
        numpyro.sample("r", dist.Normal(0.0, s.sigma_r_true / jnp.sqrt(self.net.w_c)[:, None]),
                       obs=r)
        if self.init_prior or self.barrier:
            numpyro.factor("extra", self.extra_factors(theta, v_c, lam))
        if self.correct_log_Z:
            numpyro.factor("log_Z_correction", self.log_Z_correction(lam))

    # ---- n_eff = tr[G (G + P)^-1] ------------------------------------------------
    def measure_n_eff(self, theta, lam=None):
        s = self.system
        lam = s.lam_true if lam is None else lam
        r_white = lambda th: (self.net.residual(th, lam)[0] * jnp.sqrt(self.net.w_c)[:, None]
                              / s.sigma_r_true).ravel()
        J = np.asarray(jax.jacrev(r_white)(theta))
        G = J.T @ J
        H = G + np.diag(np.asarray(self.net.theta_std) ** -2.0)
        return float(np.trace(np.linalg.solve(H, G)))

    # ---- MAP warm start and NUTS ------------------------------------------------
    def fit_map(self, lam, key, n_steps, init="lecun", lr=3e-3, n_chunks=8):
        """Adam with a cosine-decayed step on -log p(theta | y, lam) at fixed lam."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        objective = lambda theta: -self.log_joint(theta, lam)

        def step(carry, it):
            theta, mom1, mom2 = carry
            val, g = jax.value_and_grad(objective)(theta)
            mom1 = beta1 * mom1 + (1 - beta1) * g
            mom2 = beta2 * mom2 + (1 - beta2) * g ** 2
            lr_t = lr * 0.5 * (1 + jnp.cos(jnp.pi * it / n_steps))
            theta = theta - lr_t * (mom1 / (1 - beta1 ** (it + 1))) / (
                jnp.sqrt(mom2 / (1 - beta2 ** (it + 1))) + eps)
            return (theta, mom1, mom2), val

        run = jax.jit(lambda c, its: jax.lax.scan(step, c, its))
        theta = self.net.init_theta(key, init)
        carry = (theta, jnp.zeros_like(theta), jnp.zeros_like(theta))
        per = n_steps // n_chunks
        for c in range(n_chunks):
            carry, vals = run(carry, c * per + jnp.arange(per))
            print("   adam %6d/%d  -log p = %12.3f" % ((c + 1) * per, n_steps, float(vals[-1])),
                  flush=True)
        return carry[0]

    def run_nuts(self, n_chains=4, n_warmup=3000, n_samples=6000, max_tree_depth=11,
                 seed=0, map_steps=20000, map_chunks=8, map_init="lecun",
                 map_key_offset=100, mcmc_key_offset=0, progress_bar=False):
        s = self.system
        t0 = time.time()
        theta0 = self.fit_map(s.lam_true, jax.random.PRNGKey(seed + map_key_offset),
                              map_steps, init=map_init, n_chunks=map_chunks)
        print("   MAP warm start %.0f s" % (time.time() - t0), flush=True)
        init = {nm: float(s.lam_true[i]) for i, nm in enumerate(s.param_names)}
        init["theta"] = theta0
        kernel = NUTS(self.model, target_accept_prob=0.8, max_tree_depth=max_tree_depth,
                      init_strategy=init_to_value(values=init))
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
                    chain_method="vectorized", progress_bar=progress_bar)
        mcmc.run(jax.random.PRNGKey(seed + mcmc_key_offset))
        smp = mcmc.get_samples()
        chain = np.stack([np.asarray(smp[nm]) for nm in s.param_names], axis=1)
        note = ("B-PINN %s, %d chains x (%d + %d), tree %d, N=%d collocation points; %.0f s"
                % (self.net.arch, n_chains, n_warmup, n_samples, max_tree_depth, self.net.N,
                   time.time() - t0))
        return chain, np.asarray(smp["theta"]), note

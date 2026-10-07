"""Damped linear oscillator,  m xdd + c xd + k x = sigma_r xi(t)   (d = 1).

lambda = (m, c, k).  The drift depends on (c/m, k/m) only, so (m, c, k) -> alpha
(m, c, k) is drift-invariant.  A_lambda = m (force) or 1 (acceleration) is
state-free, and the model is linear-Gaussian, so the chain evidence is
available exactly from a Kalman filter.
"""
import jax
import jax.numpy as jnp

from .base import System


class Oscillator(System):
    name = "oscillator"
    d = 1
    phys_names = ("m", "c", "k")
    scale_idx = (0, 1, 2)
    sigma_obs = 0.02
    x0_mean, x0_sd = jnp.array([1.0]), 0.05
    v0_mean, v0_sd = jnp.array([0.0]), 0.3

    def __init__(self, residual="force", horizon=24.0, K=40,
                 n_sub=32, data_seed=2, c_true=1.2, sigma_r_true=0.30):
        self.lam_phys_true = (1.0, c_true, 4.0)
        self.prior_sd_phys = (0.4, 0.4, 0.25)
        self.sigma_r_true = sigma_r_true
        super().__init__(residual, horizon, K, n_sub, data_seed)

    @property
    def A_state_free(self):
        return True

    def accel(self, x, v, t, lam):
        return -(lam[1] * v + lam[2] * x) / lam[0]

    def A_scalar(self, lam):
        return lam[0] if self.residual == "force" else jnp.ones_like(lam[0])

    def A(self, x, v, lam):
        return self.A_scalar(lam) * jnp.eye(1)

    def log_det_A(self, x, v, lam):
        return jnp.log(self.A_scalar(lam))

    def div_f(self, lam):
        """Phase-space divergence tr(d a / d v) = -c/m (constant)."""
        return -lam[1] / lam[0]

    def _noise_scale(self, lam, h):
        return self.sigma_r_true / self.A_scalar(lam) * jnp.sqrt(h)

    def em_step(self, z, t, lam, h, key, noise=True):
        v = z[..., 1:] + h * self.accel(z[..., :1], z[..., 1:], t, lam)
        if noise:
            v = v + jax.random.normal(key, z.shape[:-1] + (1,)) * self._noise_scale(lam, h)
        x = z[..., :1] + h * v
        return jnp.concatenate([x, v], axis=-1)

    def step_particles(self, Z, t, lam, h, key):
        """All particles share one key (one joint normal draw per step)."""
        return self.em_step(Z, t, lam, h, key)

    def simulate_data(self):
        t_obs = jnp.linspace(0.0, self.horizon, self.K)
        dt_obs = float(t_obs[1] - t_obs[0])
        n_fine = max(self.n_sub, 64)
        h = dt_obs / n_fine
        key = jax.random.PRNGKey(self.data_seed)
        key, _ = jax.random.split(key)
        z = jnp.concatenate([self.x0_mean, self.v0_mean])
        xs = [z[0]]
        for _ in range(self.K - 1):
            for _ in range(n_fine):
                key, k = jax.random.split(key)
                xi = jax.random.normal(k, ())
                v = (z[1] + h * self.accel(z[0], z[1], 0.0, self.lam_true)
                     + xi * self._noise_scale(self.lam_true, h))
                z = jnp.stack([z[0] + h * v, v])
            xs.append(z[0])
        x_path = jnp.stack(xs)
        key, ky = jax.random.split(key)
        y = x_path + self.sigma_obs * jax.random.normal(ky, x_path.shape)
        return t_obs, y[:, None], t_obs, x_path[:, None]

    def features(self, t, n_fourier):
        """Half-period cosine basis cos(k pi t / horizon), k = 0..2K."""
        omega = jnp.arange(0, 2 * n_fourier + 1) * (jnp.pi / self.horizon)
        return jnp.cos(omega[None, :] * t[:, None])

    # ---- exact evidence ------------------------------------------------------
    def log_evidence_kalman(self, lam, n_sub=None):
        """log p(y | lam) of the chain model, exactly, by a Kalman filter."""
        n_sub = n_sub or self.n_sub
        h = self.dt_obs / n_sub
        m, c, k = lam[0], lam[1], lam[2]
        a = 1.0 - h * c / m
        b = -h * k / m
        Phi = jnp.array([[1.0 + h * b, h * a], [b, a]])
        g = jnp.array([h, 1.0])
        Q = (self._noise_scale(lam, h) ** 2) * jnp.outer(g, g)
        mu = self.init_mean
        cov = jnp.diag(self.init_sd ** 2)
        H = jnp.array([1.0, 0.0])
        y = self.y[:, 0]

        def update(mu, cov, yk):
            S = H @ cov @ H + self.sigma_obs ** 2
            gain = (cov @ H) / S
            e = yk - H @ mu
            return (mu + gain * e, cov - jnp.outer(gain, H @ cov),
                    -0.5 * (jnp.log(2 * jnp.pi * S) + e ** 2 / S))

        mu, cov, ll = update(mu, cov, y[0])

        def step(carry, yk):
            mu, cov, ll = carry

            def sub(c, _):
                mu, cov = c
                return (Phi @ mu, Phi @ cov @ Phi.T + Q), None

            (mu, cov), _ = jax.lax.scan(sub, (mu, cov), None, length=n_sub)
            mu, cov, l = update(mu, cov, yk)
            return (mu, cov, ll + l), None

        (_, _, ll), _ = jax.lax.scan(step, (mu, cov, ll), y[1:])
        return ll

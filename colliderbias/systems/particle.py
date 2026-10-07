"""Relativistic charged particle in a harmonic field   (d = 1).

    m0 gamma(v)^3 xdd = q E0 cos(omega t),   gamma = (1 - v^2/c^2)^{-1/2}

lambda = (m0, q, c).  The drive amplitude E0 is set so that max|v| = beta_max c
at the truth.  The drift depends on q/m0 and c, so (m0, q) -> alpha (m0, q) is
drift-invariant; A_lambda = M_lambda(v) = m0 gamma^3 under the force residual.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .base import System

GAMMA_FLOOR = 1e-9


class Particle(System):
    name = "particle"
    d = 1
    phys_names = ("m0", "q", "c")
    lam_phys_true = (1.0, 1.0, 1.0)
    prior_sd_phys = (0.4, 0.4, 0.25)
    scale_idx = (0, 1)
    sigma_r_true = 0.02
    sigma_obs = 0.005
    x0_mean, x0_sd = jnp.zeros(1), 0.02
    v0_mean, v0_sd = jnp.zeros(1), 0.10
    dt_sim = 2.0e-5
    omega = np.pi

    def __init__(self, residual="force", horizon=12.0, K=20,
                 n_sub=640, data_seed=0, beta_max=0.5):
        self.beta_max = float(beta_max)
        u_max = beta_max / np.sqrt(1.0 - beta_max ** 2)
        self.E0 = float(u_max * self.lam_phys_true[0] * self.omega / self.lam_phys_true[1])
        super().__init__(residual, horizon, K, n_sub, data_seed)

    def gamma(self, v, lam):
        return jnp.power(jnp.maximum(1.0 - (v / lam[2]) ** 2, GAMMA_FLOOR), -0.5)

    def M(self, v, lam):
        return lam[0] * self.gamma(v, lam) ** 3

    def F(self, t, lam):
        return lam[1] * self.E0 * jnp.cos(self.omega * t) * jnp.ones(1)

    def accel(self, x, v, t, lam):
        return self.F(t, lam) / self.M(v, lam)

    def A(self, x, v, lam):
        return (self.M(v, lam) if self.residual == "force" else jnp.ones(1)) * jnp.eye(1)

    def log_det_A(self, x, v, lam):
        if self.residual == "force":
            return jnp.log(self.M(v, lam))[0]
        return jnp.zeros(())

    def em_step(self, z, t, lam, h, key, noise=True):
        x, v = z[:1], z[1:]
        v = v + self.accel(x, v, t, lam) * h
        if noise:
            kick = self.sigma_r_true * jnp.sqrt(h) * jax.random.normal(key, (1,))
            v = v + (kick / self.M(z[1:], lam) if self.residual == "force" else kick)
        return jnp.concatenate([x + v * h, v])

    def simulate_data(self):
        n = int(round(self.horizon / self.dt_sim)) + 1
        t_path = self.dt_sim * jnp.arange(n)
        key = jax.random.PRNGKey(0 if self.data_seed == 0 else 1000 + self.data_seed)
        obs_seed = 1 if self.data_seed == 0 else 5000 + self.data_seed

        def step(z, arg):
            t, k = arg
            z = self.em_step(z, t, self.lam_true, self.dt_sim, k)
            return z, z

        _, zs = jax.lax.scan(step, self.init_mean, (t_path[:-1], jax.random.split(key, n - 1)))
        x_path = jnp.vstack([self.x0_mean[None, :], zs[:, :1]])
        idx = np.arange(self.K) * ((n - 1) // (self.K - 1))
        t_obs = jnp.array(np.asarray(t_path)[idx])
        y = jnp.array(np.asarray(x_path)[idx]
                      + self.sigma_obs * np.random.default_rng(obs_seed).normal(size=(self.K, 1)))
        return t_obs, y, t_path, x_path

    def skeleton(self, lam, ts):
        """Noise-free solution in closed form: u = gamma v = (q E0 / m0 omega) sin(omega t)."""
        u = (lam[1] * self.E0) / (lam[0] * self.omega) * jnp.sin(self.omega * ts)[:, None]
        v = u / jnp.sqrt(1.0 + (u / lam[2]) ** 2)
        x = jnp.concatenate([jnp.zeros((1, 1)),
                             jnp.cumsum(0.5 * (v[1:] + v[:-1]) * jnp.diff(ts)[:, None], axis=0)])
        return x, v

    def features(self, t, n_fourier):
        """[t / horizon, s_k sin(omega_k t), s_k cos(omega_k t)], s_k = 2 / k^2."""
        k = jnp.arange(1, n_fourier + 1)
        omega = k * (2.0 * jnp.pi / self.horizon)
        s = 2.0 / k ** 2.0
        a = omega[None, :] * t[:, None]
        return jnp.concatenate([(t / self.horizon)[:, None], jnp.sin(a) * s, jnp.cos(a) * s], axis=1)

    # ---- surrogate barrier ---------------------------------------------------
    beta_cap = 0.95
    barrier_weight = 1e4

    def barrier(self, v, lam):
        """One-sided penalty keeping a surrogate subluminal (|v| < beta_cap c)."""
        over = jnp.maximum(jnp.abs(v) / lam[2] - self.beta_cap, 0.0)
        return self.barrier_weight * jnp.sum(over ** 2)

"""Double pendulum,  M_lambda(x) xdd = F_lambda(x, xd)   (d = 2, x = link angles).

lambda = (m1, m2, l1, l2).  The drift is mass-free under (m1, m2) -> alpha
(m1, m2) while det M_lambda is homogeneous of degree d in the masses.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .base import System

GRAVITY = 9.81


class Pendulum(System):
    name = "pendulum"
    d = 2
    phys_names = ("m1", "m2", "l1", "l2")
    lam_phys_true = (1.0, 0.5, 1.0, 0.8)
    prior_sd_phys = (0.4, 0.4, 0.25, 0.25)
    scale_idx = (0, 1)
    sigma_r_true = 0.02
    sigma_obs = 0.02
    x0_mean, x0_sd = jnp.array([0.5, 0.3]), 0.05
    v0_mean, v0_sd = jnp.zeros(2), 0.3
    dt_sim = 2.0e-4

    def __init__(self, residual="force", horizon=12.0, K=20,
                 n_sub=65, data_seed=0):
        super().__init__(residual, horizon, K, n_sub, data_seed)

    def M(self, x, lam):
        m1, m2, l1, l2 = lam[0], lam[1], lam[2], lam[3]
        cos_dx = jnp.cos(x[0] - x[1])
        return jnp.array([[(m1 + m2) * l1 ** 2, m2 * l1 * l2 * cos_dx],
                          [m2 * l1 * l2 * cos_dx, m2 * l2 ** 2]])

    def F(self, x, v, lam):
        m1, m2, l1, l2 = lam[0], lam[1], lam[2], lam[3]
        sin_dx = jnp.sin(x[0] - x[1])
        f1 = -(m2 * l1 * l2 * v[1] ** 2 * sin_dx + (m1 + m2) * GRAVITY * l1 * jnp.sin(x[0]))
        f2 = -(-m2 * l1 * l2 * v[0] ** 2 * sin_dx + m2 * GRAVITY * l2 * jnp.sin(x[1]))
        return jnp.array([f1, f2])

    def accel(self, x, v, t, lam):
        return jnp.linalg.solve(self.M(x, lam), self.F(x, v, lam))

    def A(self, x, v, lam):
        return self.M(x, lam) if self.residual == "force" else jnp.eye(2)

    def log_det_A(self, x, v, lam):
        if self.residual == "force":
            return jnp.linalg.slogdet(self.M(x, lam))[1]
        return jnp.zeros(())

    def em_step(self, z, t, lam, h, key, noise=True):
        x, v = z[:2], z[2:]
        M = self.M(x, lam)
        v = v + jnp.linalg.solve(M, self.F(x, v, lam)) * h
        if noise:
            kick = self.sigma_r_true * jnp.sqrt(h) * jax.random.normal(key, (2,))
            v = v + (jnp.linalg.solve(M, kick) if self.residual == "force" else kick)
        return jnp.concatenate([x + v * h, v])

    def simulate_data(self):
        n = int(round(self.horizon / self.dt_sim))
        key = jax.random.PRNGKey(0 if self.data_seed == 0 else 1000 + self.data_seed)
        obs_seed = 1 if self.data_seed == 0 else 5000 + self.data_seed

        def step(z, k):
            z = self.em_step(z, 0.0, self.lam_true, self.dt_sim, k)
            return z, z

        _, zs = jax.lax.scan(step, self.init_mean, jax.random.split(key, n - 1))
        x_path = jnp.vstack([self.x0_mean[None, :], zs[:, :2]])
        t_path = jnp.linspace(0.0, self.horizon, n)
        idx = np.arange(self.K) * ((n - 1) // (self.K - 1))
        t_obs = jnp.array(np.asarray(t_path)[idx])
        y = jnp.array(np.asarray(x_path)[idx]
                      + self.sigma_obs * np.random.default_rng(obs_seed).normal(size=(self.K, 2)))
        return t_obs, y, t_path, x_path

    def features(self, t, n_fourier):
        """[t / horizon, sin(omega_k t), cos(omega_k t)], omega_k = 2 pi k / horizon."""
        omega = jnp.arange(1, n_fourier + 1) * (2.0 * jnp.pi / self.horizon)
        a = omega[None, :] * t[:, None]
        return jnp.concatenate([(t / self.horizon)[:, None], jnp.sin(a), jnp.cos(a)], axis=1)

"""Common interface of the three ODE systems.

A system holds the physics (a_lambda, A_lambda), the priors, one synthetic
dataset y, and the semi-implicit Euler--Maruyama step of the Ito SDE,

    v_i = v_{i-1} + h a_lambda(z_{i-1}) + sqrt(h) A_lambda(z_{i-1})^{-1} sigma_r xi_i,
    x_i = x_{i-1} + h v_i.

The parameter vector lam is the physical parameters; the residual scale sigma_r
is fixed at its true value.
"""
import jax
import jax.numpy as jnp
import numpy as np


class System:
    name = None
    d = None                   # configuration dimension
    phys_names = ()            # physical parameters lambda
    lam_phys_true = ()
    prior_sd_phys = ()
    scale_idx = ()             # components rescaled along the drift-invariant ray

    def __init__(self, residual="force", horizon=12.0, K=20,
                 n_sub=32, data_seed=0):
        assert residual in ("accel", "force"), residual
        self.residual = residual
        self.horizon = float(horizon)
        self.K = int(K)
        self.n_sub = int(n_sub)
        self.data_seed = int(data_seed)

        self.param_names = list(self.phys_names)
        self.lam_true = jnp.array(self.lam_phys_true)
        self.prior_median = self.lam_true
        self.prior_sd = jnp.array(self.prior_sd_phys)

        self.t_obs, self.y, self.t_path, self.x_path = self.simulate_data()
        self.dt_obs = float(self.t_obs[1] - self.t_obs[0])

    # ---- to be provided by each system ---------------------------------------
    def accel(self, x, v, t, lam):
        """a_lambda(z, t) for one state, shape (d,)."""
        raise NotImplementedError

    def A(self, x, v, lam):
        """Shaping matrix A_lambda(z), (d, d)."""
        raise NotImplementedError

    def log_det_A(self, x, v, lam):
        return jnp.linalg.slogdet(self.A(x, v, lam))[1]

    def em_step(self, z, t, lam, h, key, noise=True):
        """One Euler--Maruyama step of a single state z = (x, v)."""
        raise NotImplementedError

    def simulate_data(self):
        """(t_obs, y (K, d), t_path, x_path) for this system's data seed."""
        raise NotImplementedError

    def features(self, t, n_fourier):
        """Input embedding of the surrogate x_theta(t)."""
        raise NotImplementedError

    # ---- shared ----------------------------------------------------------------
    @property
    def A_state_free(self):
        """True when log|det A_lambda| does not depend on the state."""
        return self.residual == "accel"

    @property
    def T(self):
        """Number of Euler--Maruyama transitions."""
        return (self.K - 1) * self.n_sub

    @property
    def h(self):
        return self.dt_obs / self.n_sub

    @property
    def init_mean(self):
        return jnp.concatenate([self.x0_mean, self.v0_mean])

    @property
    def init_sd(self):
        return jnp.concatenate([self.x0_sd * jnp.ones(self.d), self.v0_sd * jnp.ones(self.d)])

    def log_prior(self, lam):
        """Independent log-normal priors with median at the truth."""
        u = jnp.log(lam)
        return jnp.sum(-0.5 * ((u - jnp.log(self.prior_median)) / self.prior_sd) ** 2
                       - u - jnp.log(self.prior_sd * jnp.sqrt(2 * jnp.pi)))

    def residual_fn(self, x, v, a, t, lam):
        """r = A_lambda(z) (xdd - a_lambda(z, t)) at one time point."""
        return self.A(x, v, lam) @ (a - self.accel(x, v, t, lam))

    def along_ray(self, alpha):
        """lam_true with the drift-invariant components scaled by alpha."""
        f = np.ones(len(self.param_names))
        f[list(self.scale_idx)] = alpha
        return self.lam_true * jnp.asarray(f)

    def step_particles(self, Z, t, lam, h, key):
        """Advance all particles (n, 2d) by one step, one key per particle."""
        keys = jax.random.split(key, Z.shape[0])
        return jax.vmap(lambda z, k: self.em_step(z, t, lam, h, k))(Z, keys)

    def describe(self):
        return ("%s  residual=%s  sigma_r=%g  horizon=%g  K=%d  n_sub=%d (T=%d)  data_seed=%d"
                % (self.name, self.residual, self.sigma_r_true,
                   self.horizon, self.K, self.n_sub, self.T, self.data_seed))

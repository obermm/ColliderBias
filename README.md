# ColliderBias

Code for *Uncovering and Fixing Generative Bias in Bayesian PINNs*.

The standard B-PINN treats the physical parameters λ and the trajectory
parameters θ as a priori independent and couples them through virtual
observations of the equation residuals, a collider λ → r ← θ. This omits the
λ-dependent normalizer Z_λ of the trajectory distribution, which biases the
posterior over λ. Throughout, the residual (process-noise) scale σ_r is fixed
at its true value. The repository contains

* the chain model λ → z → y on a causal Euler–Maruyama latent, with a bootstrap
  particle filter and particle-marginal Metropolis–Hastings (PMMH);
* the collider on the same latent, and the B-PINN on a Fourier-feature MLP (NUTS);
* the B-PINN with the approximate −log Z_λ restored ("Chain (surrogate)");
* the three test systems, guards, analysis and figure scripts.

## Installation

```bash
pip install -r requirements.txt          # jax (with CUDA for the full runs), numpyro, scipy, matplotlib
```

Everything runs in double precision (`colliderbias/__init__.py` enables `jax_enable_x64`).

## Layout

```
colliderbias/
  systems/         oscillator, double pendulum, relativistic particle: physics, priors, data
  statespace.py    particle filter (chain / collider), PMMH, joint NUTS on the path
  bpinn.py         Fourier-feature surrogate, B-PINN posterior, -log Z_lambda correction, NUTS
  guards.py        sanity checks run alongside every leg
  diagnostics.py   split R-hat and ESS
  runs.py          storage of a run (results/runs/<cell>/<leg>.npz)
experiments/
  configs.py       every run of the paper: cells, legs, seeds and their settings
  run.py           run one (cell, leg)
  metric_law.py    exact residual-metric bias (appendix figure), particle filter vs. Kalman
  replication_exact.py  exact posteriors on a linear surrogate over datasets (appendix)
  neff_sensitivity.py   n_eff away from its expansion point (appendix)
  conservative.py  the dissipation term on the pendulum and the particle (appendix)
  nested_baselines/  NSVI / NPSGLD comparison (appendix)
  slurm/           array launchers
analysis/
  tables.py        main grid table, per-system tables, settings, Chain (Fourier MLP)
  replication.py   replication over datasets
  nested.py        nested inference against PF + PMMH, and the 4-chain PMMH check
figures/           Fig. 2, Fig. 3 and the appendix metric-law figure
```

## Notation

| paper | code | meaning |
|---|---|---|
| λ | `lam` | physical parameters |
| θ | `theta` | surrogate weights (flat vector) |
| z = (x, v) | `z`, `x`, `v` | state: configuration and velocity |
| a_λ, M_λ, F_λ, A_λ | `accel`, `M`, `F`, `A` | acceleration, mass matrix, force, residual shaping matrix |
| r | `residual_fn`, `Surrogate.residual` | A_λ (ẍ − a_λ) |
| σ_r, σ_obs | `sigma_r_true`, `sigma_obs` | residual (process) and observation noise scale, both fixed |
| 𝒯 | `horizon` | observation window |
| K | `K` | number of observations |
| T, h | `T`, `h` | number of Euler–Maruyama transitions and step size |
| – | `n_sub` | Euler–Maruyama steps per observation interval, T = (K − 1) `n_sub` |
| N, w_n | `N`, `w_c` | collocation points and trapezoid weights |
| Δ | `delta` (per step) | collider bias log p_collider − log p_chain = log Z_λ − log Z_r |
| n_eff, G_λ, P | `n_eff`, `G`, `theta_std**-2` | effective constraints, Gauss–Newton curvature, weight-prior precision |
| α | `alpha` | scale along the drift-invariant ray (`System.along_ray`) |
| residual metric | `residual="accel"` (A_λ = I) or `"force"` (A_λ = M_λ) | |

## Runs

A *cell* fixes the system, the residual metric and the dataset, e.g.
`oscillator-accel` or `pendulum-force`. The Fig. 3 horizon cells carry a suffix,
`oscillator-accel-T12` / `-T48`, and cells on another dataset carry the data
seed, `oscillator-accel-d3`. A *leg* fixes model, latent and sampler:

| leg | model | latent | sampler |
|---|---|---|---|
| `chain_ss` | chain | Euler–Maruyama path | particle filter + PMMH |
| `collider_ss` | collider | Euler–Maruyama path | particle filter + PMMH |
| `bpinn` | collider | Fourier-feature MLP | NUTS |
| `chain_surrogate` | chain (−log Z_λ restored) | Fourier-feature MLP | NUTS |
| `chain_ss_nuts`, `collider_ss_nuts` | chain / collider | Euler–Maruyama path | joint NUTS (estimator check) |
| `chain_kalman`, `collider_kalman` | chain / collider | Euler–Maruyama path | exact Kalman likelihood, random-walk Metropolis (oscillator) |

`--seed k` reruns a leg with another sampler seed and stores it as
`<leg>-seed<k>.npz`. The runs of the paper come in groups:

| group | runs |
|---|---|
| `grid` | 3 systems × 2 residuals: `chain_ss`, `collider_ss`, `bpinn`, and `chain_surrogate` on the oscillator |
| `ladder` | `oscillator-accel-T12`, `-T48`, all four legs of the table above |
| `kalman` | `oscillator-force` / `chain_kalman`, `collider_kalman` |
| `estimator` | `{pendulum,particle}-force` / `chain_ss_nuts`, `collider_ss_nuts` |
| `multichain` | `chain_ss --seed 1,2,3` on `oscillator-force`, `oscillator-accel`, `pendulum-force` |
| `replication` | `oscillator-accel-d{0,1,3,...,9}` / `chain_ss`, `bpinn` (data seed 2 is the grid cell) |

```bash
python experiments/run.py --list                              # all 61 runs of the paper
python experiments/run.py --list --groups replication         # one group
python experiments/run.py oscillator-accel chain_ss           # one run
python experiments/run.py oscillator-force chain_ss --seed 1
python experiments/run.py oscillator-accel bpinn --quick      # smoke test, seconds on a CPU
```

`chain_surrogate` measures n_eff at the last draw of the `bpinn` run of the
same cell, so run `bpinn` first. On a SLURM cluster:

```bash
N=$(python experiments/run.py --list --legs chain_ss,collider_ss,bpinn | wc -l)
sbatch --array=0-$((N-1)) --export=ALL,LEGS=chain_ss,collider_ss,bpinn experiments/slurm/submit_runs.sh
sbatch --array=0-3 --export=ALL,LEGS=chain_surrogate experiments/slurm/submit_runs.sh
sbatch --array=0-5 --export=ALL,RUN_GROUPS=kalman,estimator experiments/slurm/submit_runs.sh
```

Wall times on one Quadro RTX 8000: PMMH 8 min (oscillator), 40 min (pendulum),
9 h (particle); B-PINN 2–5 h (oscillator), 5–17 h (pendulum), 10–26 h (particle);
joint NUTS on the state-space latent 1–2 h (pendulum), 51 h (particle). The
Kalman legs take a few minutes on a CPU.

## Reproducing the paper

After the runs:

```bash
python experiments/metric_law.py         # exact metric bias, particle filter vs. Kalman evidence
python experiments/replication_exact.py  # exact posteriors over datasets (CPU, ~10 min)
python experiments/neff_sensitivity.py   # needs oscillator-{force,accel} / chain_ss, bpinn
python experiments/conservative.py       # no runs needed
python analysis/tables.py                # tables -> results/tables/, and the checks quoted in the text
python analysis/replication.py           # replication table
python analysis/nested.py                # nested-inference table, needs results/nested
python figures/fig2_posteriors.py        # Fig. 2
python figures/fig3_dissipation_horizon.py  # Fig. 3, and its numbers in the text
python figures/figA_metric_law.py        # appendix figure on the exact metric bias
```

| result | source |
|---|---|
| Fig. 2 | `oscillator-accel` / `chain_ss`, `bpinn`; `figures/fig2_posteriors.py` |
| main grid table | `results/tables/scorecard.tex`, the `grid` and `kalman` groups |
| per-system settings | `results/tables/system_settings.tex` |
| Fig. 3, horizon ladder, reweighting by exp(Δ_diss) | `oscillator-accel{,-T12,-T48}` / `chain_ss`, `collider_ss`, `bpinn`, `chain_surrogate`; `figures/fig3_dissipation_horizon.py` |
| exact metric bias (appendix figure), filter vs. Kalman | `experiments/metric_law.py`, `figures/figA_metric_law.py` |
| per-system tables (appendix) | `results/tables/grid.tex` |
| exact Kalman posteriors (‡), Kalman chain vs. PMMH | `kalman` group; printed by `analysis/tables.py` |
| estimator check (joint NUTS on the path), cost per effective sample | `estimator` group; printed by `analysis/tables.py` |
| simulation step (data 32× finer) | `experiments/replication_exact.py`; printed by `analysis/replication.py` |
| replication over datasets | `replication` group and `experiments/replication_exact.py`; `results/tables/replication.tex` |
| dissipation term on the conservative systems | `experiments/conservative.py` |
| Chain (Fourier MLP) vs. B-PINN | `results/tables/surrogate_chain.tex` |
| n_eff and its sensitivity | `chain_surrogate` runs; `experiments/neff_sensitivity.py` |
| nested inference (NSVI, NPSGLD), 4-chain PMMH check | `experiments/nested_baselines`, `multichain` group; `results/tables/nested.tex` |

Set `COLLIDERBIAS_RESULTS` to write and read results somewhere other than `results/`.

The datasets, the particle-filter evidence and the B-PINN log density are
identical, to floating-point precision, to those behind the published numbers.
PMMH chains repeat exactly for the same seed on the same platform. NUTS chains
on a GPU are reproducible only statistically.

## Nested-inference baselines

`experiments/nested_baselines` runs NSVI and NPSGLD (Hao & Bilionis) on the
chain model's Euler–Maruyama latent, so model, latent and data match the
`chain_ss` leg and only the inference algorithm differs. The engines come from
[niff-replication](https://github.com/BoltMaxwell/niff-replication) (commit
`dab4403`), which is not included here and needs `optax`:

```bash
pip install optax git+https://github.com/BoltMaxwell/niff-replication@dab4403
python experiments/nested_baselines/run_nested.py --sweep 0    # line 0 of sweeps.txt
sbatch --array=0-43 experiments/slurm/submit_nested.sh         # all oscillator and pendulum runs
python analysis/nested.py
```

## Guards

Every run prints a few checks before sampling:

* `ray`: drift invariance along the drift-invariant ray, log|det A_λ| scaling;
* `kalman`: particle filter against the exact evidence (oscillator);
* `skeleton`: Euler integrator against the closed-form solution (particle);
* `filter_noise`, `step`: sd of log p̂ at the truth, convergence in `n_sub`;
* `tilt`: Δ along the ray against −T d log α (exact for the oscillator);
* `normalized`: the residual likelihood is a normalized Gaussian;
* `fit`, `barrier`: surrogate fit to the data, subluminal barrier inactive (particle).

`analysis/tables.py` additionally checks that chain and collider state-space
chains are bit-identical under the acceleration residual (`guards.null`).

## License

MIT, see `LICENSE`.

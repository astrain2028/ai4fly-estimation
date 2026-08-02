# ai4fly-estimation

Deterministic Bayesian measurement models for fault-aware state estimation on
embedded hardware.

Cooperative Human-Robot Intelligence (COHRINT) Lab, Ann and H.J. Smead
Department of Aerospace Engineering Sciences, University of Colorado Boulder.

## Overview

Recursive Bayesian estimators require a measurement model `h(x)` together
with a measurement noise covariance `R`. In practice `R` is identified
offline and held fixed, so degradation of a sensor in operation — bias,
drift, variance inflation, freezing, intermittent dropout — is unmodelled.
The estimator does not merely lose accuracy; its reported covariance remains
optimistic, and downstream consumers inherit that optimism.

This work investigates learned measurement models that predict a
state-conditioned noise covariance decomposed into aleatoric and epistemic
components, and evaluates whether that decomposition supports correct fault
response. The object of study is the measurement map rather than any
particular filter: the UKF is one consumer, as are the EKF, particle filters,
and factor-graph smoothers.

Online noise covariance identification is a mature problem, treated since the
1970s by innovation-based adaptive estimation. Such methods are necessarily
reactive — they infer covariance from realised residuals, requiring several
epochs of degraded behaviour before responding, and do not separate sensor
degradation from model mismatch. The approach taken here conditions on state
and recent measurements directly, and distinguishes irreducible sensor noise
from model ignorance, a separation the innovation sequence does not expose.

Inference is constrained to bounded worst-case execution time. This is the
binding requirement for real-time scheduling, and it excludes sampling-based
posteriors, whose cost trades against accuracy through a tunable parameter
rather than a bound, as well as deep ensembles, which scale both memory and
computation in the ensemble size. A closed-form Laplace posterior admits a
fixed operation count, a fixed memory footprint, and no data-dependent
control flow.

## Methodology

Three complementary evaluation settings, chosen so that failure modes
obscured in one are exposed in another:

1. **Controlled simulation.** A ground vehicle with known kinematics, known
   sensor geometry, and specified fault models. All quantities are
   observable, so negative results are attributable. Independent randomised
   episodes are generated in the volume required for statistical inference.
2. **Flight data with injected faults.** Logged trajectories supply
   unmodelled dynamics, vibration, and correlated error that simulation does
   not reproduce. Ground truth is itself an estimate, which constrains what
   may be concluded.
3. **Minimal isolating experiments.** Individual claims are reduced to the
   smallest system capable of expressing them, admitting no confounders.

Reported effects are accompanied by permutation nulls, group-level splits,
and within-group comparisons.

## Contents

| Path | Description |
|---|---|
| `robot/dynamics.py` | Differential-drive kinematics: RK4 state propagation, wheel-speed relations, and the encoder inverse |
| `robot/trajectories.py` | Command profiles and rollout: a deterministic reference serpentine, and band-limited randomised episodes for Monte Carlo generation |

```bash
python robot/dynamics.py        # property-based verification of the motion model
python robot/trajectories.py    # trajectory determinism and state-space coverage
```

## Related work

- Z. Chen, H. Biggie, N. Ahmed, S. Julier, and C. Heckman, "Kalman Filter
  Auto-Tuning With Consistent and Robust Bayesian Optimization," *IEEE
  Transactions on Aerospace and Electronic Systems*, vol. 60, no. 2,
  pp. 2236–2250, 2024.
  [doi:10.1109/TAES.2024.3350587](https://doi.org/10.1109/taes.2024.3350587)

- B. W. Israelsen, N. R. Ahmed, M. Aitken, E. W. Frew, D. A. Lawrence, and
  B. M. Argrow, "'A Good Bot Always Knows Its Limitations': Assessing
  Autonomous System Decision-making Competencies through Factorized Machine
  Self-confidence," 2024. [arXiv:2407.19631](https://arxiv.org/abs/2407.19631)

- A. Immer, E. Palumbo, A. Marx, and J. Vogt, "Effective Bayesian
  Heteroscedastic Regression with Deep Neural Networks," *Advances in Neural
  Information Processing Systems 36*, 2023.

## Status

Research code under active development. Results are provisional and reported
with the conditions under which they were obtained.

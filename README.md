# ai4fly-estimation

Research code for **deterministic Bayesian measurement models for fault-aware
state estimation on embedded hardware**.

Developed with the Cooperative Human-Robot Intelligence (COHRINT) Lab, Ann and
H.J. Smead Department of Aerospace Engineering Sciences, University of Colorado
Boulder.

---

## Problem

A state estimator needs a **nonlinear map from state to expected
measurements**, together with a trustworthy statement of how much to rely on
that prediction. In practice the second half is a matrix chosen offline by
hand and then held fixed for the mission.

That holds while sensors behave as they did during tuning. When one degrades
in flight — bias, drift, inflated noise, a frozen reading, intermittent
dropout — the estimator has no mechanism to notice. It is not that faults are
handled badly; they are outside the model entirely.

The consequential failure is not the estimation error. It is that the
estimator's *reported confidence* stays high. Control, planning, and
collision avoidance consume that confidence as though it were earned. An
estimate that is wrong and says so is recoverable. An estimate that is wrong
and confident is not.

## Scope

The object of study is the measurement map itself, not any particular filter.
An unscented Kalman filter is one consumer; an extended Kalman filter,
particle filter, moving-horizon estimator, or factor-graph smoother all
require the same thing. Framing the work around the map keeps it portable.

Three constraints shape the approach:

**The problem is not new.** Estimating measurement noise online has been
studied for fifty years, beginning with innovation-based adaptive estimation.
Classical methods infer noise from the filter's own residuals — necessarily
reactive, requiring several steps of degraded behaviour before responding,
and unable to separate a failing sensor from a mismatched model. What differs
here is the information source: predicting noise from state and recent
measurements *before* a residual exists, and decomposing that prediction into
expected noise versus unfamiliarity.

**Two kinds of uncertainty are not interchangeable.** A sensor can be noisy
and perfectly healthy — vibration during aggressive manoeuvring. It can be
precise and completely wrong — frozen at a plausible value. Treating these as
one quantity yields a system that is confident exactly when it should not be.

**Inference must be schedulable.** The binding requirement for a real-time
system is not average speed but bounded worst-case execution time. That
single constraint excludes most of the Bayesian deep-learning toolkit:
sampling-based methods trade accuracy against latency by a tunable knob
rather than a bound, and ensembles multiply both memory and computation. A
closed-form posterior costs one forward pass and one Jacobian evaluation —
fixed operation count, fixed memory, no sampling, no data-dependent
branching. The argument is not that it is faster; it is that it has a
worst-case bound worth certifying against.

## Approach

The problem is attacked from several directions at once, because failure
modes visible from one angle are invisible from the others.

**A controlled simulation, built from first principles** *(this repository)*.
A ground robot with a known motion model, known sensor geometry, and exactly
specified faults. Nothing is unmodelled, so when a method fails the cause is
unambiguous. It also produces independent randomized runs in the quantity
statistics requires — conclusions drawn from a handful of trajectories are
not conclusions.

**Real flight recordings with injected faults.** Real logs carry the
messiness a simulation cannot invent: vibration, unmodelled dynamics,
imperfect timing, correlated error. This is the credibility check, and also
where diagnosis is hardest, because the ground truth is itself an estimate.

**Minimal single-claim demonstrations.** Where one claim must be established
or refuted, the smallest system that can carry it is built in isolation.
Deliberately too simple to be interesting alone; valuable because they admit
no confounders.

## Repository layout

| Path | Contents |
|---|---|
| `robot/dynamics.py` | Kinematic model of a four-wheeled differential-drive robot: state propagation, wheel-speed relationships, and the encoder inverse |

Components are added incrementally as they are built and independently
checked.

### Running

```bash
python robot/dynamics.py
```

Runs property-based self-tests on the motion model — straight-line motion,
inside/outside wheel-speed reversal, exact recovery of commands from wheel
speeds, turn-in-place, circle closure, turn radius, and angle wrapping.

## Related work at CU Boulder

This work sits alongside several lines of research in the Smead Aerospace
department, particularly on filter consistency and on autonomous systems that
reason about their own limitations.

- Z. Chen, H. Biggie, N. Ahmed, S. Julier, and C. Heckman, "Kalman Filter
  Auto-Tuning With Consistent and Robust Bayesian Optimization," *IEEE
  Transactions on Aerospace and Electronic Systems*, vol. 60, no. 2,
  pp. 2236–2250, 2024. [doi:10.1109/TAES.2024.3350587](https://doi.org/10.1109/taes.2024.3350587)
  — uses chi-squared consistency tests as the objective for tuning filter
  noise parameters. The consistency machinery used here to grade filters is
  the same statistical apparatus, applied to evaluate a learned noise model
  rather than to tune a fixed one.

- B. W. Israelsen, N. R. Ahmed, M. Aitken, E. W. Frew, D. A. Lawrence, and
  B. M. Argrow, "'A Good Bot Always Knows Its Limitations': Assessing
  Autonomous System Decision-making Competencies through Factorized Machine
  Self-confidence," 2024. [arXiv:2407.19631](https://arxiv.org/abs/2407.19631)
  — treats self-assessment of competency as a first-class capability of an
  autonomous system. The uncertainty decomposition studied here is a
  narrow, estimator-level instance of the same idea: a component reporting
  where its own knowledge runs out.

## Status

Active research. Results are provisional and stated with the conditions under
which they were observed. Findings that have not survived scrutiny are
documented as such rather than removed.

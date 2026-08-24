# ai4fly-estimation

Learned measurement models for joint state and sensor-health estimation on
unmanned aircraft, under onboard compute constraints.

In collaboration with Anthony (AJ) La Barca, Cooperative Human-Robot
Intelligence (COHRINT) Lab, Ann and H.J. Smead Department of Aerospace
Engineering Sciences, University of Colorado Boulder. The composite
state-and-health formulation below follows his proposal.

## Formulation

Estimation seeks `p(x | z)` for state `x` given measurements `z`. The
approach taken here defines the state as a composite vector holding not only
the vehicle states but the health of each sensor:

```
x = [ x_vehicle , x_sensor_1 , x_sensor_2 , ... , x_sensor_N ]
```

where each `x_sensor_i` is a continuous degradation level for that sensor
rather than a discrete healthy/faulty label. Estimating over this composite
state with a nonlinear filter yields both a distribution over vehicle states,
through ordinary sensor fusion, and a distribution over the health of each
sensor.

The continuous parameterisation is deliberate. A Gaussian filter propagates
means and covariances, so a discrete mode variable is not something it can
represent — estimating one requires a multiple-model or particle formulation,
a different architecture with different cost. A continuous health variable is
a quantity the filter can carry natively, and it admits partial degradation,
which is what most real faults look like before they become total. Detection
is consequently a regression problem throughout: the network predicts
continuous measurement values, and the filter estimates a continuous health
level. Nothing in the loop performs classification.

Dynamics propagation applies only to the vehicle states. Sensor-health
variables carry no explicit process model — they are static across the
prediction step and updated only when measurements arrive, which places the
burden of fault detection entirely on the measurement model. A persistence
term may be added to discourage rapid switching between healthy and faulty
under transient noise.

The measurement model is therefore the crux. It is conditioned on health
rather than on vehicle state alone,

```
z = h(x_vehicle, x_sensor_1, ..., x_sensor_N) + v
```

and must capture three things: how each measurement relates to the vehicle
states, how sensor health alters that relationship, and how health variables
relate to one another where one fault cascades into others.

A single learned function spans the range. At full health it reproduces the
standard state-to-measurement mapping; as health decreases it reproduces the
degraded characteristics typical of that fault — systematic bias, inflated
variance, or loss of correlation with the true state — with the intermediate
values interpolating between them. At inference, a sensor whose measurements
stop being consistent with the fused state drives its health estimate
downward, which both reduces that sensor's influence on the update and
improves the vehicle estimate by leaning more heavily on the sensors that
remain consistent. The adjustment is graded rather than a switch, so a
partially degraded sensor is partially trusted instead of being kept or
discarded outright.

Not every fault is uniquely isolable with a given sensor suite — a clogged
pitot tube resembles a change in wind absent a multi-hole probe to
disambiguate. Where observability fails, the filter's own consistency
statistics still carry information: a measurement that drives the innovation
covariance up is evidence for lowering the corresponding health estimate,
even where the responsible fault cannot be named.

## Relation to conservative fusion and adaptive filtering

Traditional fusion under possible faults, such as Covariance Intersection
[1], provides rigorous consistency guarantees by combining measurements
conservatively without assuming their cross-correlations are known. The cost
is inflated uncertainty and a less accurate state estimate. Conditioning the
measurement model on health trades that conservatism for adaptivity: faulty
measurements are downweighted rather than hedged against, and healthy ones
are used fully. Some formal guarantees are given up in exchange for accuracy,
which is likely to matter most where faults are transient or intermittent.

Identifying noise covariances online is itself long established, beginning
with Mehra [2], who derived consistency tests on the innovation sequence and
estimators for the process and measurement noise covariances from it. Such
methods are necessarily reactive: they infer covariance from realised
residuals, so several epochs of degraded behaviour must elapse before the
estimate responds, and a degraded sensor is not distinguished from a
misspecified model — both inflate residuals identically. Conditioning on
health acts earlier, on information available before a residual exists, at
the cost of requiring that the relevant fault mode was represented during
training.

## Learned measurement models

Learning the measurement map is an established response to model
misspecification. Ko and Fox [3] set out the general form, replacing the
parametric prediction and observation models of a Bayes filter with Gaussian
processes and instantiating the result as GP-UKF, GP-EKF, and GP-PF,
validated on a robotic blimp. Gupta and Guven [4] augment a Kalman filter
with a neural measurement model for UAV tracking in degraded sensing; de
Curtò and de Zarzà [5] combine a physics-informed network with an adaptive
UKF. All three show a learned map absorbing structure a hand-specified model
omits. Liu et al. [14] make the covariance itself the learned object rather
than the map, regressing a measurement's noise directly from raw sensor data
with no hand-specified parametric form — the narrower ancestor of what is
attempted here.

The Gaussian process case is worth separating out, because it supplies one
half of what is needed here and not the other. Its predictive variance grows
with distance from the training data, which is epistemic uncertainty obtained
for free and without a separate mechanism. But the noise term in a standard
GP is a single learned constant: homoscedastic by construction, and therefore
unable to express noise that varies with operating condition or with sensor
health. The approach here targets both components, at the cost of having to
construct the epistemic term rather than inheriting it.

Lee et al. [6] give a taxonomy of hybrid filters worth positioning against:
state-correction methods, which learn a residual on the predicted state or
innovation; model-learning methods, which parameterise components of the
state-space model with a network; and gain-learning methods, which replace
the analytic Kalman gain outright. The work here is model-learning applied to
the measurement side — the map and its covariance are learned, while the
prediction and update recursions remain analytic.

Their own method sits on the opposite side of the same filter, learning
structured corrections to the dynamics and the process noise, and it differs
in two ways that bear directly on the choices made here. It is trained from
measurement data alone, requiring no ground-truth states, where conditioning
on health requires labelled fault data — which is one of the two barriers
this project has to work around rather than solve. And it obtains model
classification from the innovation likelihood under generalised Bayesian
inference, which is an alternative route to identifying *which* fault is
present that needs neither health in the state nor an uncertainty
decomposition. Whether that route is sufficient is an open question here, and
a live one: if innovation likelihood alone discriminates fault type well,
part of the motivation for decomposing predictive uncertainty weakens.

The taxonomy also carries a caution. Model-learning approaches, Lee et al.
[6], tend to replace explicitly structured components with learned
parameterisations and become harder to interpret in state-space terms. That
applies here, and the mitigation is to keep the learned object recognisable:
a measurement map and a covariance, conditioned on a health variable with a
physical meaning, rather than an unconstrained network mapping states to
corrections.

The nearest published neighbour is Unscented KalmanNet, Ko and Shafieezadeh
[15], which leaves the sigma-point recursions intact and learns time-varying
`Q` and `R` alongside them, trained against an objective rewarding calibration
as well as accuracy and evaluated on both NIS and NEES. Two differences bear
on the choices made here. Its covariances are produced as bounded triangular
multipliers on a baseline Cholesky factor, `R = (L_base A)(L_base A)ᵀ`, which
makes positive-definiteness structural but presupposes a hand-tuned baseline
to deform, and never meets the gradient pathology that motivates the natural
parameterisation below. And it addresses noise misspecification rather than
sensor degradation: no health variable, no epistemic term, no fault to detect.
The overlap covers the healthy-condition case; the divergence is what remains.

### Why the map must predict a covariance, not only a mean

In the formulation above, health enters through `h`, which is the conditional
mean. The noise term `v` is drawn from a covariance that appears nowhere in
the learned function. A network trained to regress `z` can therefore
represent a fault that *shifts* a measurement, but not one that makes it
*noisier* — the mean is unchanged and the additional spread has nowhere to
reside. Since variance inflation is among the degradation modes the model is
required to capture, the covariance must itself be conditioned on health,
`R(x_health)`. A state-conditioned noise covariance is heteroscedastic
regression by construction; it is what completes the formulation rather than
an alternative to it.

Training such a model is not straightforward. Under the usual mean-variance
parameterisation the gradient on the mean is scaled by the inverse predicted
variance, so wherever variance is large the mean stops being fitted, and the
resulting residual justifies a larger variance still — a self-reinforcing
degenerate solution documented by Seitzer et al. [7]. Immer et al. [8] avoid
it by regressing the Gaussian natural parameters, under which the objective
is concave for a linear output layer, and pair this with a Laplace
approximation that keeps the weight posterior closed-form rather than
sampled.

### Distinguishing sensor degradation from model ignorance

That posterior separates predictive uncertainty into aleatoric and epistemic
parts: irreducible sensor noise against ignorance of the model itself. The
distinction addresses a limitation intrinsic to health estimation alone.

Residuals grow for two reasons — a sensor has degraded, or the model is wrong
about the present regime through unmodelled dynamics or conditions absent
from training. A filter estimating health has one explanatory variable
available, so it will attribute model error to sensor failure and report a
healthy sensor as degraded, precisely when the estimator is already under
strain. Epistemic uncertainty reads a different quantity — the input's
position relative to the training distribution rather than residual structure
— so high epistemic alongside large residuals indicates the model is
extrapolating, whereas low epistemic alongside large residuals indicates the
sensor.

De Lucas Álvarez et al. [16] report the closest empirical test of that claim.
Training a heteroscedastic correction to gyro measurements and taking the
epistemic term from an ensemble of five networks, they inject degradation at
three graded severities — correlated noise, periodic vibration, bias steps and
drift, dropout — and find the epistemic term separates the severity regimes
where the aleatoric term does not. That supports the decomposition and
narrows what is left to establish. The open question is no longer whether
epistemic uncertainty responds to degradation, but whether the response
survives two constraints their setup does not impose: obtaining it at a fixed
single-pass cost rather than by sampling an ensemble, and placing the model
inside the filter rather than upstream of it, since their network corrects
measurements before the estimator sees them and reports no consistency
metrics.

The same signal bears on fault coverage. A health-conditioned map learns the
degradation modes present in its training data; a mode outside that set is
unrepresented, and the model still returns a health estimate with nothing to
indicate it is extrapolating. Ovadia et al. [9] show this is the expected
behaviour rather than a defect of any particular model: predictive
uncertainty degrades under distributional shift, and calibration holding
in-distribution fails under even mild shift.

Whether a full Bayesian posterior is warranted, or whether a simpler
distributional score in the manner of Hendrycks and Gimpel [10] suffices, is
open here and consequential for deployment, since the two differ by roughly
an order of magnitude in inference cost.

Reporting where a learned component's knowledge runs out is a narrow instance
of a broader capability: Israelsen et al. [13] treat competency
self-assessment as a first-class function of an autonomous system, rather
than as diagnostics attached after the fact.

### What this does not resolve

Where two faults are indistinguishable given the sensor suite, no
decomposition of uncertainty separates them; that is a property of the
hardware. Correlations between health variables — one fault cascading into
another — require training examples of joint faults, which are
combinatorially many. And the aleatoric component must still be learned from
data, so only the epistemic half is trainable on healthy operation alone.

A more basic assumption runs underneath all of it. The sigma-point recursions
carry a mean and a covariance, the chi-square targets for NIS and NEES hold
only for Gaussian errors, and the natural-parameter likelihood is Gaussian by
construction — so the entire apparatus presumes the posterior stays
approximately Gaussian. Sensor degradation is precisely the regime where that
is least safe: a stuck channel, an intermittent dropout, or a bimodal
"working or not" condition produces a posterior that no single covariance
describes, and a filter can then report excellent consistency statistics
while its distributional shape is wrong. DeMars, Bishop, and Jah [17] address
this directly in orbit determination, using the entropy difference between a
propagated distribution and its Gaussian fit as a criterion for when a single
Gaussian has stopped being adequate, and splitting a mixture component when
it has. Nothing here detects that condition, and the fault modes most likely
to cause it are the ones this project intends to inject.

## Evaluation

Filter consistency is the primary criterion, following Chen et al. [11], [12]:
normalised innovation and estimation-error squared statistics tested against
chi-square bounds. That work uses consistency as an objective for tuning a
fixed noise covariance; here it evaluates a learned, health-conditioned one.
The direction of failure is informative — innovations larger than predicted
indicate an unmodelled noise source, while innovations smaller than predicted
indicate a measurement that has stopped varying — and the two imply opposite
corrective actions.

Validation proceeds from controlled simulation, where all quantities are
observable and negative results are attributable, toward flight data and
ultimately hardware, with faults introduced deliberately in a controlled
setting.

## Known barriers

**Training data.** Supervised training requires runs pairing vehicle states
and sensor measurements with known health labels. Real in-flight faults are
rarely recorded cleanly enough to serve, so simulation carries most of the
burden — which makes the realism of the injected fault models a load-bearing
assumption rather than an implementation detail.

**Onboard compute.** The target platforms have limited onboard resources, so
inference must fit a real-time budget with a bounded worst-case execution
time. This excludes sampling-based posteriors, whose cost trades against
accuracy through a tunable parameter rather than a bound, and deep ensembles,
which scale in the ensemble size. A health-conditioned map is evaluated at
every sigma point — forward evaluations only, exportable to a lean runtime,
but scaling with the augmented state dimension. An epistemic term adds a
Jacobian: bounded, but requiring a runtime with automatic differentiation.
Both admit an execution bound; the constants differ and the comparison is
empirical.

## Contents

| Path | Description |
|---|---|
| `robot/dynamics.py` | Differential-drive kinematics: RK4 state propagation, wheel-speed relations, and the encoder inverse |
| `robot/trajectories.py` | Command profiles and rollout: a deterministic reference serpentine, and band-limited randomised episodes for Monte Carlo generation |
| `robot/sensors.py` | Healthy sensor models: quantised wheel encoders and a biased gyro, with state-dependent noise |
| `robot/faults.py` | Degradation modes applied to readings after the fact: bias, drift, variance inflation, scale error, stuck, dropout, on a continuous severity scale |
| `robot/ukf.py` | Unscented Kalman filter over `[x, y, heading, speed, turn_rate]`, returning innovations and their covariances, accepting a per-step measurement covariance, and notifying measurement models that adapt online |
| `models/fixed/measurement.py` | Control arm: the analytic measurement model with a constant covariance tuned by covariance matching |
| `models/plain/train.py` | Point-prediction baseline: state to sensor readings, mean squared error, no uncertainty output |
| `models/resnet/train.py` | The same objective through residual blocks: a capacity control isolating depth from every other difference |
| `models/adaptive/measurement.py` | Mehra-style online covariance matching: no training, covariance estimated from a window of innovations |
| `models/gp/train.py` | Gaussian process measurement model: epistemic uncertainty from the kernel, with a single homoscedastic noise term |
| `models/bhr/train.py` | Heteroscedastic model: predicts a mean and a state-dependent variance per sensor, trained in the Gaussian natural parameterisation |
| `models/ensemble/train.py` | Five heteroscedastic models from different initialisations; predictive variance splits into within-member noise and between-member disagreement |
| `models/*/measurement.py` | Each arm on one interface: states in, readings out, and a per-step covariance where the arm has one |
| `robot/make_dataset.py` | Regenerates the dataset, with sensor noise growth, encoder resolution, and vehicle calibration error as parameters |
| `models/bhr/laplace.py` | Laplace posterior over the heteroscedastic model's last layer: epistemic uncertainty at a single forward pass, prior precision chosen by evidence |
| `experiments/common.py` | Shared filter settings, arm registry, and scoring, so no experiment can quietly choose its own baseline |
| `experiments/calibration.py` | Test 1: what a hand-written measurement model is worth when the vehicle differs from its specification |
| `experiments/heteroscedasticity.py` | Test 2: when a state-dependent covariance starts earning its cost, swept from homoscedastic upward |
| `experiments/envelope.py` | Test 3: whether the model reports its own ignorance outside the states it was trained on, and at what cost |
| `experiments/tune.py` | Searches process and measurement covariances against both moments of the chi-square criterion |
| `experiments/healthy.py` | Every arm on identical runs with no degradation: accuracy, NIS and NEES with both moments, and cost per filter step |
| `experiments/degradation.py` | Every arm under graded sensor degradation: whether each arm's claimed noise tracks the real noise |
| `data/robot_data.csv` | 100 runs of 20 s at 50 Hz (100,000 rows): state, true wheel rates, sensor readings, and the noise level that produced each reading |
| `data/robot_data_sample.csv` | The first two runs, for inspection without loading the full file |

Each row carries a `run` column. Splits must be made by run rather than by
row: the gyro bias is drawn once per run and shared by every sample in it, so
a row-level split leaks it across the divide.

![dataset](data/dataset_check.png)

A ground robot serves as the controlled setting: known kinematics, known
sensor geometry, and a redundant suite — three channels constraining two
quantities — so that health is observable and cross-channel disagreement is
meaningful.

```bash
python robot/dynamics.py        # property-based verification of the motion model
python robot/trajectories.py    # trajectory determinism and state-space coverage
python robot/sensors.py         # noise calibration, quantisation, channel redundancy
python robot/faults.py          # degradation modes and the severity scale
python robot/ukf.py             # filter consistency on a simulated run

python models/plain/train.py    # point-prediction baseline
python models/resnet/train.py   # the same objective, deeper
python models/bhr/train.py      # heteroscedastic mean and variance
python models/gp/train.py       # Gaussian process, epistemic uncertainty
python models/ensemble/train.py # five heteroscedastic models

python models/bhr/laplace.py    # epistemic term, fitted after training

python experiments/common.py             # which arms are trained and ready
python experiments/healthy.py            # all arms, no degradation
python experiments/degradation.py        # all arms, sensor going bad
python experiments/calibration.py        # test 1: is the map right?
python experiments/heteroscedasticity.py # test 2: is the noise structure right?
python experiments/envelope.py           # test 3: does it know what it does not know?
```

Fusing the gyro with the encoder difference recovers turn rate to 0.0083
rad/s, against 0.0477 rad/s from differencing the encoders alone.
Substituting the learned measurement model for the analytic one leaves
consistency unchanged and costs roughly 10% on turn-rate accuracy — the mean
is learned, while trust remains a fixed matrix chosen by hand.

Consistency is assessed on both moments, following Chen et al. [12], who
show that matching the mean alone is insufficient: a filter can be mistuned
and still produce exactly the expected average NIS, with only the variance
revealing it. For a correctly tuned filter with `n` measurements, NIS has
mean `n` and variance `2n`. Measured over thirty runs with a fixed
covariance:

An initial covariance taken from the sensors' own noise figures gave a mean
NIS of 3.17 against a target of 3 — apparently correct — while the variance
was 7.20 against 6. Whitening the innovations located the cause: their
kurtosis was 2.96 on every channel, so the Gaussian assumption held, but
their variances were 1.28, 1.18, and 0.70 where all three should be 1. Both
encoders were over-trusted and the gyro under-trusted.

That combination is invisible in the mean and plain in the variance. The
mean is a sum, in which the errors cancel (1.28 + 1.18 + 0.70 = 3.17); the
variance is quadratic, in which they compound. This is precisely the failure
mode Chen et al. describe, encountered independently here.

Scaling each channel by its own whitened variance, iterated three times,
gives encoder standard deviations of 0.1805 and 0.1708 and a gyro figure of
0.0080, against starting values of 0.15 and 0.011.

That result was subsequently found to be misleading, and the correction is
worth recording rather than quietly replacing. The tuned encoder figures are
*larger* than the sensors themselves, whose average spread is nearer 0.150.
The excess was state uncertainty being absorbed into the measurement
covariance, because the process model was wrong — see below. Used as the
baseline for a learned covariance, an inflated constant flatters anything
compared against it, and it did: the heteroscedastic model appeared to beat
the analytic one on 17 runs of 20, and against a properly chosen constant it
does not beat it at all. Experiments now compute the best available constant,
which is the average variance, rather than reusing this one.

### The process model, and what it invalidated

NEES failed for every arm, including the hand-written control, which fails it
at 3.65 against a target of 2. A test the control fails is not reporting on
the measurement models.

Tuning could not repair it: process noise large enough to bring NEES to
target pushed NIS away from it, and the reverse. Chen et al. [12] note that
this trade is characteristic of these objectives, so it is suggestive rather
than conclusive, and a direct test settles it. Speed and turn rate were
modelled as random walks, which asserts that their step-to-step changes are
independent. Measured over twenty runs, the lag-1 autocorrelation of those
increments is **0.998**, where a random walk requires zero, and the true
acceleration changes by around two per cent of its own spread per step. The
robot accelerates smoothly; the model said it jitters.

The state now carries acceleration, and speed and turn rate are moved by it:

| | 5 states | 7 states |
|---|---|---|
| speed error, m/s | 0.0080 | 0.0052 |
| NEES, speed and turn rate | 3.65 | 1.81 |
| NEES variance | 13.44 | 3.37 |
| target | 2 and 4 | 2 and 4 |

NEES passes on both moments for the first time. Every NEES figure reported
before this change was grading the process model rather than any measurement
model, and the earlier full-state figure of 22.49 belongs to that category.

NIS remains structurally blind to any state nothing observes — here position
and heading, which are free to drift while every innovation stays well
behaved — which is why both are computed. NEES additionally requires the true
state, so it exists only in simulation; on hardware only NIS is available.

The baseline reaches the noise floor on held-out runs — 0.153 against a
limit of 0.149 for the encoders, 0.0123 against 0.0117 for the gyro — where
the limit combines sensor noise, encoder tick quantisation, and, for the
gyro, the per-run bias that is unknowable for a run not seen in training.
Its residual error tracks wheel speed, rising from 0.110 to 0.130 across the
range, while the model reports a single value throughout. That gap is what a
state-conditioned covariance is for.

## Results

Seven measurement models on one interface, evaluated by three experiments
that each isolate a single mechanism. Organising by mechanism rather than by
scenario is deliberate: the question is which component earns its compute, and
only a design that varies one thing at a time can answer it.

**Healthy, correctly calibrated.** Twenty runs, every arm given the best
constant covariance available to it.

| | speed, m/s | NIS (want 3 / 6) | NEES (want 2 / 4) |
|---|---|---|---|
| analytic + best constant | 0.0052 | 2.52 / 4.31 | 1.81 / 3.37 |
| plain network | 0.0056 | 2.52 / 4.34 | 1.99 / 4.12 |
| residual network | 0.0087 | 2.51 / 4.30 | 3.26 / 12.16 |
| adaptive covariance | 0.0052 | 2.96 / 6.00 | 2.66 / 8.37 |
| Gaussian process | 0.0061 | 2.48 / 4.09 | 2.41 / 6.27 |
| heteroscedastic | 0.0055 | 2.49 / 4.26 | 1.83 / 3.53 |
| ensemble of five | 0.0053 | 2.44 / 4.10 | 1.70 / 3.07 |

No learned arm beats the analytic model here, and two lose measurably on a
paired run-by-run comparison. This is the control condition rather than a
result: the simulator generates readings from the same equations the analytic
model uses, so that model is not an approximation of the truth but the truth
itself, and nothing fitted from data can do better than tie. Every argument
for a learned model lives in the conditions where that assumption fails.

**Test 1, is the map right?** Wheel radius made to differ from the value the
analytic model was built with, which is the ordinary condition of hardware.

| radius error | analytic | plain | heteroscedastic |
|---|---|---|---|
| 0% | 0.0052 | 0.0063 | 0.0087 |
| 1% | 0.0114 | 0.0066 | 0.0074 |
| 3% | 0.0309 | 0.0057 | 0.0080 |

The crossover is below one per cent, and by three per cent the analytic
model's NEES reaches 30.3 while the learned arms are unmoved. A learned map
is not better because it is more expressive; it is better because it fits the
vehicle that exists rather than the one that was written down. The two
learned arms move together, so this argues for learning the map and not for
learning the noise — which is why the plain arm is present.

**Test 2, is the noise structure right?** Sensor noise growth swept from zero,
where a single constant is provably optimal, upward.

| variation | constant, NIS variance | heteroscedastic, NIS variance |
|---|---|---|
| 0% | 5.65 | 5.04 |
| 36% | 6.02 | 5.81 |
| 108% | 6.81 | 6.36 |
| 234% | 8.16 | 6.74 |

Target is 6.0. The constant's second moment degrades by 44% across the sweep
and the learned covariance by 34%, staying nearer target throughout. The
effect is real and confined to the second moment; on accuracy the analytic
model leads at every level. The homoscedastic row is included because it is
where the method should fail, and a sweep reported only where it succeeds
would be a knob turned to suit a conclusion.

**Test 3, do you know what you do not know?** Trained on a restricted
envelope, evaluated on the full range, with healthy sensors throughout — so
anything unfamiliar is unfamiliar rather than broken.

| | speed outside | NIS outside | novelty ratio | cost |
|---|---|---|---|---|
| heteroscedastic | 0.099 | 3.26 | — | 1× |
| + Laplace | 0.099 | 2.83 | 99× | 1× |
| ensemble of five | 0.043 | 1.37 | 121× | 5× |

Target NIS is 3.0. The Laplace posterior gives the best-calibrated
innovations outside the training envelope, and does so at a single forward
pass; the ensemble over-inflates its covariance and becomes under-confident.
Novelty detection is equivalent between them, so the five-fold cost buys no
additional discrimination. What it does buy is accuracy — averaging five
means is more than twice as good where the model extrapolates, which is an
argument for ensembles that has nothing to do with uncertainty.

NEES outside the envelope is poor for all three. The epistemic term keeps
innovations honest without making the state estimate correct: knowing that
one is guessing is not the same as guessing well.

**What is not yet measured.** Faults reach the filter through the
measurement, while every learned arm takes the state as its input, so a
degraded sensor leaves their predictions unchanged — confirmed directly, with
the learned covariance flat across a fourfold change in true noise while the
adaptive arm tracked it to within five per cent. Closing that requires health
in the state, which requires training data with graded fault severities.
Until then the fault results describe the classical baseline only.

## References

**Conservative fusion and adaptive filtering**

[1] S. J. Julier and J. K. Uhlmann, "A Non-divergent Estimation Algorithm in
the Presence of Unknown Correlations," *Proceedings of the American Control
Conference*, vol. 4, pp. 2369–2373, 1997.

[2] R. K. Mehra, "On the Identification of Variances and Adaptive Kalman
Filtering," *IEEE Transactions on Automatic Control*, vol. 15, no. 2,
pp. 175–184, 1970.
[doi:10.1109/TAC.1970.1099422](https://doi.org/10.1109/TAC.1970.1099422)

**Learned measurement models in filtering**

[3] J. Ko and D. Fox, "GP-BayesFilters: Bayesian Filtering Using Gaussian
Process Prediction and Observation Models," *Autonomous Robots*, vol. 27,
no. 1, pp. 75–90, 2009.
[doi:10.1007/s10514-009-9119-x](https://doi.org/10.1007/s10514-009-9119-x)

[4] A. Gupta and E. Guven, "Neural Aided Kalman Filtering for UAV State
Estimation in Degraded Sensing Environments," arXiv:2604.28107.

[5] J. de Curtò and I. de Zarzà, "Hybrid State Estimation: Integrating
Physics-Informed Neural Networks with Adaptive UKF for Dynamic Systems,"
*Electronics*, vol. 13, no. 11, 2208, 2024.
[doi:10.3390/electronics13112208](https://doi.org/10.3390/electronics13112208)

[6] J. Lee, N. R. Ahmed, and R. Russell, "Hybrid Adaptive Kalman Filtering
for Data-Efficient Joint Tracking and Classification," 2026.
[arXiv:2606.02767](https://arxiv.org/abs/2606.02767)

[14] K. Liu, K. Ok, W. Vega-Brown, and N. Roy, "Deep Inference for Covariance
Estimation: Learning Gaussian Noise Models for State Estimation," *IEEE
International Conference on Robotics and Automation*, pp. 1436–1443, 2018.
[doi:10.1109/ICRA.2018.8461047](https://doi.org/10.1109/ICRA.2018.8461047)

[15] M. Ko and A. Shafieezadeh, "Unscented KalmanNet: Structure-Preserving
Deep Learning with Calibrated Posterior Uncertainty under Incomplete Physics
and Unknown Noise," 2026. [arXiv:2608.04201](https://arxiv.org/abs/2608.04201)

**Heteroscedastic regression and predictive uncertainty**

[7] M. Seitzer, A. Tavakoli, D. Antic, and G. Martius, "On the Pitfalls of
Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks,"
*International Conference on Learning Representations*, 2022.
[arXiv:2203.09168](https://arxiv.org/abs/2203.09168)

[8] A. Immer, E. Palumbo, A. Marx, and J. Vogt, "Effective Bayesian
Heteroscedastic Regression with Deep Neural Networks," *Advances in Neural
Information Processing Systems 36*, 2023.

[9] Y. Ovadia et al., "Can You Trust Your Model's Uncertainty? Evaluating
Predictive Uncertainty Under Dataset Shift," *Advances in Neural Information
Processing Systems 32*, 2019.

[10] D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks," *International Conference
on Learning Representations*, 2017.
[arXiv:1610.02136](https://arxiv.org/abs/1610.02136)

[16] M. De Lucas Álvarez, M. Laux, A. de Freitas Precht, M. Martin,
E. Caroselli, F. Kirchner, and A. Fabisch, "Attribution and Uncertainty
Behavior of Learned Residual Gyro Correction for Gyro-Stellar Estimation,"
2026. [arXiv:2607.24608](https://arxiv.org/abs/2607.24608)

**Filter consistency and self-assessment**

[11] Z. Chen, C. Heckman, S. Julier, and N. Ahmed, "Weak in the NEES?:
Auto-tuning Kalman Filters with Bayesian Optimization," 2018.
[arXiv:1807.08855](https://arxiv.org/abs/1807.08855)

[12] Z. Chen, H. Biggie, N. Ahmed, S. Julier, and C. Heckman, "Kalman Filter
Auto-Tuning With Consistent and Robust Bayesian Optimization," *IEEE
Transactions on Aerospace and Electronic Systems*, vol. 60, no. 2,
pp. 2236–2250, 2024.
[doi:10.1109/TAES.2024.3350587](https://doi.org/10.1109/taes.2024.3350587)

[17] K. J. DeMars, R. H. Bishop, and M. K. Jah, "Entropy-Based Approach for
Uncertainty Propagation of Nonlinear Dynamical Systems," *Journal of
Guidance, Control, and Dynamics*, vol. 36, no. 4, pp. 1047–1057, 2013.
[doi:10.2514/1.58987](https://doi.org/10.2514/1.58987)

[13] B. W. Israelsen, N. R. Ahmed, M. Aitken, E. W. Frew, D. A. Lawrence, and
B. M. Argrow, "'A Good Bot Always Knows Its Limitations': Assessing Autonomous
System Decision-making Competencies through Factorized Machine
Self-confidence," 2024. [arXiv:2407.19631](https://arxiv.org/abs/2407.19631)

## Status

Research code under active development. Results are provisional and reported
with the conditions under which they were obtained.

"""
The layered arm, on the quadcopter.

WHAT IS BORROWED AND WHAT IS NEW

The estimator is borrowed whole. robot/ukf.py takes its motion model and its
measurement model as arguments, so nothing about the unscented machinery cares
that the vehicle changed, and models/layered supplies the covariance algebra
with the vehicle-specific parts as constructor arguments rather than hardcoded.
Reusing rather than copying is the point: two implementations of the same
estimator are free to drift apart, and then a result on one says nothing about
the other.

What is new here is the vehicle. Twelve filter states:

    [roll, pitch, yaw, p, q, r,
     bias_accel, bias_gyro, bias_mag,
     noise_accel, noise_gyro, noise_mag]

Six attitude states and six health levels, two per device. Twenty-five sigma
points against the ground robot's twenty-seven, so the cost is comparable and
any difference in the results is about the problem rather than the budget.

WHY HEALTH HAS NO DYNAMICS

Same reason as on the robot. A device does not get better or worse because
time passed, only because evidence arrived, so health is carried forward
untouched through the prediction and moves only in the update. That puts the
entire burden of estimating it on the measurement model, which is where the
argument lives.

WHAT THIS PROBLEM TESTS THAT THE ROBOT COULD NOT

The robot's measurement map is exactly linear, so its learned arms were never
demonstrating that a network can represent a hard function -- linearity.py
measured a least-squares fit beating the trained model on every channel. Here
a linear fit leaves twelve times the noise floor unexplained on the
magnetometer. If the learned map is going to earn its place anywhere, it is
here.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import dynamics
from sensors import CHANNELS


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_borrowing(path, name, extra, shadows=()):
    """Load a module that imports its own neighbours by bare name.

    robot/ukf.py does `from dynamics import step, wheel_spin_rates`, so robot/
    has to be importable while it executes. quad_sim has its own dynamics.py
    and sensors.py, and two things go wrong if that is not handled.

    The first is sys.path: whichever directory comes first silently wins, which
    is the collision common.py's load_arm was written to avoid and which cost
    an afternoon the first time.

    The second is subtler and bit immediately. Python checks sys.modules before
    sys.path, so once quad_sim's `dynamics` is imported, no amount of path
    rearranging makes a bare `import dynamics` find the robot's -- the cached
    one is returned and the path is never consulted. The names have to be
    lifted out of the cache for the duration and put back afterwards.

    Renaming the files would avoid all of it, at the cost of the two
    simulators no longer having the same shape. Keeping them parallel is worth
    twenty lines here.
    """
    saved_path = list(sys.path)
    saved_modules = {m: sys.modules.pop(m) for m in shadows
                     if m in sys.modules}
    sys.path.insert(0, str(extra))
    try:
        return _load(path, name)
    finally:
        sys.path[:] = saved_path
        for m in shadows:
            sys.modules.pop(m, None)     # drop whatever the borrow imported
        sys.modules.update(saved_modules)


train = _load(HERE / "train.py", "quad_train")
layered = _load(ROOT / "models" / "layered" / "measurement.py", "layered_algebra")
ukf = _load_borrowing(ROOT / "robot" / "ukf.py", "robot_ukf_for_quad",
                      ROOT / "robot", shadows=("dynamics", "sensors"))
bhr = train.bhr

N_VEHICLE = 6
N_STATES = 12
HEALTH_STATES = [6, 7, 8, 9, 10, 11]
N_CHANNELS = len(CHANNELS)

# The filter's state is exactly the model's input, in the same order, so the
# model reads every entry. On the ground robot two states -- the accelerations
# -- had to be skipped because no sensor reports them, and TAKE existed to do
# the skipping. Here there is nothing to skip.
TAKE = list(range(N_STATES))

HEALTH_PROCESS_NOISE = 1e-6
HEALTH_START_SPREAD = 0.1

# How much the motion model is distrusted, over the six attitude states. The
# angles are driven entirely by the rates, so their own entries are small; the
# rates are what actually wander, and that is where the process noise goes.
Q = np.diag([1e-8, 1e-8, 1e-8, 2e-3, 2e-3, 2e-3])
P0 = np.diag([0.02, 0.02, 0.30, 0.05, 0.05, 0.05])


def move_state(state, dt):
    """Attitude forward by dt; health carried across untouched."""
    moved = dynamics.step(np.asarray(state, dtype=float)[:N_VEHICLE], dt)
    if len(state) > N_VEHICLE:
        return np.concatenate([moved, state[N_VEHICLE:]])
    return moved


def analytic_readings(states):
    """The hand-written measurement model, for the baseline arm.

    What the sensors should read given the attitude, with no health input and
    no learned anything. This is the equivalent of expected_readings on the
    robot, and it is the thing a learned map has to beat.
    """
    states = np.atleast_2d(states)
    return dynamics_readings(states)


def dynamics_readings(states):
    force = dynamics.specific_force(states[:, 0], states[:, 1], states[:, 2])
    field = dynamics.magnetic_field(states[:, 0], states[:, 1], states[:, 2])
    return np.hstack([force, states[:, 3:6], field])


def filter_settings(Q6=None, P6=None):
    """Widen the six-state Q and P0 to carry health as well."""
    Q6 = Q if Q6 is None else Q6
    P6 = P0 if P6 is None else P6

    Q12 = np.zeros((N_STATES, N_STATES))
    P12 = np.zeros((N_STATES, N_STATES))
    Q12[:N_VEHICLE, :N_VEHICLE] = Q6
    P12[:N_VEHICLE, :N_VEHICLE] = P6
    for i in HEALTH_STATES:
        Q12[i, i] = HEALTH_PROCESS_NOISE
        P12[i, i] = HEALTH_START_SPREAD
    return Q12, P12


def load_health_model(path=None):
    """The learned h(x, m) and R(x, m), as a function the filter can call."""
    if path is None:
        path = HERE / "quad_health.pt"
    saved = torch.load(path, weights_only=False)

    model = train.make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    # bhr's split_outputs reads the channel count from its own module scope,
    # which is three for the robot. The quad has nine.
    bhr.OUTPUTS = train.OUTPUTS

    def measure(states):
        states = np.atleast_2d(states)
        picked = states[:, TAKE].copy()
        picked[:, N_VEHICLE:] = np.clip(picked[:, N_VEHICLE:], 0.0, None)

        x = torch.tensor(picked, dtype=torch.float32)
        with torch.no_grad():
            eta1, eta2 = bhr.split_outputs(model((x - x_mean) / x_std), False)
            mean_s, var_s = bhr.to_mean_and_var(eta1, eta2)

        readings = (mean_s * y_std + y_mean).numpy().astype(float)
        variances = (var_s * y_std ** 2).numpy().astype(float)

        R = np.zeros((len(states), N_CHANNELS, N_CHANNELS))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    return measure


def load_measurement_model(path=None):
    """The learned model wrapped in the additive residual layer."""
    return layered.Layered(load_health_model(path),
                           health_states=HEALTH_STATES,
                           n_channels=N_CHANNELS)


def best_constant_R(n_runs=20):
    """The best single covariance a constant-R filter could use.

    The average measurement variance over healthy flights. A constant cannot
    track anything, so matching the average is the most it can do, and using
    anything else would be handicapping the baseline rather than testing it.
    """
    from sensors import noise_levels
    from trajectories import random_run

    total = np.zeros(N_CHANNELS)
    for seed in range(n_runs):
        total += (noise_levels(random_run(seed)) ** 2).mean(axis=0)
    return np.diag(total / n_runs)


if __name__ == "__main__":
    from faults import apply_fault
    from sensors import read_sensors, stack
    from trajectories import DT, random_run, truth_matrix

    UKF, nis_of = ukf.UKF, ukf.nis

    R_const = best_constant_R()
    print("QUADCOPTER ATTITUDE, %d STATES, %d SIGMA POINTS\n"
          % (N_STATES, 2 * N_STATES + 1))

    if not (HERE / "quad_health.pt").exists():
        print("No quad_health.pt yet -- run train.py first.")
        sys.exit(0)

    measure = load_measurement_model()
    Q12, P12 = filter_settings()

    def go(arm, device=None, mode="bias", severity=0.0, wide=True, seeds=None):
        errors, nis_all = [], []
        for seed in (seeds or range(300, 306)):
            run = random_run(seed)
            meas = read_sensors(run, seed=seed)
            if severity > 0:
                meas = apply_fault(meas, device, mode, severity, seed=seed,
                                   dt=DT)
            readings = stack(meas)
            truth = truth_matrix(run)

            if wide:
                start = np.zeros(N_STATES)
                Qu, Pu = Q12, P12
            else:
                start = np.zeros(N_VEHICLE)
                Qu, Pu = Q, P0
            start[:N_VEHICLE] = truth[0]

            if hasattr(arm, "reset"):
                arm.reset()
            means, _, innov, S = UKF(Qu, R_const, move=move_state,
                                     measure=arm).run(readings, start, Pu, DT)

            # Attitude error in degrees, over roll and pitch. Yaw is left out
            # of the score because it wraps, and a wrapped residual would
            # dominate any average without saying anything about the filter.
            error = np.degrees(means[:, :2] - truth[:, :2])
            errors.append(np.sqrt(np.mean(error ** 2)))
            nis_all.append(nis_of(innov, S).mean())
        return float(np.mean(errors)), float(np.mean(nis_all))

    print("  %-26s %14s %10s" % ("", "attitude rmse", "NIS"))
    print("  " + "-" * 54)
    for label, arm, wide in [("analytic + best const R", analytic_readings,
                              False),
                             ("layered (learned + resid)", measure, True)]:
        e, n = go(arm, wide=wide)
        print("  %-26s %11.3f deg %10.2f" % (label, e, n))
    print("  " + "-" * 54)
    print("\n  Roll and pitch only, in degrees. Target NIS is %d, one per"
          % N_CHANNELS)
    print("  channel. Healthy flights, nothing broken.")

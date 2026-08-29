"""
Shared setup for every experiment.

The point of this file is that no experiment gets to choose its own filter
settings. Before it existed, each model's demo script picked its own R, its
own number of runs, and its own metrics -- so the numbers printed by one
script could not honestly be put in a table beside another's. One of them was
using R = 0.15 for the encoders while the rest used the tuned 0.1805, which
is a 40% difference in how much the filter trusted that sensor.

Everything an experiment needs is defined once, here.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from trajectories import DT, random_run
from sensors import read_sensors, TICKS_PER_TURN
from ukf import UKF, nis, nees

# ---------------------------------------------------------------- settings

# How much the motion model is distrusted, over
#   [x, y, heading, speed, turn_rate, accel, turn_accel]
#
# Speed and turn rate are driven by the accelerations rather than by noise, so
# their own entries are tiny.
#
# The acceleration entries are searched, not measured. The step-to-step change
# in the true accelerations is about 0.0040 m/s^2 and 0.0033 rad/s^2, which
# would suggest 1.6e-5 and 1.1e-5 -- and those values make the filter far
# worse, giving NIS 6.3 against a target of 3. The reason is that acceleration
# sits two integrations away from anything a sensor reports, so it is only
# weakly observable, and the filter needs more freedom to move it than the
# true signal actually uses. What the process noise has to cover here is the
# filter's difficulty in estimating the state, not only the state's own
# variability.
Q = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-3, 1e-1])

# The hand-tuned sensor noise. Arrived at by covariance matching: filter,
# measure how big the innovations really were, adjust, repeat. This is the
# number a learned covariance has to beat, and it is the only R any arm
# without its own covariance is allowed to use.
R_TUNED = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2])

# The best constant covariance is not quite the sensors' average variance.
# tune.py searched for it against Chen's C_NIS / C_NEES and came back with
# 0.85 of that average, which brings all four consistency moments to target
# together: NIS 2.94 / 5.91 and NEES 2.13 / 4.63, against 3 / 6 and 2 / 4.
#
# The direction makes sense. The filter's own state uncertainty already
# explains part of the innovation spread, so R has less to account for than
# the raw sensor noise. The hand-tuned R_TUNED above erred the other way, and
# for the same reason: it was fitted to absorb whatever the wrong process
# model left over, and came out larger than the sensors are.
#
# Fitted on twenty runs, and Chen et al. are explicit that these statistics
# are noisy at that sample size. Treat 0.85 as approximate.
R_SCALE = 0.85

# How unsure the filter is at the start. The accelerations begin unknown, so
# their entries are the spread of the true accelerations across runs.
P0 = np.diag([0.01, 0.01, 0.01, 0.10, 0.10, 0.04, 0.04])

DURATION = 20.0
N_RUNS = 20

# NEES over speed and turn rate only. The full-state number is dominated by
# position and heading, which no sensor measures, so it mostly reports how Q
# was tuned rather than anything about the sensors.
NEES_STATES = [3, 4]

NIS_DOF = 3          # three sensors
NEES_DOF = 2         # two states being checked

# ---------------------------------------------------------------- the arms

# Ordered as the capability ladder, not alphabetically. Each step adds one
# thing to the one before it.
ARMS = [
    "fixed",       # 1  analytic model, hand-tuned constant R
    "plain",       # 2  learned mean, constant R
    "resnet",      # 3  learned mean, deeper -- a capacity control
    "adaptive",    # 4  analytic model, R adapted online from innovations
    "gp",          # 5  learned mean, epistemic from the kernel
    "bhr",         # 6  learned mean and state-dependent R
    "ensemble",    # 7  five heteroscedastic models, spread across them
    "health",      # 8  learned mean and R, conditioned on sensor health
    "combined",    # 9  the health model with an adaptive multiplier on R
]

LABELS = {
    "fixed": "analytic + best const R",
    "plain": "plain network",
    "resnet": "residual network",
    "adaptive": "adaptive R (Mehra)",
    "gp": "Gaussian process",
    "bhr": "heteroscedastic",
    "ensemble": "ensemble of 5",
    "health": "health-conditioned",
    "combined": "combined",
}

# Arms too slow to give the full sweep. Experiments cut their run count and
# say so in the output rather than quietly leaving them out.
#
# The GP was in here until its cost turned out to be a mistake in our own
# code rather than a property of the method: the predictive variance solved
# against a Cholesky factor with a general solver, refactorising a 1500x1500
# matrix on every sigma point of every step. Using the triangular structure
# took it from about 5,900 ms per step to 17. It now runs the same twenty
# runs as everything else.
SLOW = set()


def load_arm(name):
    """Load one arm's measurement model.

    Loaded by full path under a unique module name. Every arm has a file
    called measurement.py and one called train.py, so a plain import would
    hand one arm's module to another -- which happened, silently, and cost an
    afternoon.
    """
    folder = ROOT / "models" / name
    spec = importlib.util.spec_from_file_location(
        "arm_%s_measurement" % name, folder / "measurement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    measure = module.load_measurement_model()

    # Arms that carry sensor health need a wider filter than the rest. Rather
    # than every experiment knowing which those are, the arm says so: if its
    # module declares N_STATES and filter_settings, they are attached here
    # and filter_once uses them. Arms that do not are unaffected.
    if hasattr(module, "N_STATES"):
        measure.n_states = module.N_STATES
        measure.filter_settings = module.filter_settings
    return measure


def available(names=None):
    """Which arms are actually trained and ready to run."""
    ready, missing = [], []
    for name in (names or ARMS):
        try:
            load_arm(name)
            ready.append(name)
        except Exception as problem:
            missing.append((name, type(problem).__name__, str(problem)[:70]))
    return ready, missing


# ---------------------------------------------------------------- one run


def make_run(seed, duration=DURATION):
    """One trajectory, its sensor readings, and the truth to score against.

    The seed drives both the trajectory and the noise, so run 7 is the same
    run 7 for every arm and every condition. That is what makes a run-by-run
    paired comparison possible.
    """
    run = random_run(seed, duration=duration)
    meas = read_sensors(run, seed=seed, dt=DT)

    # The trajectory carries no accelerations, so they are differenced out of
    # the speed and turn rate. The last sample is repeated rather than
    # dropped, to keep every array the same length as the measurements.
    accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
    turn_accel = np.diff(run["turn_rate"], append=run["turn_rate"][-1]) / DT

    truth = np.column_stack([run["x"], run["y"], run["heading"],
                             run["speed"], run["turn_rate"],
                             accel, turn_accel])
    return run, meas, truth


def stack(meas):
    """The three sensor channels as one array the filter can step through."""
    return np.column_stack([meas["left_encoder"], meas["right_encoder"],
                            meas["gyro"]])


def filter_once(measure, meas, truth, R=R_TUNED):
    """Run the filter over one set of readings and score it.

    An arm supplying its own covariance overrides R inside the update, so
    passing R_TUNED here is harmless for those arms and correct for the rest.
    """
    readings = stack(meas)

    # An arm that adapts as it goes has to start each run fresh. Without
    # this, run 2 would begin with whatever run 1 talked itself into, and
    # the twenty runs would stop being independent samples.
    if hasattr(measure, "reset"):
        measure.reset()

    # An arm carrying sensor health needs a wider Q, P0 and start vector.
    # The extra entries are health, which is unknown at the start and has no
    # true value to score against, so the truth array is padded with zeros --
    # correct for a healthy run and simply not scored otherwise, since
    # NEES_STATES only ever asks for speed and turn rate.
    n_states = getattr(measure, "n_states", 7)
    if n_states > 7:
        Q_use, P_use = measure.filter_settings(Q, P0)
        start = np.zeros(n_states)
        start[:7] = truth[0, :7]
        truth = np.column_stack(
            [truth[:, :7], np.zeros((len(truth), n_states - 7))])
    else:
        Q_use, P_use, start = Q, P0, truth[0].copy()

    started = time.time()
    means, covs, innov, S = UKF(Q_use, R, measure=measure).run(
        readings, start, P_use, DT)
    elapsed = time.time() - started

    return {
        "speed_rmse": float(np.sqrt(np.mean((means[:, 3] - truth[:, 3]) ** 2))),
        "turn_rmse": float(np.sqrt(np.mean((means[:, 4] - truth[:, 4]) ** 2))),
        "nis": nis(innov, S),
        "nees": nees(means, covs, truth, states=NEES_STATES),
        "seconds": elapsed,
        "steps": len(readings),
        # The states the filter actually visited. A fault experiment has to
        # ask the model what it believed at THESE states, not at the true
        # ones -- the true states are the same whether a sensor broke or not,
        # so asking there would answer nothing.
        "means": means,
    }


def gather(measure, seeds, fault=None, R=R_TUNED):
    """Filter every seed and pool the results.

    `fault` is an optional function taking the readings dict and returning a
    faulted copy. Healthy is simply not passing one, so there is no flag that
    can be left in the wrong position.
    """
    rows = []
    for seed in seeds:
        run, meas, truth = make_run(seed)
        if fault is not None:
            meas = fault(meas, seed)
        rows.append(filter_once(measure, meas, truth, R))

    return {
        "speed_rmse": np.array([r["speed_rmse"] for r in rows]),
        "turn_rmse": np.array([r["turn_rmse"] for r in rows]),
        "nis": np.concatenate([r["nis"] for r in rows]),
        "nees": np.concatenate([r["nees"] for r in rows]),
        "ms_per_step": 1000 * sum(r["seconds"] for r in rows)
                       / sum(r["steps"] for r in rows),
    }


def best_constant_R(frame=None):
    """The best single covariance a constant-R filter could possibly use.

    A constant cannot track anything, so the most it can do is match the
    average variance over the run. Anything else is worse by construction.

    This matters more than it looks. The hand-tuned R_TUNED above was found by
    covariance matching and came out at 0.1805 for the left encoder, where the
    average is nearer 0.157 -- it absorbs state uncertainty as well as sensor
    noise, so it is inflated as a description of the sensor. Using the
    inflated value as the baseline made the learned covariance look better
    than it is. A learned model should have to beat the best constant
    available, not a convenient one.
    """
    if frame is None:
        frame = pd.read_csv(ROOT / "data" / "robot_data.csv")
    tick = (2 * np.pi / TICKS_PER_TURN) / DT
    quant = tick ** 2 / 12
    average = np.diag([
        frame["left_noise"].pow(2).mean() + quant,
        frame["right_noise"].pow(2).mean() + quant,
        frame["gyro_noise"].pow(2).mean() + frame["gyro_bias"].var(),
    ])
    return R_SCALE * average


def two_moment(values, dof):
    """Chen's criterion: a consistent filter has to get both moments right.

    Average should be dof and spread should be 2*dof. Checking the average
    alone is not enough -- we have already been bitten by a mean of 3.17
    against a target of 3 while the spread was 7.2 against 6, because two
    channels were over-trusted and one under-trusted by amounts that
    cancelled in the sum and compounded in the square.
    """
    values = np.asarray(values)
    return values.mean(), values.var(), float(dof), float(2 * dof)


if __name__ == "__main__":
    print("Filter settings every experiment shares")
    print("  encoder sd  %.4f, %.4f rad/s"
          % (np.sqrt(R_TUNED[0, 0]), np.sqrt(R_TUNED[1, 1])))
    print("  gyro sd     %.5f rad/s" % np.sqrt(R_TUNED[2, 2]))
    print("  %d runs of %.0f s at %d Hz" % (N_RUNS, DURATION, round(1 / DT)))

    ready, missing = available()
    print("\nReady: %s" % (", ".join(ready) if ready else "none"))
    if missing:
        print("\nNot ready:")
        for name, kind, detail in missing:
            print("  %-10s %s: %s" % (name, kind, detail))

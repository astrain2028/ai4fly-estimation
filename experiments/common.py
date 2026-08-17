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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from trajectories import DT, random_run
from sensors import read_sensors
from ukf import UKF, nis, nees

# ---------------------------------------------------------------- settings

# How much the motion model is distrusted.
Q = np.diag([1e-9, 1e-9, 1e-9, 2e-5, 1e-4])

# The hand-tuned sensor noise. Arrived at by covariance matching: filter,
# measure how big the innovations really were, adjust, repeat. This is the
# number a learned covariance has to beat, and it is the only R any arm
# without its own covariance is allowed to use.
R_TUNED = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2])

# How unsure the filter is at the start.
P0 = np.diag([0.01, 0.01, 0.01, 0.10, 0.10])

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
]

LABELS = {
    "fixed": "analytic + tuned R",
    "plain": "plain network",
    "resnet": "residual network",
    "adaptive": "adaptive R (Mehra)",
    "gp": "Gaussian process",
    "bhr": "heteroscedastic",
    "ensemble": "ensemble of 5",
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
    return module.load_measurement_model()


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
    truth = np.column_stack([run["x"], run["y"], run["heading"],
                             run["speed"], run["turn_rate"]])
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

    started = time.time()
    means, covs, innov, S = UKF(Q, R, measure=measure).run(
        readings, truth[0].copy(), P0, DT)
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


def gather(measure, seeds, fault=None):
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
        rows.append(filter_once(measure, meas, truth))

    return {
        "speed_rmse": np.array([r["speed_rmse"] for r in rows]),
        "turn_rmse": np.array([r["turn_rmse"] for r in rows]),
        "nis": np.concatenate([r["nis"] for r in rows]),
        "nees": np.concatenate([r["nees"] for r in rows]),
        "ms_per_step": 1000 * sum(r["seconds"] for r in rows)
                       / sum(r["steps"] for r in rows),
    }


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

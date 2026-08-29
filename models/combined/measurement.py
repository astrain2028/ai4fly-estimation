"""
The health-conditioned model with an adaptive correction on top.

WHY BOTH

The two fault modes hide in different places, and the two methods read
different things.

A bias shifts what a sensor reads. That is visible in the direction of the
innovations, which is what a Kalman update responds to, so a model told how
degraded a sensor is can learn to predict the shifted reading and cancel it.
Measured: severity recovered to about two per cent, and the state estimate
half as wrong at severity three.

Noise inflation leaves the reading centred and widens its spread. The
expected reading is unchanged, so the covariance between a sample point's
health and its predicted measurement is zero, and no first-order update can
move health at all. Measured: 0.29 estimated against a true 3.0. That is not
a training failure, it is arithmetic, and no amount of data or
re-parameterisation reaches it.

What does see it is the size of the innovations rather than their direction --
which is exactly what covariance matching has read since 1970. So the two
methods are complementary by mechanism rather than by luck, and this file is
the two of them in one measurement model.

HOW THEY COMBINE

The obvious arrangement is wrong. Both methods want to set R, and if the
adaptive part estimates R outright it discards everything the model knew
about which channel is bad and how the noise varies with speed.

Instead the adaptive part estimates a MULTIPLIER on the model's covariance:

    R_used = c * R_model(x, m)

which fixes three things at once.

It cannot double-count. If the health model has already explained a bias, the
innovations are small, c stays near 1, and nothing is inflated twice. If a
variance fault the model cannot see is present, the innovations stay large and
c rises.

It converges quickly. Estimating three scalars is a far smaller job than
estimating a covariance from scratch, which is where the classical method's
lag comes from -- about 70 steps in this simulation.

And it keeps the structure. The model supplies which channel and how the noise
varies with state; the multiplier only says "and everything is worse than that
by this factor."

The update is geometric rather than additive, for the reason Chen et al. give
for scoring consistency in logs: a factor of two too large and a factor of two
too small are equally wrong, and an additive rule does not treat them so.
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = _load(ROOT / "models" / "health" / "measurement.py", "health_for_combined")

N_STATES = health.N_STATES
HEALTH_STATES = health.HEALTH_STATES
filter_settings = health.filter_settings

WINDOW = 100          # innovations averaged over, two seconds at 50 Hz
BLEND = 0.05          # how fast the multiplier may move, per update

# The multiplier is allowed to raise the model's covariance and not to lower
# it. That asymmetry is deliberate and was arrived at by measurement.
#
# Allowed to fall, it settles near 0.72 on healthy data, because the learned
# covariance really is about a fifth too large. That correction brings NIS
# from 1.98 to 3.08 -- exactly on target -- and makes the state estimate
# WORSE, from 0.0074 to 0.0097. The over-large covariance was buying accuracy
# while costing calibration, and fixing the calibration gave the accuracy
# back.
#
# So the floor at 1.0 is not a fudge. It says the adaptive layer is here to
# catch what the model did not see, not to second-guess what it did.
LIMITS = (1.0, 25.0)


class Combined:
    """Health-conditioned readings and covariance, scaled by what arrives.

    Used like any other arm. The filter calls `observe` after each update,
    which is where the multiplier learns, and `constrain` to keep health
    non-negative.
    """

    def __init__(self, base, window=WINDOW, blend=BLEND, limits=LIMITS):
        self.base = base
        self.scale = np.ones(3)
        self.history = deque(maxlen=window)
        self.window = window
        self.blend = blend
        self.limits = limits
        self.trace = []

    def __call__(self, states):
        readings, R = self.base(states)
        # The model's covariance is diagonal -- it treats the three sensors as
        # independent, which here they are -- so scaling row i scales exactly
        # the variance of channel i and touches nothing else. A general
        # covariance would need sqrt(c_i) sqrt(c_j) on entry (i, j).
        return readings, R * self.scale[None, :, None]

    def observe(self, innovation, S):
        """Told how the last step went, and adjusts the multiplier.

        A correctly scaled filter has innovations whose squared size matches
        the diagonal of S it predicted. The ratio of the two is therefore the
        factor by which the covariance is wrong, and the multiplier chases it
        -- slowly, and in logs.
        """
        innovation = np.asarray(innovation, dtype=float)
        diag = np.maximum(np.diag(np.asarray(S, dtype=float)), 1e-12)
        self.history.append(innovation ** 2 / diag)

        if len(self.history) == self.window:
            ratio = np.maximum(np.mean(np.array(self.history), axis=0), 1e-6)
            # Geometric step: a ratio of 2 and a ratio of 1/2 move the
            # multiplier the same distance in opposite directions.
            self.scale = np.clip(self.scale * ratio ** self.blend,
                                 self.limits[0], self.limits[1])

        self.trace.append(self.scale.copy())

    def constrain(self, mean):
        mean = np.array(mean, dtype=float)
        mean[HEALTH_STATES] = np.maximum(mean[HEALTH_STATES], 0.0)
        return mean

    def reset(self):
        self.scale = np.ones(3)
        self.history.clear()
        self.trace = []


def load_measurement_model(path=None):
    """The health arm, wrapped in an adaptive multiplier."""
    return Combined(health.load_measurement_model(path))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import Q, P0, best_constant_R
    from faults import apply_fault
    from sensors import read_sensors
    from trajectories import DT, random_run
    from ukf import UKF

    measure = load_measurement_model()
    Q_use, P_use = filter_settings(Q, P0)
    R_const = best_constant_R()

    print("WHAT DOES THE MULTIPLIER DO?\n")
    print("Left encoder faulted. The health model should absorb a bias on")
    print("its own, leaving the multiplier near 1. It cannot see a variance")
    print("fault at all, so the multiplier should have to rise for that one.\n")
    print("%-18s %8s %14s %14s" % ("mode", "severity", "left multiplier",
                                   "health (bias)"))
    print("-" * 58)

    for mode in ["none", "bias", "noise_inflation"]:
        for severity in ([0.0] if mode == "none" else [1.0, 3.0]):
            scales, health_est = [], []
            for seed in range(700, 704):
                run = random_run(seed, duration=20.0)
                meas = read_sensors(run, seed=seed, dt=DT)
                if mode != "none":
                    meas = apply_fault(meas, "left_encoder", mode, severity,
                                       seed=seed, dt=DT)
                readings = np.column_stack([meas["left_encoder"],
                                            meas["right_encoder"],
                                            meas["gyro"]])
                start = np.zeros(N_STATES)
                start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                             run["speed"][0], run["turn_rate"][0]]

                measure.reset()
                means, _, _, _ = UKF(Q_use, R_const, measure=measure).run(
                    readings, start, P_use, DT)
                scales.append(measure.scale[0])
                health_est.append(means[len(means) // 2:, 7].mean())

            print("%-18s %8.2f %14.2f %14.3f"
                  % (mode, severity, np.mean(scales), np.mean(health_est)))

    print("-" * 58)
    print("\nA multiplier near 1 for a bias means the model handled it and the")
    print("adaptive layer had nothing left to do -- which is what stops the")
    print("two mechanisms from correcting the same fault twice.")

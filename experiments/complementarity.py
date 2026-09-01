"""
Two mechanisms, each blind exactly where the other sees.

THE CLAIM

A Kalman update reads the direction of its innovations. A bias shifts a
reading, so it lives in that direction, and a measurement model told how
degraded a sensor is can learn to predict the shifted value and cancel it.
Covariance matching reads the size of the innovations instead. Noise
inflation leaves the reading centred and widens its spread, so it lives in
that size and nowhere else.

Neither mechanism can reach the other's fault, and the reason is arithmetic
rather than engineering. Where a fault leaves the expected reading unchanged,
the covariance between a sample point's health and its predicted measurement
is identically zero, and there is no first-order path for the update to move
health along. Where a fault is a pure offset, discounting the sensor is the
wrong response anyway: the reading is as informative as it ever was once the
offset is known, and throwing it away costs accuracy.

So this is not a horse race. It is a map of which mechanism owns which
failure, and a demonstration that one filter can carry both.

WHAT IS COMPARED

    analytic + constant R    no response to anything; the control
    adaptive R               reads innovation size only
    health-conditioned       reads innovation direction only
    combined                 the health model with an adaptive multiplier
                             on the covariance it predicts

The combined arm should match whichever of the middle two is better in each
cell. If it does, the two mechanisms compose; if it is worse than both
somewhere, they interfere and the arrangement is wrong.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import NEES_STATES, NIS_DOF, P0, Q, best_constant_R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import sensors
from faults import apply_fault
from trajectories import DT, random_run
from ukf import UKF, expected_readings, nis as nis_of, nees as nees_of


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = _load(ROOT / "models" / "health" / "measurement.py", "health_c")
combined = _load(ROOT / "models" / "combined" / "measurement.py", "combined_c")
adaptive = _load(ROOT / "models" / "adaptive" / "measurement.py", "adaptive_c")

SEVERITIES = [0.0, 1.0, 2.0, 3.0]
EVAL_RUNS = 8


def score(measure, wide, mode, severity, R, n_runs=EVAL_RUNS):
    """Filter runs with one channel degraded, and score the state estimate."""
    speed, nis_all = [], []
    for seed in range(800, 800 + n_runs):
        run = random_run(seed, duration=20.0)
        meas = sensors.read_sensors(run, seed=seed, dt=DT)
        if severity > 0:
            meas = apply_fault(meas, "left_encoder", mode, severity,
                               seed=seed, dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])

        if wide:
            Q_use, P_use = health.filter_settings(Q, P0)
            start = np.zeros(health.N_STATES)
        else:
            Q_use, P_use = Q, P0
            start = np.zeros(7)
        start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                     run["speed"][0], run["turn_rate"][0]]

        if hasattr(measure, "reset"):
            measure.reset()
        means, covs, innov, S = UKF(Q_use, R, measure=measure).run(
            readings, start, P_use, DT)

        speed.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
        nis_all.append(nis_of(innov, S).mean())

    return float(np.mean(speed)), float(np.mean(nis_all))


def main():
    R = best_constant_R()
    arms = [
        ("analytic + constant R", expected_readings, False),
        ("adaptive R", adaptive.AdaptiveR(R), False),
        ("health-conditioned", health.load_measurement_model(), True),
        ("combined", combined.load_measurement_model(), True),
    ]

    print("COMPLEMENTARITY\n")
    print("Left encoder degraded for the whole run. Speed error in m/s,")
    print("with mean NIS underneath -- a filter that has handled a fault")
    print("keeps NIS near %.1f, one that has not lets it climb.\n" % NIS_DOF)

    results = {}
    for mode in ["bias", "noise_inflation"]:
        print("  %s" % mode)
        print("  %-24s" % "severity"
              + "".join("%11.1f" % s for s in SEVERITIES))
        print("  " + "-" * (24 + 11 * len(SEVERITIES)))

        for label, measure, wide in arms:
            errors, nises = [], []
            for severity in SEVERITIES:
                e, n = score(measure, wide, mode, severity, R)
                errors.append(e)
                nises.append(n)
            results[(mode, label)] = errors
            print("  %-24s" % label + "".join("%11.4f" % v for v in errors))
            print("  %-24s" % "" + "".join("%11.2f" % v for v in nises))
        print()

    print("  " + "-" * (24 + 11 * len(SEVERITIES)))
    print("\nWHO OWNS WHICH FAULT\n")
    print("  %-20s %14s %14s" % ("", "bias", "noise inflation"))
    for label in ["adaptive R", "health-conditioned", "combined"]:
        row = []
        for mode in ["bias", "noise_inflation"]:
            control = results[(mode, "analytic + constant R")][-1]
            arm = results[(mode, label)][-1]
            row.append(100.0 * (1.0 - arm / control))
        print("  %-20s %13.0f%% %13.0f%%" % (label, row[0], row[1]))
    print("\n  Improvement over the control at severity 3. Positive is")
    print("  better. The claim is that the two middle rows are strong in")
    print("  opposite columns, and that the last row is strong in both.")


if __name__ == "__main__":
    main()

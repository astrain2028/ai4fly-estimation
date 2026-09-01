"""
Two broken sensors, and a fault that arrives partway through.

Every fault experiment in this project so far has broken one channel, from
the first sample to the last. Both of those are conveniences, and each hides
a question the formulation makes a claim about.

HOW MANY FAILURES CAN THREE SENSORS SURVIVE?

The robot carries two encoders and a gyro, and they constrain two quantities:
speed and turn rate. Three measurements for two unknowns leaves exactly one
spare, and that spare is the whole basis for noticing anything. With one
channel lying, the other two still agree with each other and disagree with
it, so the odd one out is identifiable. With two lying, the remaining sensor
has nobody to agree with, and "two sensors are wrong" and "one sensor is
right" describe the same readings.

So the expectation is a cliff rather than a slope: single faults handled,
double faults not, and no amount of model quality changing that because the
information is not present. Worth testing rather than asserting, because a
result that contradicts it would mean the redundancy argument in the README
is wrong.

WHAT HAPPENS WHEN A FAULT ARRIVES LATE?

Training severity is constant within a run, so the health model has only ever
seen sensors that were already broken when the run began. A sensor that fails
at ten seconds is a different situation: the filter has spent half the run
becoming confident, and now has to change its mind.

That bears directly on the classical comparison. Covariance matching needs
its window to fill before it reacts -- about 70 steps, measured -- and a
model that reads health from the state should not need to wait at all. If it
does, the lag advantage claimed for the learned approach is not real.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import NIS_DOF, P0, Q, best_constant_R, load_arm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import sensors
from faults import apply_fault
from trajectories import DT, random_run
from ukf import UKF, nis as nis_of

ARMS = ["fixed", "adaptive", "health", "combined"]
LABELS = {"fixed": "analytic + constant R", "adaptive": "adaptive R",
          "health": "health-conditioned", "combined": "combined"}

CHANNELS = ["left_encoder", "right_encoder", "gyro"]
SEVERITY = 2.0
EVAL_RUNS = 8
ONSET = 0.5              # halfway through, for the late-arrival case


def filtered(measure, meas, run):
    """Run one filtered pass, returning the estimates and the innovations."""
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    n_states = getattr(measure, "n_states", 7)
    if n_states > 7:
        Q_use, P_use = measure.filter_settings(Q, P0)
        start = np.zeros(n_states)
    else:
        Q_use, P_use = Q, P0
        start = np.zeros(7)
    start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                 run["speed"][0], run["turn_rate"][0]]

    if hasattr(measure, "reset"):
        measure.reset()
    return UKF(Q_use, best_constant_R(), measure=measure).run(
        readings, start, P_use, DT)


def break_channels(meas, channels, severity, seed, onset=0.0):
    """Degrade one or more channels, optionally only after `onset`.

    A late fault is applied to the whole run and then the healthy readings
    are put back over the first part, so the two halves come from the same
    draw and differ only in whether the fault is present.
    """
    broken = dict(meas)
    for channel in channels:
        faulted = apply_fault(broken, channel, "bias", severity,
                              seed=seed, dt=DT)[channel]
        if onset > 0:
            cut = int(onset * len(faulted))
            faulted = np.concatenate([meas[channel][:cut], faulted[cut:]])
        broken[channel] = faulted
    return broken


def score(measure, channels, severity, onset=0.0, n_runs=EVAL_RUNS):
    """Speed error and mean NIS, over the part of the run that is faulted."""
    speed, nis_all = [], []
    for seed in range(1000, 1000 + n_runs):
        run = random_run(seed, duration=20.0)
        meas = sensors.read_sensors(run, seed=seed, dt=DT)
        if channels:
            meas = break_channels(meas, channels, severity, seed, onset)

        means, _, innov, S = filtered(measure, meas, run)

        # Only the faulted stretch is scored, so a late fault is not diluted
        # by the healthy half that preceded it.
        cut = int(onset * len(means))
        speed.append(np.sqrt(np.mean((means[cut:, 3]
                                      - run["speed"][cut:]) ** 2)))
        nis_all.append(nis_of(innov, S)[cut:].mean())
    return float(np.mean(speed)), float(np.mean(nis_all))


def main():
    arms = [(LABELS[name], load_arm(name)) for name in ARMS]

    print("HOW MANY BROKEN SENSORS CAN IT SURVIVE?\n")
    print("Bias faults at severity %.1f. Three sensors constrain two" % SEVERITY)
    print("quantities, so one spare: a single failure should be identifiable")
    print("and a double failure should not be.\n")

    conditions = [
        ("none", []),
        ("left encoder", ["left_encoder"]),
        ("gyro", ["gyro"]),
        ("both encoders", ["left_encoder", "right_encoder"]),
        ("encoder + gyro", ["left_encoder", "gyro"]),
    ]

    print("  %-24s" % "" + "".join("%16s" % c[0][:15] for c in conditions))
    print("  " + "-" * (24 + 16 * len(conditions)))
    for label, measure in arms:
        row = [score(measure, channels, SEVERITY)[0]
               for _, channels in conditions]
        print("  %-24s" % label + "".join("%16.4f" % v for v in row))
    print("  " + "-" * (24 + 16 * len(conditions)))
    print("\n  If the last two columns are far worse than the middle two for")
    print("  every arm alike, the limit is the sensor suite and not the")
    print("  method, which is what the redundancy argument claims.")

    print("\n\nWHAT IF THE FAULT ARRIVES HALFWAY THROUGH?\n")
    print("Left encoder, same severity, scored over the faulted half only.")
    print("Covariance matching has to fill a window before it reacts; a model")
    print("that reads health from the state should not.\n")
    print("  %-24s %14s %14s %12s"
          % ("", "from the start", "from halfway", "difference"))
    print("  " + "-" * 66)
    for label, measure in arms:
        early, _ = score(measure, ["left_encoder"], SEVERITY, onset=0.0)
        late, _ = score(measure, ["left_encoder"], SEVERITY, onset=ONSET)
        print("  %-24s %14.4f %14.4f %11.0f%%"
              % (label, early, late, 100.0 * (late / early - 1.0)))
    print("  " + "-" * 66)
    print("\n  A large positive difference means the arm was caught out by the")
    print("  onset and never fully recovered within the run. Near zero means")
    print("  it adapted quickly enough for the timing not to matter.")


if __name__ == "__main__":
    main()

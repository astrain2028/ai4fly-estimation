"""
Which of the tuned constants actually matter.

WHY THIS EXISTS

There are now eleven or so numbers in this project that somebody chose. Some
were measured, some were searched, and some were picked because they seemed
reasonable and never revisited. Individually each has a justification written
next to it. Collectively they are the first thing a reviewer asks about, and
"we tuned it" is not an answer when there are eleven of them.

The cheap version of the answer is not a sweep. It is one question per
constant: does halving or doubling it change any conclusion? A constant whose
value can move by a factor of four without moving the result needs no defence
at all, and knowing which ones those are tells you where the remaining
defence has to go.

WHAT IS AND IS NOT COVERED

Constants that only affect inference are testable here, because the arm can
be rebuilt with a different value in a second.

Two are not. ONSET_FRACTION and MODE_PRESENT in robot/make_faulted.py shape
the training set, so changing either means regenerating five hundred runs and
retraining, about half an hour each. They are listed at the end as untested
rather than quietly omitted, because an unlisted constant reads as one nobody
thought about.

HOW TO READ IT

The "spread" column is the range of the metric across the three values, as a
fraction of the nominal. Under about 5 per cent means the constant is not
doing much and the exact value is not load-bearing. Large means it is, and
that one needs either a real sweep or an honest sentence in the paper.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import common
from common import P0, Q, best_constant_R, load_arm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from faults import apply_fault
from sensors import read_sensors
from trajectories import DT, random_run
from ukf import UKF, nis as nis_of

SEEDS = range(2000, 2005)
CHANNEL = "left_encoder"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health_m = _load(ROOT / "models" / "health" / "measurement.py", "sens_health")
combined_m = _load(ROOT / "models" / "combined" / "measurement.py",
                   "sens_combined")


def evaluate(arm, settings, mode="bias", severity=2.0, R=None):
    """Speed error and mean NIS for one arm under one fault."""
    Q_use, P_use = settings
    R = best_constant_R() if R is None else R
    errors, nis_all = [], []

    for seed in SEEDS:
        run = random_run(seed, duration=20.0)
        meas = read_sensors(run, seed=seed, dt=DT)
        if severity > 0:
            meas = apply_fault(meas, CHANNEL, mode, severity, seed=seed, dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])
        start = np.zeros(len(Q_use))
        start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                     run["speed"][0], run["turn_rate"][0]]

        if hasattr(arm, "reset"):
            arm.reset()
        if callable(arm):
            means, _, innov, S = UKF(Q_use, R, measure=arm).run(
                readings, start, P_use, DT)
        else:
            means, _, innov, S = arm.run(readings, start, P_use, DT)

        errors.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
        nis_all.append(nis_of(innov, S).mean())
    return float(np.mean(errors)), float(np.mean(nis_all))


def report(name, nominal, results):
    """One row: three values, and how far apart the outcomes were."""
    speeds = [r[0] for r in results]
    nises = [r[1] for r in results]
    spread = (max(speeds) - min(speeds)) / max(np.mean(speeds), 1e-12)
    flag = "" if spread < 0.05 else ("  <-- matters" if spread > 0.15 else "")
    print("  %-24s %10.4g %9.4f %9.4f %9.4f %9.0f%%%s"
          % (name, nominal, speeds[0], speeds[1], speeds[2],
             100 * spread, flag))
    return spread


def main():
    print("DOES HALVING OR DOUBLING IT CHANGE ANYTHING?\n")
    print("%d runs per setting, left encoder biased at severity 2.0 unless"
          % len(list(SEEDS)))
    print("noted. Speed error in m/s at half, nominal and double.\n")
    print("  %-24s %10s %9s %9s %9s %10s"
          % ("constant", "nominal", "half", "nominal", "double", "spread"))
    print("  " + "-" * 84)

    spreads = {}

    # --- health: how freely the filter may move its health estimate --------
    nominal = health_m.HEALTH_PROCESS_NOISE
    results = []
    for value in [nominal / 2, nominal, nominal * 2]:
        health_m.HEALTH_PROCESS_NOISE = value
        arm = health_m.load_measurement_model()
        arm.n_states = health_m.N_STATES
        results.append(evaluate(arm, health_m.filter_settings(Q, P0)))
    health_m.HEALTH_PROCESS_NOISE = nominal
    spreads["HEALTH_PROCESS_NOISE"] = report("HEALTH_PROCESS_NOISE", nominal,
                                             results)

    nominal = health_m.HEALTH_START_SPREAD
    results = []
    for value in [nominal / 2, nominal, nominal * 2]:
        health_m.HEALTH_START_SPREAD = value
        arm = health_m.load_measurement_model()
        results.append(evaluate(arm, health_m.filter_settings(Q, P0)))
    health_m.HEALTH_START_SPREAD = nominal
    spreads["HEALTH_START_SPREAD"] = report("HEALTH_START_SPREAD", nominal,
                                            results)

    # --- combined: the multiplier's three knobs, on a variance fault, ------
    # which is the only condition where the multiplier does anything at all.
    settings = combined_m.filter_settings(Q, P0)
    base = combined_m.health.load_measurement_model()

    for label, attr, values in [
            ("WINDOW (noise fault)", "window",
             [combined_m.WINDOW // 2, combined_m.WINDOW, combined_m.WINDOW * 2]),
            ("BLEND (noise fault)", "blend",
             [combined_m.BLEND / 2, combined_m.BLEND, combined_m.BLEND * 2]),
            ("LIMITS high (noise)", "hi",
             [combined_m.LIMITS[1] / 2, combined_m.LIMITS[1],
              combined_m.LIMITS[1] * 2])]:
        results = []
        for value in values:
            if attr == "window":
                arm = combined_m.Combined(base, window=int(value))
            elif attr == "blend":
                arm = combined_m.Combined(base, blend=value)
            else:
                arm = combined_m.Combined(base, limits=(1.0, value))
            results.append(evaluate(arm, settings, mode="noise_inflation",
                                    severity=3.0))
        spreads[label] = report(label, values[1], results)

    # --- the shared filter setting every arm inherits ----------------------
    nominal = common.R_SCALE
    results = []
    for value in [nominal / 2, nominal, nominal * 2]:
        common.R_SCALE = value
        arm = load_arm("fixed")
        results.append(evaluate(arm, (Q, P0), R=common.best_constant_R()))
    common.R_SCALE = nominal
    spreads["R_SCALE"] = report("R_SCALE", nominal, results)

    print("  " + "-" * 84)
    print("\n  Spread is the range of speed error across the three settings,")
    print("  as a fraction of their mean. Under 5 per cent means the exact")
    print("  value is not load-bearing and needs no defending.")

    loud = [k for k, v in spreads.items() if v > 0.15]
    print("\n\nWHAT STILL NEEDS DEFENDING\n")
    if loud:
        for name in loud:
            print("  %-24s spread %.0f%%" % (name, 100 * spreads[name]))
        print("\n  These want a real sweep, or a sentence in the paper saying")
        print("  what they trade off and why the chosen value sits there.")
    else:
        print("  None of the tested constants moved the result by more than")
        print("  15 per cent across a factor of four. That is worth stating:")
        print("  the conclusions do not rest on the tuning.")

    print("\n\nNOT TESTED HERE\n")
    print("  ONSET_FRACTION and MODE_PRESENT in robot/make_faulted.py shape")
    print("  the training set rather than inference, so moving either means")
    print("  regenerating 500 runs and retraining -- about half an hour per")
    print("  value. They are listed so that nobody mistakes an untested")
    print("  constant for one that was checked.")
    print("\n  BANK_SEVERITY and WEIGHT_FLOOR in models/mmae belong to the")
    print("  baseline rather than to the contribution, and BANK_SEVERITY is")
    print("  already swept by construction: the mmae self-test evaluates")
    print("  below, at, and above it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

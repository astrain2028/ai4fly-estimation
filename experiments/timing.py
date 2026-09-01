"""
How long a filter step actually takes, measured in a way worth quoting.

Every cost claim in this project has so far come from wall-clock timing
inside a scoring run, and those numbers are not trustworthy. The Gaussian
process has been recorded at 23, 29, 96 and 567 ms per step on the same
machine on the same day. The arithmetic was right each time; the measurement
was not, because it included whatever else the machine was doing.

That matters more here than in most projects. The whole argument for a
deterministic Bayesian method over an ensemble is that it costs less, and an
unquotable cost is an unmade argument.

WHAT THIS DOES DIFFERENTLY

Warmup, so the first call's import and allocation costs are not counted.
Repeats, so a single unlucky window does not decide the answer. And the
median rather than the mean, because the distribution has a long right tail
made of scheduler interruptions, which say nothing about the method.

The spread between the fastest and slowest repeat is reported alongside. If
that spread is wide the number should not be quoted at all, whatever the
median says.

WHAT IS TIMED

One filter step: predict, evaluate the measurement model at every sigma
point, rebuild, update. That is the quantity a real-time budget constrains --
at 50 Hz there are 20 ms, and the flight stack needs most of them.

Sigma points scale with the state, so the arms that carry health are doing
more filter arithmetic as well as more network: 15 points for seven states,
27 for thirteen. Both effects are in the number, which is correct, since both
are paid on the vehicle.
"""

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import ARMS, LABELS, P0, Q, best_constant_R, load_arm, make_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from trajectories import DT
from ukf import UKF

WARMUP = 20            # steps discarded before timing starts
STEPS = 200            # steps per repeat
REPEATS = 7            # repeats, so the median has something to sit in
BUDGET_MS = 1000 * DT  # 20 ms at 50 Hz


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def time_arm(measure, wide_module=None, R=None):
    """Median milliseconds per filter step, and the spread across repeats."""
    run, meas, truth = make_run(0)
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])

    if wide_module is not None:
        Q_use, P_use = wide_module.filter_settings(Q, P0)
        start = np.zeros(wide_module.N_STATES)
    else:
        Q_use, P_use = Q, P0
        start = np.zeros(7)
    start[:5] = truth[0, :5]

    per_repeat = []
    for repeat in range(REPEATS + 1):        # the first is warmup
        if hasattr(measure, "reset"):
            measure.reset()
        mean, cov = start.copy(), P_use.copy()

        for _ in range(WARMUP):
            mean, cov = UKF(Q_use, R, measure=measure).predict(mean, cov, DT)

        f = UKF(Q_use, R, measure=measure)
        started = time.perf_counter()
        for k in range(STEPS):
            mean, cov = f.predict(mean, cov, DT)
            mean, cov, innovation, S = f.update(mean, cov,
                                                readings[k % len(readings)])
            if hasattr(f.measure, "observe"):
                f.measure.observe(innovation, S)
            if hasattr(f.measure, "constrain"):
                mean = f.measure.constrain(mean)
        elapsed = time.perf_counter() - started

        if repeat > 0:                        # discard the warmup repeat
            per_repeat.append(1000.0 * elapsed / STEPS)

    per_repeat = np.array(per_repeat)
    return float(np.median(per_repeat)), float(per_repeat.min()), \
        float(per_repeat.max())


def main():
    R = best_constant_R()

    entries = []
    for name in ARMS:
        try:
            entries.append((LABELS[name], load_arm(name), None, 7))
        except Exception:
            pass

    for folder, label in [("health", "health-conditioned"),
                          ("combined", "combined")]:
        path = ROOT / "models" / folder / "measurement.py"
        if not path.exists():
            continue
        try:
            module = _load(path, "timing_" + folder)
            entries.append((label, module.load_measurement_model(), module,
                            module.N_STATES))
        except Exception as problem:
            print("  (%s unavailable: %s)" % (label, str(problem)[:60]))

    print("COST PER FILTER STEP")
    print("%d steps per repeat, %d repeats, %d discarded as warmup.\n"
          % (STEPS, REPEATS, WARMUP))
    print("%-24s %7s %10s %10s %10s %9s"
          % ("", "states", "sigma pts", "median ms", "spread", "of budget"))
    print("-" * 76)

    for label, measure, module, n_states in entries:
        median, low, high = time_arm(measure, module, R)
        spread = high - low
        flag = " *" if spread > 0.5 * median else ""
        print("%-24s %7d %10d %10.3f %9.3f%s %8.0f%%"
              % (label, n_states, 2 * n_states + 1, median, spread, flag,
                 100.0 * median / BUDGET_MS))

    print("-" * 76)
    print("%-24s %7s %10s %10.1f" % ("budget at 50 Hz", "", "", BUDGET_MS))
    print("\n* means the fastest and slowest repeat differed by more than half")
    print("  the median, which is a machine-load problem rather than a")
    print("  property of the arm. Do not quote a starred number.")


if __name__ == "__main__":
    main()

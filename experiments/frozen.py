"""
Does a three-line check beat the learned machinery on a frozen sensor?

Motivation
----------

``stuck`` is the fault mode that falls outside the moment-order taxonomy. A
frozen reading neither shifts the mean predictably nor widens the spread; it
sits at one plausible value while the true quantity moves away from it. The
health-conditioned model was worse than doing nothing on it in early
measurements and its best result since is modest, and the README lists it as
open with a specific suggestion: a variance check over the last few readings
may beat anything learned.

That is the kind of claim that should be measured rather than left as a
sentence, because if it is true the honest recommendation for that fault is
three lines of numpy, and if it is false the sentence should go.

The detector
------------

A stuck sensor repeats its last reading exactly, so the variance of its last
``WINDOW`` readings is exactly zero. Real sensors carry noise, so a healthy
channel's window variance is never zero. The test is therefore

    frozen = var(last WINDOW readings) < THRESHOLD

with the threshold far below any healthy variance and far above float
rounding. No training, no state, no model.

What to do about it
-------------------

When a channel is frozen it carries no information, so its measurement
variance is inflated by a large factor and the filter stops listening. That
is done by wrapping whichever measurement model is in use, so the same
detector sits in front of the analytic model and in front of layered alike.
The comparison is then four arms on the same seeds:

    analytic                     the baseline
    analytic + frozen detector   three lines added to the baseline
    layered                      the best learned arm, on its own
    layered + frozen detector    the same three lines added to it

If the detector closes most of the gap on its own, the fault does not need a
learned model. If layered still adds something on top of it, both belong.

Seeds and conditions match experiments/bakeoff.py so the numbers sit
alongside its tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from common import P0, Q, best_constant_R, load_arm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from faults import apply_fault
from trajectories import DT, random_run
from sensors import read_sensors
from ukf import UKF, expected_readings, nis as nis_of

SEEDS = range(2000, 2008)
FRACTIONS = [0.3, 0.5]          # of the run frozen, counting from the end
CHANNEL = "left_encoder"

# Readings compared for zero variance. Five steps is a tenth of a second at
# 50 Hz. The encoders are quantised, so at low wheel speed a healthy channel
# can legitimately repeat a tick count for several steps, and a short window
# mistakes that for a freeze; a longer one trades detection delay for fewer
# false alarms. Settable from the command line to measure that trade:
#
#     python experiments/frozen.py --window=25
WINDOW = 5
for _arg in sys.argv[1:]:
    if _arg.startswith("--window="):
        WINDOW = int(_arg.split("=", 1)[1])
THRESHOLD = 1e-12
INFLATE = 1e6                   # a frozen channel's variance, times this

RESULTS = ROOT / "results"


def frozen_mask(readings, window=WINDOW, threshold=THRESHOLD):
    """True where a channel's last ``window`` readings have zero variance."""
    n, m = readings.shape
    mask = np.zeros((n, m), dtype=bool)
    for k in range(window - 1, n):
        block = readings[k - window + 1:k + 1]
        mask[k] = block.var(axis=0) < threshold
    return mask


class Gated:
    """Any measurement model, with frozen channels switched off.

    ``self.frozen`` is set from outside before each update, from the mask the
    detector computed on the raw readings -- the measurement model itself
    only ever sees sigma points, never readings, so the check cannot live
    inside it.
    """

    def __init__(self, base, n_channels, R_const):
        self.base = base
        self.n_channels = n_channels
        self.R_const = np.asarray(R_const, dtype=float)
        self.frozen = np.zeros(n_channels, dtype=bool)

    def __call__(self, states):
        out = self.base(states)
        if isinstance(out, tuple):
            readings, R = out
        else:
            readings = np.asarray(out)
            R = np.repeat(self.R_const[None, :, :], len(readings), axis=0)
        R = np.array(R, dtype=float)
        for i in np.flatnonzero(self.frozen):
            R[:, i, i] *= INFLATE
        return readings, R

    def observe(self, innovation, S):
        if hasattr(self.base, "observe"):
            self.base.observe(innovation, S)

    def constrain(self, mean):
        if hasattr(self.base, "constrain"):
            return self.base.constrain(mean)
        return mean

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.frozen[:] = False


def run_filter(arm, readings, truth, n_states, gated_mask=None):
    """One run, stepped by hand so the frozen mask can be applied per step."""
    if n_states > 7:
        Q_use, P_use = arm.filter_settings(Q, P0) if hasattr(
            arm, "filter_settings") else arm.base.filter_settings(Q, P0)
    else:
        Q_use, P_use = Q, P0
    start = np.zeros(n_states)
    start[:7] = truth[0, :7]

    if hasattr(arm, "reset"):
        arm.reset()
    f = UKF(Q_use, best_constant_R(), measure=arm)
    mean, cov = start.copy(), P_use.copy()
    means = np.zeros((len(readings), n_states))
    nis_all = np.zeros(len(readings))

    for k in range(len(readings)):
        if gated_mask is not None:
            arm.frozen = gated_mask[k]
        mean, cov = f.predict(mean, cov, DT)
        mean, cov, innovation, S = f.update(mean, cov, readings[k])
        if hasattr(arm, "observe"):
            arm.observe(innovation, S)
        if hasattr(arm, "constrain"):
            mean = arm.constrain(mean)
        means[k] = mean
        nis_all[k] = innovation @ np.linalg.solve(S, innovation)
    return means, nis_all


def score(make_arm, n_states, gate, fraction, seeds=SEEDS):
    errors, nis_vals, detected = [], [], []
    for seed in seeds:
        run = random_run(seed, duration=20.0)
        meas = read_sensors(run, seed=seed, dt=DT)
        if fraction > 0:
            meas = apply_fault(meas, CHANNEL, "stuck", fraction, seed=seed,
                               dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])
        accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
        turn_accel = np.diff(run["turn_rate"],
                             append=run["turn_rate"][-1]) / DT
        truth = np.column_stack([run["x"], run["y"], run["heading"],
                                 run["speed"], run["turn_rate"],
                                 accel, turn_accel])

        arm = make_arm()
        mask = frozen_mask(readings) if gate else None
        means, nis_run = run_filter(arm, readings, truth, n_states, mask)

        # Score the frozen stretch only, so a short freeze is not diluted by
        # the healthy majority of the run.
        cut = int((1.0 - fraction) * len(readings)) if fraction > 0 else 0
        errors.append(np.sqrt(np.mean((means[cut:, 3] - truth[cut:, 3]) ** 2)))
        nis_vals.append(nis_run[cut:].mean())
        if gate and fraction > 0:
            truly = np.zeros(len(readings), dtype=bool)
            truly[cut + WINDOW - 1:] = True
            hit = (mask[:, 0] & truly).sum() / max(truly.sum(), 1)
            false = (mask[:, 0] & ~truly).sum()
            detected.append((hit, false))
    out = {"speed": float(np.mean(errors)), "nis": float(np.mean(nis_vals))}
    if detected:
        out["hit"] = float(np.mean([d[0] for d in detected]))
        out["false"] = float(np.mean([d[1] for d in detected]))
    return out


def main():
    R = best_constant_R()
    try:
        layered_module = load_arm("layered")
    except Exception as problem:
        print("layered unavailable: %s" % str(problem)[:70])
        return 1

    def analytic():
        return Gated(expected_readings, 3, R)

    def layered():
        arm = load_arm("layered")
        g = Gated(arm, 3, R)
        g.filter_settings = arm.filter_settings
        return g

    arms = [("analytic", analytic, 7, False),
            ("analytic + frozen detector", analytic, 7, True),
            ("layered", layered, 13, False),
            ("layered + frozen detector", layered, 13, True)]

    print("A FROZEN SENSOR, AND A THREE-LINE DETECTOR\n")
    print("Left encoder frozen for the last fraction of the run, %d seeds,"
          % len(list(SEEDS)))
    print("scored over the frozen stretch only. Speed error in m/s.\n")

    conditions = [("healthy", 0.0)] + [("stuck %.1f" % f, f) for f in FRACTIONS]
    print("  %-28s" % "" + "".join("%12s" % c[0] for c in conditions)
          + "%14s" % "detector")
    print("  " + "-" * (28 + 12 * len(conditions) + 14))

    records = []
    for label, make, n_states, gate in arms:
        row = "  %-28s" % label
        note = ""
        for name, fraction in conditions:
            r = score(make, n_states, gate, fraction)
            row += "%12.4f" % r["speed"]
            records.append({"arm": label, "condition": name,
                            "fraction": fraction, **r})
            if gate and fraction == FRACTIONS[-1]:
                note = "%3.0f%% hit, %.1f false" % (100 * r["hit"], r["false"])
        print(row + "%14s" % note)
    print("  " + "-" * (28 + 12 * len(conditions) + 14))

    print("\n  'detector' is how much of the frozen stretch the check caught,")
    print("  and how many healthy steps it flagged per run. A stuck reading")
    print("  repeats exactly, so the check should be near 100% with no false")
    print("  alarms; the first %d steps of a freeze are undetectable by"
          % (WINDOW - 1))
    print("  construction, since the window has to fill.")

    frame = pd.DataFrame(records)
    base = frame[frame["arm"] == "analytic"].set_index("condition")["speed"]
    print("\n\nWHAT EACH ARM BUYS ON THE WORST CASE, stuck %.1f\n"
          % FRACTIONS[-1])
    worst = "stuck %.1f" % FRACTIONS[-1]
    ref = float(base[worst])
    for label, *_ in arms:
        v = float(frame[(frame["arm"] == label)
                        & (frame["condition"] == worst)]["speed"].iloc[0])
        print("  %-28s %8.4f   %+5.0f%% vs analytic" % (label, v,
                                                         100 * (v / ref - 1)))
    print("\n  If the detector on its own recovers most of the gap, this fault")
    print("  wants three lines of numpy and not a learned model. If layered")
    print("  still adds on top of it, both belong.")

    RESULTS.mkdir(exist_ok=True)
    stem = "frozen" if WINDOW == 5 else "frozen_w%d" % WINDOW
    frame.to_csv(RESULTS / ("%s.csv" % stem), index=False, float_format="%.6g")
    print("\nWrote %d rows to results/%s.csv" % (len(frame), stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())

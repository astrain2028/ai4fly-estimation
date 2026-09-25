"""
Does the complementarity crossing survive a nonlinear measurement map?

Motivation
----------

The crossing -- health conditioning owning faults that shift a reading,
covariance matching owning faults that only widen it -- was derived from
dh/dm before it was measured, and the derivation says nothing about whether h
is linear. A first-moment fault moves the expected reading whatever shape the
map has; a second-moment fault does not. So the taxonomy ought to transfer,
and if it does not, the argument is narrower than it claims and the ground
robot was carrying it on a special case.

That makes this a genuine test rather than a replication. Three ways it could
fail:

    the crossing does not appear      the split was a property of a linear
                                      map, and the derivation is missing
                                      something

    it appears but is much weaker     the nonlinearity blurs the mechanisms
                                      into each other, which would matter for
                                      any real vehicle

    it appears as it did              the taxonomy is about moment order and
                                      not about the map, which is what the
                                      derivation says

Differences from the ground robot
---------------------------------

More redundancy. Nine channels constrain six states, so three are spare, where
the robot had exactly one. That could plausibly help every arm equally, or it
could help the classical arms more -- covariance matching has more disagreeing
sensors to work with. Worth watching rather than assuming.

Faults are per device, so a fault takes three channels with it rather than
one. That is a bigger perturbation relative to the suite than the robot's
single broken encoder, and it cuts the other way.

Metrics
-------

Attitude error over roll and pitch, in degrees. Yaw is left out because it
wraps, and a wrapped residual would dominate any average without saying
anything about the filter. NIS against a target of nine, one per channel.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import measurement as quad
from faults import MOMENT, TRAINED_ON, apply_fault
from sensors import read_sensors, stack
from trajectories import DT, random_run, truth_matrix

SEEDS = range(400, 408)
SEVERITIES = [0.5, 1.5, 3.0]
MODES = ["bias", "noise_inflation"]

# Which device breaks. The accelerometer by default, and the magnetometer is
# the informative alternative: the accelerometer is the primary observer of
# roll and pitch, the states that are scored, while the magnetometer mostly
# informs yaw, which is not. If the accuracy crossing fails on the
# accelerometer but holds on the magnetometer, the failure was about breaking
# the sensor the estimate depends on, not about the map being nonlinear -- and
# the two are confounded on the accelerometer alone.
DEVICE = "accel"
for _arg in sys.argv[1:]:
    if _arg.startswith("--device="):
        DEVICE = _arg.split("=", 1)[1]

RESULTS = ROOT / "results"

# The four arms the argument is about, matching the robot's bakeoff so the two
# tables can be read against each other.
#
#   analytic   the hand-written map with the best constant covariance
#   adaptive   the same map, covariance matched online from innovations
#   health     the learned map and covariance, conditioned on health
#   layered    health plus the additive residual
#
# The learned arms carry health and need the wider filter; the classical ones
# do not. That is handled per arm rather than by a flag, because a flag left in
# the wrong position is exactly how the robot's timing experiment broke.
#
# The adaptive arm is models/adaptive, the estimator the ground robot uses,
# given this vehicle's measurement function. An earlier version of this file
# carried its own copy under the same label, and the copy was not Mehra's
# estimator at all: it scaled R multiplicatively by the ratio of observed to
# predicted innovation size, which is the combined arm's multiplier rather than
# covariance matching. A crossing measured with a different algorithm on each
# simulator would not be evidence that the crossing transfers.


def health_arm():
    """The learned model on its own, without the additive residual.

    load_health_model returns the bare measurement function; the filter also
    needs a constraint keeping health non-negative. Without it the region below
    zero costs nothing -- the model clips its input, so it returns the same
    prediction there -- and an update will happily move into it while still
    absorbing correction that belonged elsewhere.
    """
    measure = quad.load_health_model()

    def constrain(mean):
        mean = np.array(mean, dtype=float)
        mean[quad.HEALTH_STATES] = np.maximum(mean[quad.HEALTH_STATES], 0.0)
        return mean

    measure.constrain = constrain
    return measure


def one_run(arm, seed, mode, severity, wide):
    run = random_run(seed)
    meas = read_sensors(run, seed=seed)
    if mode is not None and severity > 0:
        meas = apply_fault(meas, DEVICE, mode, severity, seed=seed, dt=DT)

    readings = stack(meas)
    truth = truth_matrix(run)

    if wide:
        Q_use, P_use = quad.filter_settings()
        start = np.zeros(quad.N_STATES)
    else:
        Q_use, P_use = quad.Q, quad.P0
        start = np.zeros(quad.N_VEHICLE)
    start[:quad.N_VEHICLE] = truth[0]

    if hasattr(arm, "reset"):
        arm.reset()
    means, _, innov, S = quad.ukf.UKF(Q_use, quad.best_constant_R(),
                                      move=quad.move_state,
                                      measure=arm).run(readings, start,
                                                       P_use, DT)

    error = np.degrees(means[:, :2] - truth[:, :2])
    return (float(np.sqrt(np.mean(error ** 2))),
            float(quad.ukf.nis(innov, S).mean()))


def score(arm, mode, severity, wide):
    rows = [one_run(arm, s, mode, severity, wide) for s in SEEDS]
    return (float(np.mean([r[0] for r in rows])),
            float(np.mean([r[1] for r in rows])))


def main():
    if not (HERE / "quad_health.pt").exists():
        print("No quad_health.pt -- run train.py first.")
        return 0

    R_const = quad.best_constant_R()
    arms = [
        ("analytic + best const R", quad.analytic_readings, False),
        ("adaptive R (Mehra)", quad.adaptive_arm(R_const), False),
        ("health-conditioned", health_arm(), True),
        ("layered", quad.load_measurement_model(), True),
    ]

    conditions = [("healthy", None, 0.0)]
    for mode in MODES:
        for severity in SEVERITIES:
            conditions.append(("%s %.1f" % (mode.split("_")[0], severity),
                               mode, severity))

    print("DOES THE CROSSING SURVIVE A NONLINEAR MAP?\n")
    print("%d flights, seeds %d-%d, %s degraded. Attitude error over roll"
          % (len(list(SEEDS)), min(SEEDS), max(SEEDS), DEVICE))
    print("and pitch, in degrees.\n")

    records, table = [], {}
    for label, arm, wide in arms:
        print("  running %-26s" % label, end="", flush=True)
        row = []
        for name, mode, severity in conditions:
            e, n = score(arm, mode, severity, wide)
            row.append((e, n))
            records.append({"arm": label, "condition": name,
                            "mode": mode or "none", "severity": severity,
                            "moment": MOMENT.get(mode, "none"),
                            "trained": mode in TRAINED_ON if mode else True,
                            "attitude_rmse_deg": e, "nis": n})
        table[label] = row
        print("done")

    heads = [c[0] for c in conditions]
    for index, title, note in [
            (0, "ATTITUDE ERROR, DEGREES",
             "health should win the bias columns and lose the noise ones."),
            (1, "NIS -- want 9",
             "whether each arm knows how wrong it is.")]:
        print("\n\n%s\n" % title)
        print("  %-26s" % "" + "".join("%12s" % h for h in heads))
        print("  " + "-" * (26 + 12 * len(heads)))
        for label, row in table.items():
            print("  %-26s" % label
                  + "".join("%12.4f" % v[index] for v in row))
        print("  " + "-" * (26 + 12 * len(heads)))
        print("\n  %s" % note)

    print("\n\nTHE CROSSING\n")
    worst = SEVERITIES[-1]
    for mode in MODES:
        name = "%s %.1f" % (mode.split("_")[0], worst)
        column = heads.index(name)
        pair = {label: table[label][column][0]
                for label in table if label in ("health-conditioned",
                                                "adaptive R (Mehra)")}
        winner = min(pair, key=pair.get)
        expected = ("health-conditioned" if MOMENT[mode] == "first"
                    else "adaptive R (Mehra)")
        print("  %-18s %8s moment   predicted %-22s got %-22s %s"
              % (mode, MOMENT[mode], expected.split(" (")[0],
                 winner.split(" (")[0],
                 "yes" if winner == expected else "NO"))

    print("\n  If both rows say yes, the taxonomy is about moment order and")
    print("  not about whether the measurement map happens to be a matrix --")
    print("  which is what the derivation from dh/dm claims, and which the")
    print("  ground robot could not test because its map is linear.")

    RESULTS.mkdir(exist_ok=True)
    stem = "quad_bakeoff" if DEVICE == "accel" else "quad_bakeoff_%s" % DEVICE
    path = RESULTS / ("%s.csv" % stem)
    pd.DataFrame(records).to_csv(path, index=False, float_format="%.6g")
    print("\nWrote %d rows to %s" % (len(records), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

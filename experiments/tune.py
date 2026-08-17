"""
Finds Q and R that make the filter honest, instead of tuning them by eye.

THE PROBLEM THIS SOLVES

Every arm currently fails NEES. The best is 2.666 against a target of 2, and
the hand-written analytic model fails it too, at 3.646. When the control fails
a test, the test is not about the measurement models -- it is about the
process model. Q is wrong, and no amount of work on the measurement side
fixes it.

WHY BOTH AT ONCE

Raising Q makes the filter less sure of itself. That pulls NEES down, which
is what is wanted. But the same change raises the predicted innovation spread

    S = spread of h(x) + R

so NIS falls too, and NIS is already slightly under target at 2.876. Fixing
one breaks the other, so they have to be searched together: raise Q to fix
NEES, lower R to put NIS back.

THE SCORE

Chen et al. give the standard: a correctly tuned filter matches BOTH moments
of a chi-square, not just the average.

    NIS  mean 3, spread 6      (three sensors)
    NEES mean 2, spread 4      (two states being checked)

Each of the four is scored by how far off it is relative to its own target,
so that a large number and a small one count equally, and the four are added.
Zero is perfect.

Searched on the analytic model, because Q is a property of the motion model
and should not be fitted through a learned component.

WHAT THIS SCORE CANNOT TELL YOU

The same authors are careful about the limits of these statistics, and this
file uses them anyway, so the limits belong here rather than buried in a
citation.

The targets are only correct at the answer. NIS and NEES follow a chi-square
distribution when the filter is already tuned properly. When it is not --
which is every candidate this search tries except the last one -- they follow
a generalised chi-square instead, so 3 / 6 / 2 / 4 is the right yardstick
only at the point the search is trying to reach. Scores far from zero should
be read as "worse" and not as any particular quantity.

Two moments are necessary and still not sufficient. Checking the spread as
well as the average removes one way a filter can look correct while being
wrong, but Chen et al. show in the 2024 paper that it does not remove all of
them, and they propose a different measure for that reason. This file uses
the older criterion because it is simple enough to read; that is a choice
made for clarity and not because the criterion is adequate.

Winning on one moment while losing on another is expected. Optimising the
mean NIS pulls the variance off, and optimising the variance pulls the mean
off -- the 2024 paper reports exactly that. So a search that fixes NEES while
breaking NIS has not necessarily discovered anything about the model; that
pattern is a property of these objectives. Deciding the process model is
wrong needs separate evidence, of the kind that does not depend on any
consistency metric.

A grid is the wrong search. These scores come from a stochastic simulation
with no derivatives, and both papers exist largely to say that such problems
want Bayesian optimisation over a surrogate rather than a fixed grid. Ten
runs per candidate is also thin for statistics this noisy. The grid here is
honest about being a starting point.

And NEES needs the true state, so it exists only in simulation. On the actual
robot only NIS can be computed, which means every NEES number in this project
is a claim about the simulator rather than about the hardware.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import (NIS_DOF, NEES_DOF, P0, Q, R_TUNED, filter_once,
                    load_arm, make_run)

SEARCH_RUNS = 10        # runs per candidate during the search
CHECK_RUNS = 20         # runs used to confirm the winner

# Multipliers tried on the speed and turn-rate entries of Q, and on R as a
# whole. Log-spaced, because these are scale parameters -- doubling matters
# the same amount whether the starting point is small or large.
Q_SCALES = [1.0, 3.0, 10.0, 30.0, 100.0]
R_SCALES = [0.6, 0.8, 1.0, 1.25]


def score(measure, q_speed, q_turn, r_scale, n_runs):
    """How far this setting is from a consistent filter. Lower is better."""
    Q_try = Q.copy()
    Q_try[3, 3] = q_speed
    Q_try[4, 4] = q_turn
    R_try = R_TUNED * r_scale

    nis_all, nees_all = [], []
    for seed in range(n_runs):
        run, meas, truth = make_run(seed)
        result = filter_once_with(measure, meas, truth, Q_try, R_try)
        nis_all.append(result["nis"])
        nees_all.append(result["nees"])

    nis = np.concatenate(nis_all)
    nees = np.concatenate(nees_all)

    parts = [
        (nis.mean() - NIS_DOF) / NIS_DOF,
        (nis.var() - 2 * NIS_DOF) / (2 * NIS_DOF),
        (nees.mean() - NEES_DOF) / NEES_DOF,
        (nees.var() - 2 * NEES_DOF) / (2 * NEES_DOF),
    ]
    return float(sum(p ** 2 for p in parts)), {
        "nis_mean": nis.mean(), "nis_var": nis.var(),
        "nees_mean": nees.mean(), "nees_var": nees.var(),
    }


def filter_once_with(measure, meas, truth, Q_try, R_try):
    """filter_once, but with Q and R supplied rather than taken from common.

    common.filter_once deliberately hard-codes the shared settings so no
    experiment can drift. Tuning is the one job that has to vary them, so it
    reaches past that on purpose.
    """
    from ukf import UKF, nis as nis_of, nees as nees_of
    from common import NEES_STATES, stack
    from trajectories import DT

    readings = stack(meas)
    if hasattr(measure, "reset"):
        measure.reset()
    means, covs, innov, S = UKF(Q_try, R_try, measure=measure).run(
        readings, truth[0].copy(), P0, DT)
    return {"nis": nis_of(innov, S),
            "nees": nees_of(means, covs, truth, states=NEES_STATES)}


def main():
    measure = load_arm("fixed")

    print("Starting point")
    base, stats = score(measure, Q[3, 3], Q[4, 4], 1.0, SEARCH_RUNS)
    print("  q_speed %.1e  q_turn %.1e  r x1.00" % (Q[3, 3], Q[4, 4]))
    print("  NIS %.3f / %.3f   NEES %.3f / %.3f   score %.3f"
          % (stats["nis_mean"], stats["nis_var"],
             stats["nees_mean"], stats["nees_var"], base))

    print("\nSearching %d combinations"
          % (len(Q_SCALES) ** 2 * len(R_SCALES)))

    best = (base, Q[3, 3], Q[4, 4], 1.0, stats)
    for qs in Q_SCALES:
        for qt in Q_SCALES:
            for rs in R_SCALES:
                q_speed = Q[3, 3] * qs
                q_turn = Q[4, 4] * qt
                value, stats = score(measure, q_speed, q_turn, rs, SEARCH_RUNS)
                if value < best[0]:
                    best = (value, q_speed, q_turn, rs, stats)
                    print("  better: q_speed %.1e  q_turn %.1e  r x%.2f"
                          "   NIS %.3f/%.3f  NEES %.3f/%.3f   score %.3f"
                          % (q_speed, q_turn, rs, stats["nis_mean"],
                             stats["nis_var"], stats["nees_mean"],
                             stats["nees_var"], value))

    value, q_speed, q_turn, r_scale, _ = best
    print("\nConfirming on %d runs" % CHECK_RUNS)
    final, stats = score(measure, q_speed, q_turn, r_scale, CHECK_RUNS)

    print("\n%-14s %12s %12s" % ("", "before", "after"))
    print("%-14s %12.3f %12.3f" % ("NIS mean", 2.876, stats["nis_mean"]))
    print("%-14s %12.3f %12.3f" % ("NIS spread", 5.545, stats["nis_var"]))
    print("%-14s %12.3f %12.3f" % ("NEES mean", 3.646, stats["nees_mean"]))
    print("%-14s %12.3f %12.3f" % ("NEES spread", 13.442, stats["nees_var"]))
    print("%-14s %12s %12s" % ("target", "", "3 / 6 / 2 / 4"))

    print("\nSettings to use:")
    print("  Q = np.diag([1e-9, 1e-9, 1e-9, %.3e, %.3e])" % (q_speed, q_turn))
    print("  R_TUNED = np.diag([%.6f ** 2, %.6f ** 2, %.6f ** 2])"
          % (np.sqrt(R_TUNED[0, 0] * r_scale),
             np.sqrt(R_TUNED[1, 1] * r_scale),
             np.sqrt(R_TUNED[2, 2] * r_scale)))

    edge = (q_speed / Q[3, 3] == Q_SCALES[-1] or q_turn / Q[4, 4] == Q_SCALES[-1]
            or r_scale in (R_SCALES[0], R_SCALES[-1]))
    if edge:
        print("\n  WARNING: a chosen value sits at the edge of its grid, so a")
        print("  better setting probably lies outside. Widen and re-run.")


if __name__ == "__main__":
    main()

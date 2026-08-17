"""
Searches for process and measurement noise covariances that leave the filter
statistically consistent, rather than setting them by hand.

Every arm currently fails the NEES test, the hand-written analytic model
included, which fails it at 3.646 against a target of 2. A test that the
control fails is not reporting on the measurement models; it is reporting on
the process model, and no refinement of the measurement side will correct it.

The two covariances have to be searched jointly rather than in sequence.
Increasing Q makes the filter less certain of its own prediction, which lowers
NEES as intended, but the same change raises the predicted innovation
covariance S = P_hh + R and so lowers NIS, which already sits slightly below
target at 2.876. Adjusting either statistic alone displaces the other.

The objective follows Chen, Heckman, Julier and Ahmed [1] and its journal
successor [2]: a correctly tuned filter matches both moments of the relevant
chi-square distribution, not its mean alone. With three sensors, NIS should
have mean 3 and variance 6; over the two states checked here, NEES should have
mean 2 and variance 4. Each of the four quantities is scored by its relative
departure from its own target and the four are summed, so that a large
statistic and a small one contribute comparably. Tuning is carried out against
the analytic measurement model, since Q describes the motion and should not be
fitted through a learned component that could absorb error properly belonging
to the dynamics.

Four limitations of this procedure deserve statement, since [1] and [2] are
candid about them and this file proceeds anyway.

NIS and NEES are chi-square distributed only when the filter is already
correctly tuned. For any other parameter setting -- which is to say every
candidate the search evaluates but the last -- [2] shows they follow a
generalised chi-square distribution instead. The targets above are therefore
valid only at the solution being sought, and intermediate scores are ordinal:
a score of 4 is worse than a score of 1, but neither is a calibrated quantity.

Matching both moments removes one failure mode and not all of them. [2]
demonstrates that the two-moment criterion still admits distinct filters with
indistinguishable statistics, and introduces a further measure for that
reason. The simpler criterion is retained here for legibility, which is a
deliberate simplification rather than a claim of sufficiency.

Improvement in one moment at the expense of another is characteristic of these
objectives rather than diagnostic of the model. [2] reports that minimising
the mean NIS displaces its variance, and conversely. A search that repairs
NEES while degrading NIS therefore establishes nothing on its own about model
structure; such a conclusion requires evidence independent of any consistency
statistic.

A fixed grid is a poor search strategy for this problem. The objective is a
stochastic, non-differentiable function evaluated by simulation, which is
precisely the setting [1] and [2] address with Bayesian optimisation over a
surrogate model, and ten Monte Carlo runs per candidate is thin for statistics
of this variance. The grid below is a coarse starting point.

NEES additionally requires the true state and is available only in simulation.
Only NIS can be computed aboard the vehicle, so every NEES result in this
project is a statement about the simulator rather than about hardware.

[1] Z. Chen, C. Heckman, S. Julier and N. Ahmed, "Weak in the NEES?:
    Auto-tuning Kalman Filters with Bayesian Optimization," 2018.
    arXiv:1807.08855
[2] Z. Chen, H. Biggie, N. Ahmed, S. Julier and C. Heckman, "Kalman Filter
    Auto-Tuning With Consistent and Robust Bayesian Optimization," IEEE
    Transactions on Aerospace and Electronic Systems, vol. 60, no. 2,
    pp. 2236-2250, 2024. doi:10.1109/TAES.2024.3350587
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

    edge = (q_speed / Q[3, 3] == Q_SCALES[-1]
            or q_turn / Q[4, 4] == Q_SCALES[-1]
            or r_scale in (R_SCALES[0], R_SCALES[-1]))
    if edge:
        print("\n  WARNING: a chosen value sits at the edge of its grid, so a")
        print("  better setting probably lies outside. Widen and re-run.")


if __name__ == "__main__":
    main()

"""
Searches for process and measurement noise covariances that leave the filter
statistically consistent, rather than setting them by hand.

The two covariances have to be searched jointly rather than in sequence.
Increasing Q makes the filter less certain of its own prediction, which lowers
NEES as intended, but the same change raises the predicted innovation
covariance S = P_hh + R and so lowers NIS. Adjusting either statistic alone
displaces the other.

The objective is C_NEES and C_NIS, equation (26) of Chen, Biggie, Ahmed,
Julier and Heckman [2]:

    C = |log(mean / dof)| + |log(variance / 2 dof)|

A correctly tuned filter matches both moments of the relevant chi-square
distribution, not its mean alone: with three sensors NIS should have mean 3
and variance 6, and over the two states checked here NEES should have mean 2
and variance 4.

An earlier version of this file scored each quantity by its squared relative
departure instead, which is not the same thing and is worse. See
consistency_cost below for what that cost us. Using a measure of one's own
devising, when the paper already cited in the repository supplies one, is a
mistake worth leaving recorded.

Tuning is carried out against the analytic measurement model, since Q
describes the motion and should not be fitted through a learned component that
could absorb error properly belonging to the dynamics.

Four limitations remain, since [1] and [2] are candid about them and this file
proceeds anyway.

NIS and NEES are chi-square distributed only when the filter is already
correctly tuned. For any other parameter setting -- which is to say every
candidate the search evaluates but the last -- [2] shows they follow a
generalised chi-square distribution instead. The targets above are therefore
valid only at the solution being sought, and intermediate scores are ordinal:
a score of 4 is worse than a score of 1, but neither is a calibrated quantity.

Equation (27) of [2] sums C over several time-discretisation intervals,
because Section III-C of that paper shows these statistics carry an implicit
dependence on the step length. This file evaluates at one dt only, so it
implements the measure but not the full metric.

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

from common import (NIS_DOF, NEES_DOF, P0, Q, best_constant_R,
                    filter_once, load_arm, make_run)

# Scaled from the best available constant, not from the hand-tuned R.
# The hand-tuned value absorbed state error as well as sensor noise and
# is about 20 per cent larger than the sensors are, so searching around
# it means searching around the wrong point -- which is how this file
# came to report a NEES of 2.53 where every experiment reported 1.81.
R_BASE = best_constant_R()

SEARCH_RUNS = 10        # runs per candidate during the search
CHECK_RUNS = 20         # runs used to confirm the winner

# Multipliers tried on the two acceleration entries of Q, and on R as a
# whole. Log-spaced, because these are scale parameters -- doubling matters
# the same amount whether the starting point is small or large.
#
# The accelerations, not speed and turn rate. Those are moved by the
# accelerations now rather than by noise, so their own entries are held at
# 1e-9 and there is nothing there to tune. This file searched them anyway
# until the state gained acceleration and nobody updated the indices.
Q_SCALES = [0.1, 0.3, 1.0, 3.0, 10.0]
R_SCALES = [0.7, 0.85, 1.0, 1.2]


def consistency_cost(mean, variance, dof):
    """C_NIS / C_NEES, equation (26) of [2]. Lower is better, zero is exact.

        C = |log(mean / dof)| + |log(variance / 2 dof)|

    The log matters, and this file used a squared relative error until it
    turned out to matter. Writing v for actual-over-target, a squared error
    charges (v-1)^2, which is 1.00 for v = 2 and 0.25 for v = 0.5: being
    double the target costs four times as much as being half of it. For a
    scale parameter that is simply wrong, and it quietly biases a search
    toward settings whose statistics come out under target.

    The filter here had been sitting at NIS 2.5 against a target of 3 --
    0.83 of target, exactly the direction that asymmetry rewards -- and that
    was being explained away as R needing a small correction.

    |log v| is symmetric: half and double both cost 0.69.
    """
    return abs(np.log(mean / dof)) + abs(np.log(variance / (2.0 * dof)))


def score(measure, q_accel, q_turn_accel, r_scale, n_runs):
    """How far this setting is from a consistent filter. Lower is better."""
    Q_try = Q.copy()
    Q_try[5, 5] = q_accel
    Q_try[6, 6] = q_turn_accel
    R_try = R_BASE * r_scale

    nis_all, nees_all = [], []
    for seed in range(n_runs):
        run, meas, truth = make_run(seed)
        result = filter_once_with(measure, meas, truth, Q_try, R_try)
        nis_all.append(result["nis"])
        nees_all.append(result["nees"])

    nis = np.concatenate(nis_all)
    nees = np.concatenate(nees_all)

    total = (consistency_cost(nis.mean(), nis.var(), NIS_DOF)
             + consistency_cost(nees.mean(), nees.var(), NEES_DOF))
    return float(total), {
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
    base, base_stats = score(measure, Q[5, 5], Q[6, 6], 1.0, SEARCH_RUNS)
    print("  q_accel %.1e  q_turn_accel %.1e  r x1.00"
          % (Q[5, 5], Q[6, 6]))
    print("  NIS %.3f / %.3f   NEES %.3f / %.3f   score %.3f"
          % (base_stats["nis_mean"], base_stats["nis_var"],
             base_stats["nees_mean"], base_stats["nees_var"], base))

    print("\nSearching %d combinations"
          % (len(Q_SCALES) ** 2 * len(R_SCALES)))

    best = (base, Q[5, 5], Q[6, 6], 1.0, base_stats)
    for qs in Q_SCALES:
        for qt in Q_SCALES:
            for rs in R_SCALES:
                q_accel = Q[5, 5] * qs
                q_turn_accel = Q[6, 6] * qt
                value, stats = score(measure, q_accel, q_turn_accel, rs,
                                     SEARCH_RUNS)
                if value < best[0]:
                    best = (value, q_accel, q_turn_accel, rs, stats)
                    print("  better: q_accel %.1e  q_turn_accel %.1e  r x%.2f"
                          "   NIS %.3f/%.3f  NEES %.3f/%.3f   score %.3f"
                          % (q_accel, q_turn_accel, rs, stats["nis_mean"],
                             stats["nis_var"], stats["nees_mean"],
                             stats["nees_var"], value))

    value, q_accel, q_turn_accel, r_scale, _ = best
    print("\nConfirming on %d runs" % CHECK_RUNS)
    final, stats = score(measure, q_accel, q_turn_accel, r_scale,
                         CHECK_RUNS)

    print("\n%-14s %12s %12s" % ("", "before", "after"))
    print("%-14s %12.3f %12.3f" % ("NIS mean", base_stats["nis_mean"],
                                   stats["nis_mean"]))
    print("%-14s %12.3f %12.3f" % ("NIS spread", base_stats["nis_var"],
                                   stats["nis_var"]))
    print("%-14s %12.3f %12.3f" % ("NEES mean", base_stats["nees_mean"],
                                   stats["nees_mean"]))
    print("%-14s %12.3f %12.3f" % ("NEES spread", base_stats["nees_var"],
                                   stats["nees_var"]))
    print("%-14s %12s %12s" % ("target", "", "3 / 6 / 2 / 4"))

    print("\nSettings to use:")
    print("  Q = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, %.3e, %.3e])"
          % (q_accel, q_turn_accel))
    print("  R = np.diag([%.6f ** 2, %.6f ** 2, %.6f ** 2])"
          % tuple(np.sqrt(np.diag(R_BASE * r_scale))))

    edge = (q_accel / Q[5, 5] in (Q_SCALES[0], Q_SCALES[-1])
            or q_turn_accel / Q[6, 6] in (Q_SCALES[0], Q_SCALES[-1])
            or r_scale in (R_SCALES[0], R_SCALES[-1]))
    if edge:
        print("\n  WARNING: a chosen value sits at the edge of its grid, so a")
        print("  better setting probably lies outside. Widen and re-run.")


if __name__ == "__main__":
    main()

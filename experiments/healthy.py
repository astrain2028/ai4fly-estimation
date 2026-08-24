"""
Every arm on healthy data, same runs, same settings, one table.

This is the control condition. Nothing is broken, no sensor is degraded, and
the only question is how well each measurement model does the ordinary job.

WHAT TO LOOK AT

Accuracy is the obvious column and the least interesting one. Every arm is
approximating a relationship that is known exactly, so they should all be
close, and a model that is much worse has a bug rather than a finding.

The consistency columns are the point. A filter is consistent when it is as
surprised as it predicted it would be -- NIS should average 3, one per
sensor, and NEES should average 2, one per state being checked. Both need
their spread checked too, not just their average, because a filter can get
the average exactly right while over-trusting one sensor and under-trusting
another by amounts that cancel.

The last column is the one that decides what can fly. A step has 20 ms at
50 Hz, and the flight stack needs most of that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import (ARMS, LABELS, SLOW, N_RUNS, NIS_DOF, NEES_DOF,
                    available, best_constant_R, gather, load_arm, two_moment)

# The GP compares every query against 1500 stored points, so a full sweep
# takes long enough to be unpleasant. It gets fewer runs, and the table says
# so rather than quietly dropping it.
SLOW_RUNS = 5


def main():
    ready, missing = available()

    print("HEALTHY CONDITION -- %d runs of 20 s at 50 Hz\n" % N_RUNS)
    print("%-20s %9s %9s   %14s   %14s %9s"
          % ("", "speed", "turn", "NIS (want 3)", "NEES (want 2)", "ms/step"))
    print("%-20s %9s %9s   %6s %7s   %6s %7s %9s"
          % ("", "m/s", "rad/s", "mean", "spread", "mean", "spread", ""))
    print("-" * 88)

    # Arms without their own covariance get the best constant available, not
    # the hand-tuned one. The hand-tuned value absorbs state uncertainty as
    # well as sensor noise and is inflated as a result, and measuring a
    # learned covariance against it flattered the learned covariance.
    R_const = best_constant_R()

    results = {}
    for name in ARMS:
        if name not in ready:
            continue
        n = SLOW_RUNS if name in SLOW else N_RUNS
        measure = load_arm(name)
        r = gather(measure, range(n), R=R_const)
        results[name] = r

        nis_mean, nis_var, _, _ = two_moment(r["nis"], NIS_DOF)
        nees_mean, nees_var, _, _ = two_moment(r["nees"], NEES_DOF)

        note = LABELS[name] + (" *" if name in SLOW else "")
        print("%-20s %9.4f %9.4f   %6.3f %7.3f   %6.3f %7.3f %9.3f"
              % (note, r["speed_rmse"].mean(), r["turn_rmse"].mean(),
                 nis_mean, nis_var, nees_mean, nees_var, r["ms_per_step"]))

    print("-" * 88)
    print("%-20s %9s %9s   %6.1f %7.1f   %6.1f %7.1f"
          % ("target", "", "", float(NIS_DOF), 2.0 * NIS_DOF,
             float(NEES_DOF), 2.0 * NEES_DOF))

    if any(name in SLOW for name in results):
        print("\n* %d runs instead of %d -- too slow for the full sweep, which"
              % (SLOW_RUNS, N_RUNS))
        print("  is itself the finding for that arm.")

    if missing:
        print("\nNot run (not trained yet):")
        for name, kind, _ in missing:
            print("  %-12s %s" % (name, LABELS.get(name, "")))

    # ---- paired comparison against the control ----
    # The same seeds went through every arm, so differences can be compared
    # run by run. A mean difference smaller than its own spread is noise, no
    # matter how good it looks in the table above.
    if "fixed" in results:
        print("\n\nAGAINST THE HAND-WRITTEN CONTROL, RUN BY RUN")
        print("(negative means the learned arm did better)\n")
        print("%-20s %24s   %24s"
              % ("", "speed", "turn"))
        print("%-20s %7s %8s %7s   %7s %8s %7s"
              % ("", "wins", "mean", "spread", "wins", "mean", "spread"))
        print("-" * 78)

        for name, r in results.items():
            if name == "fixed" or name in SLOW:
                continue
            row = "%-20s" % LABELS[name]
            for key in ["speed_rmse", "turn_rmse"]:
                diff = r[key] - results["fixed"][key][:len(r[key])]
                row += ("   %5d/%-2d %+8.5f %7.5f"
                        % (np.sum(diff < 0), len(diff), diff.mean(), diff.std()))
            print(row)

        print("\nA mean smaller than its spread is not a result. Read those")
        print("columns together or not at all.")


if __name__ == "__main__":
    main()

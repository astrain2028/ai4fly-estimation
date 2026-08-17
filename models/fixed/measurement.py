"""
The control arm: the measurement model written out by hand.

Nothing is learned here. The wheel speeds follow from the kinematics, the
gyro reads the turn rate, and the noise is a constant matrix tuned once by
covariance matching -- filter, see how big the innovations actually were,
adjust R, repeat.

This arm exists because every claim in the project is a claim about beating
it. It already lives inside ukf.py as the filter's default, but a default is
awkward to compare against: it has no folder, no name, and no way to be
loaded by the same call as everything else. This file is a thin wrapper that
makes it addressable like any other arm.

Worth remembering what it costs. It runs in microseconds, needs no training
data, no GPU, and no Python at all -- it is a dozen lines of arithmetic that
would port to C in an afternoon. Anything learned has to be better by enough
to justify replacing that.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

from ukf import expected_readings


def load_measurement_model(path=None):
    """Return the hand-written model, for symmetry with the learned arms.

    `path` is accepted and ignored. There are no weights to load, which is
    the entire point of this arm.
    """
    return expected_readings


if __name__ == "__main__":
    import numpy as np
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (R_TUNED, N_RUNS, NIS_DOF, NEES_DOF, gather,
                        two_moment, load_arm)

    measure = load_arm("fixed")
    results = gather(measure, range(N_RUNS))

    print("The hand-written model over %d runs\n" % N_RUNS)
    print("  speed error  %.4f m/s" % results["speed_rmse"].mean())
    print("  turn error   %.4f rad/s" % results["turn_rmse"].mean())

    for name, dof in [("NIS", NIS_DOF), ("NEES", NEES_DOF)]:
        mean, var, t_mean, t_var = two_moment(results[name.lower()], dof)
        print("  %-5s mean %6.3f (want %.1f)   spread %6.3f (want %.1f)"
              % (name, mean, t_mean, var, t_var))

    print("\n  %.3f ms per filter step" % results["ms_per_step"])
    print("\n  R was tuned by hand to make those two moments come out right.")
    print("  A learned covariance has to earn its keep against this.")

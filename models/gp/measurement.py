"""
Uses the Gaussian process as the filter's measurement model.

Like the heteroscedastic model, this one supplies its own noise. Unlike it,
the noise is the same everywhere: the varying part is the GP's doubt about
its own answer, which grows away from the stored points.

A warning about cost. The GP answers a question by comparing it against
every stored point, so the work per call grows with how many points were
kept -- and it grows as the square, because the comparison feeds a
triangular solve.

With 1500 stored points this arm has been timed at 23, 29, 96 and 567 ms per
filter step on the same machine, against the 20 ms available at 50 Hz. That
spread is not a mistake in the timing -- the reported figure tracks wall clock
to within one per cent -- it is the arm itself, whose large matrix work
competes for memory bandwidth in a way the small networks do not. Those run at
about 1 ms and vary by a few per cent.

So the honest statement is that the GP misses a 50 Hz budget by somewhere
between a little and a lot, and that no single number should be quoted for it
without a benchmark that controls for machine load. Keeping fewer points would
fit, at the cost of accuracy -- a trade the network models never face, since
their cost is set by their shape rather than by how much data they saw.

Even the worst of those figures is far better than before this file used a
triangular solve, and that story is worth keeping. The predictive variance was
being solved against a Cholesky factor with a general solver, refactorising a
1500x1500 matrix on every sigma point of every step, for a measured 243-fold
penalty. Cost is a property of the implementation until proven otherwise, and
"this method is too slow" is a claim that needs a profiler behind it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
from scipy.linalg import solve_triangular



def _sibling_train():
    """Load train.py from THIS folder.

    Every arm has a file called train.py, so a plain `import train` picks
    whichever one Python happened to load first -- the plain arm's module
    would satisfy the bhr arm's import, silently. Loading by full path under
    a unique name avoids that.
    """
    import importlib.util
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        here.name + "_train", here / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _sibling_train()
rbf_kernel = _train.rbf_kernel
OUTPUTS = _train.OUTPUTS
INPUTS = _train.INPUTS



def load_measurement_model(path=None):
    """Load the fitted GPs and return a function the filter can call.

    Returns readings, shape (n, 3), and a covariance per state,
    shape (n, 3, 3) -- the same shape the heteroscedastic model returns, so
    the two are interchangeable in the filter.
    """
    if path is None:
        path = Path(__file__).parent / "gp_model.npz"
    saved = np.load(path)

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    n_out = int(saved["n_outputs"])
    gps = [{k: saved["%s_%d" % (k, i)] for k in
            ("X", "alpha", "L", "lengthscale", "noise", "y_mean", "y_std")}
           for i in range(n_out)]

    def measure(states):
        # The filter carries more states than this model was trained
        # on -- the accelerations are appended after speed and turn
        # rate, which the sensors do not see. Take the five it knows.
        states = np.atleast_2d(states)[:, :len(INPUTS)]
        xs = (states - x_mean) / x_std

        readings = np.zeros((len(states), n_out))
        variances = np.zeros((len(states), n_out))

        for i, gp in enumerate(gps):
            k = rbf_kernel(xs, gp["X"], float(gp["lengthscale"]))
            mean_s = k @ gp["alpha"]

            # doubt left over after the stored points have had their say.
            #
            # L is a Cholesky factor, so it is lower triangular and this
            # solve is a cheap back-substitution. np.linalg.solve does not
            # know that and factorises the whole 1500x1500 matrix again on
            # every call -- which it was doing here, once per sigma point per
            # filter step, and it made the GP look hundreds of times more
            # expensive than it is.
            v = solve_triangular(gp["L"], k.T, lower=True)
            epistemic = np.maximum(1.0 - np.sum(v ** 2, axis=0), 0.0)
            total_s = epistemic + float(gp["noise"]) ** 2

            readings[:, i] = mean_s * gp["y_std"] + gp["y_mean"]
            variances[:, i] = total_s * gp["y_std"] ** 2

        R = np.zeros((len(states), n_out, n_out))
        for j in range(len(states)):
            R[j] = np.diag(variances[j])
        return readings, R

    return measure


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (N_RUNS, NIS_DOF, NEES_DOF, best_constant_R, gather,
                        load_arm)

    # The best single covariance available, which is the average variance.
    # Every arm without its own covariance gets this one, so the numbers in
    # these blocks can honestly be compared with each other.
    R = best_constant_R()

    print("Gaussian process against the hand-written model, %d runs."
          % N_RUNS)
    print()
    print("%-24s %9s %9s %9s %9s %10s"
          % ("", "speed", "turn", "NIS", "NEES", "ms/step"))
    for label, name in [("written by hand", "fixed"),
                        ("Gaussian process", "gp")]:
        r = gather(load_arm(name), range(N_RUNS), R=R)
        print("%-24s %9.4f %9.4f %9.3f %9.3f %10.2f"
              % (label, r["speed_rmse"].mean(),
                 r["turn_rmse"].mean(), r["nis"].mean(),
                 r["nees"].mean(), r["ms_per_step"]))
    print("%-24s %9s %9s %9.1f %9.1f"
          % ("should be", "", "", float(NIS_DOF), float(NEES_DOF)))

    print()
    print("Timing is wall clock on a shared machine and moves around a")
    print("great deal between runs. Treat it as an order of magnitude:")
    print("a real benchmark needs warmup and repeats.")
    print()
    print("The GP gets epistemic uncertainty free from its kernel, and")
    print("pays by comparing every query against every stored point. Its")
    print("noise term is one constant, so it does half of what is needed.")

"""
Uses the Gaussian process as the filter's measurement model.

Like the heteroscedastic model, this one supplies its own noise. Unlike it,
the noise is the same everywhere: the varying part is the GP's doubt about
its own answer, which grows away from the stored points.

A warning about cost. The GP answers a question by comparing it against
every stored point, so the work per call grows with how many points were
kept -- and it grows as the square, because the comparison feeds a
triangular solve. Keeping enough points to be accurate makes it far too slow
to run in a real-time loop. That tension has no equivalent in the network
models, whose cost is fixed by their shape rather than by how much data they
were trained on.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np

from train import rbf_kernel, OUTPUTS


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
        states = np.atleast_2d(states)
        xs = (states - x_mean) / x_std

        readings = np.zeros((len(states), n_out))
        variances = np.zeros((len(states), n_out))

        for i, gp in enumerate(gps):
            k = rbf_kernel(xs, gp["X"], float(gp["lengthscale"]))
            mean_s = k @ gp["alpha"]

            # doubt left over after the stored points have had their say
            v = np.linalg.solve(gp["L"], k.T)
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
    import time
    from trajectories import DT, random_run
    from sensors import read_sensors
    from ukf import UKF, expected_readings, nis

    gp_measure = load_measurement_model()
    Q = np.diag([1e-9, 1e-9, 1e-9, 2e-5, 1e-4])
    fixed_R = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2])

    # A short run: the GP is slow enough that a full-length one is painful.
    run = random_run(0, duration=2.0)
    meas = read_sensors(run, seed=0, dt=DT)
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    truth = np.column_stack([run["x"], run["y"], run["heading"],
                             run["speed"], run["turn_rate"]])
    start_cov = np.diag([0.01, 0.01, 0.01, 0.10, 0.10])

    print("Two seconds of filtering, %d steps.\n" % len(readings))
    print("%-26s %10s %10s %9s %10s"
          % ("", "speed err", "turn err", "NIS", "seconds"))

    for label, fn, R in [("written by hand", expected_readings, fixed_R),
                         ("Gaussian process", gp_measure, fixed_R)]:
        t0 = time.time()
        means, covs, innov, S = UKF(Q, R, measure=fn).run(
            readings, truth[0].copy(), start_cov, DT)
        elapsed = time.time() - t0
        print("%-26s %10.4f %10.4f %9.3f %10.1f"
              % (label,
                 np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)),
                 np.sqrt(np.mean((means[:, 4] - run["turn_rate"]) ** 2)),
                 nis(innov, S).mean(), elapsed))

    per_step = elapsed / len(readings) * 1000
    print("\n   %.0f ms per filter step, against %.0f ms available at 50 Hz."
          % (per_step, 1000 * DT))
    print("   That is %.0f times over budget. Keeping fewer points would fit,"
          % (per_step / (1000 * DT)))
    print("   at the cost of accuracy -- a trade the network models never face.")

"""
Uses the trained network as the filter's measurement model.

Normally a filter needs someone to write down what each sensor should read
given the state. Here that function is learned from data instead, and this
file is the piece that lets the UKF call it.

The network was trained on scaled inputs and outputs, so the wrapper has to
scale the state going in and unscale the readings coming out. Those scaling
numbers are saved with the weights -- the weights on their own are not
enough to use the model.
"""

import sys
from pathlib import Path

# The simulation lives in robot/. Find it relative to THIS file, so the
# script works no matter which directory you run it from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_data.csv"

import numpy as np
import torch

from train import make_model


def load_measurement_model(path=str(Path(__file__).parent / "baseline_model.pt")):
    """Load the trained network and return a function the UKF can use.

    The returned function takes an array of states, shape (n, 5), and gives
    back predicted sensor readings, shape (n, 3) -- the same shape as
    ukf.expected_readings, so the two are interchangeable.
    """
    saved = torch.load(path, weights_only=False)

    model = make_model()
    model.load_state_dict(saved["weights"])
    model.eval()          # no dropout or batch-norm surprises at inference

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    def measure(states):
        states = np.atleast_2d(states)
        x = torch.tensor(states, dtype=torch.float32)

        # no_grad because we only want the answer, not derivatives.
        # Without it torch builds a graph for every sigma point and never
        # frees it.
        with torch.no_grad():
            scaled = (x - x_mean) / x_std
            out = model(scaled) * y_std + y_mean

        return out.numpy().astype(float)

    return measure


if __name__ == "__main__":
    import time
    from trajectories import DT, random_run
    from sensors import read_sensors
    from ukf import UKF, expected_readings, nis
    from scipy.stats import chi2

    learned = load_measurement_model()

    print("Does the learned model agree with the real formula?")
    run = random_run(seed=0, duration=20.0)
    states = np.column_stack([run["x"], run["y"], run["heading"],
                              run["speed"], run["turn_rate"]])
    exact = expected_readings(states)
    guess = learned(states)
    for i, name in enumerate(["left encoder", "right encoder", "gyro"]):
        print("   %-14s differs by %.4f rad/s"
              % (name, np.sqrt(np.mean((guess[:, i] - exact[:, i]) ** 2))))

    # ---- run the filter both ways ----
    meas = read_sensors(run, seed=0, dt=DT)
    readings = np.column_stack([meas["left_encoder"],
                                meas["right_encoder"],
                                meas["gyro"]])

    Q = np.diag([1e-9, 1e-9, 1e-9, 2e-5, 1e-4])
    R = np.diag([0.15 ** 2, 0.15 ** 2, 0.011 ** 2])
    start_mean = np.array([0.0, 0.0, 0.0, run["speed"][0], run["turn_rate"][0]])
    start_cov = np.diag([0.01, 0.01, 0.01, 0.10, 0.10])

    lo, hi = chi2.ppf(0.025, 3), chi2.ppf(0.975, 3)

    print("\n%-22s %10s %10s %9s %9s %8s"
          % ("measurement model", "speed err", "turn err", "mean NIS",
             "in bounds", "seconds"))
    for label, h in [("written by hand", expected_readings),
                     ("learned network", learned)]:
        t0 = time.time()
        means, covs, innov, S = UKF(Q, R, measure=h).run(
            readings, start_mean, start_cov, DT)
        elapsed = time.time() - t0

        v = nis(innov, S)
        print("%-22s %10.4f %10.4f %9.2f %9.2f %8.1f"
              % (label,
                 np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)),
                 np.sqrt(np.mean((means[:, 4] - run["turn_rate"]) ** 2)),
                 v.mean(),
                 np.mean((v >= lo) & (v <= hi)),
                 elapsed))

    print("\nThe learned model still outputs one number per sensor and says")
    print("nothing about how noisy that reading should be, so R is still a")
    print("fixed matrix chosen by hand. Replacing that is the next step.")

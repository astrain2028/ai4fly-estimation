"""
Uses the residual network as the filter's measurement model.

Same interface as the plain arm, and the same limitation: one number per
sensor, nothing about noise, so R stays a fixed matrix chosen by hand. The
only difference is what is inside the network.

Running this file times the filter as well as scoring it. Depth is not free
on a small computer, and the point of this arm is to find out what it buys
and what it costs.
"""

import sys
from pathlib import Path

# The simulation lives in robot/. Find it relative to THIS file, so the
# script works no matter which directory you run it from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
import torch



def _sibling_train():
    """Load train.py from THIS folder.

    Every arm has a file called train.py, so a plain `import train` picks
    whichever one Python happened to load first -- the plain arm's module
    would satisfy this arm's import, silently. Loading by full path under a
    unique name avoids that.
    """
    import importlib.util
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        here.name + "_train", here / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _sibling_train()
make_model = _train.make_model



def load_measurement_model(path=None):
    """Load the trained network and return a function the UKF can use.

    Takes states, shape (n, 5), and gives back predicted sensor readings,
    shape (n, 3) -- the same shape as ukf.expected_readings, so the two are
    interchangeable.
    """
    if path is None:
        path = Path(__file__).parent / "resnet_model.pt"
    saved = torch.load(path, weights_only=False)

    model = make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    def measure(states):
        states = np.atleast_2d(states)
        x = torch.tensor(states, dtype=torch.float32)

        # no_grad because we only want the answer, not derivatives
        with torch.no_grad():
            out = model((x - x_mean) / x_std) * y_std + y_mean

        return out.numpy().astype(float)

    return measure


if __name__ == "__main__":
    import time
    import importlib.util
    from trajectories import DT, random_run
    from sensors import read_sensors
    from ukf import UKF, expected_readings, nis

    def load_arm(name):
        """Load another arm's measurement.py, by path, for comparison."""
        spec = importlib.util.spec_from_file_location(
            name + "_measurement",
            Path(__file__).resolve().parents[1] / name / "measurement.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_measurement_model()

    deep = load_measurement_model()
    plain = load_arm("plain")

    Q = np.diag([1e-9, 1e-9, 1e-9, 2e-5, 1e-4])
    R = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2])
    start_cov = np.diag([0.01, 0.01, 0.01, 0.10, 0.10])

    print("Filtering twenty runs. Depth is the only difference.\n")
    print("%-24s %10s %10s %9s %9s"
          % ("", "speed err", "turn err", "mean NIS", "seconds"))

    scores = {}
    for label, h in [("written by hand", expected_readings),
                     ("plain network", plain),
                     ("residual network", deep)]:
        speed_err, turn_err, nis_all = [], [], []
        t0 = time.time()
        for seed in range(20):
            run = random_run(seed, duration=20.0)
            meas = read_sensors(run, seed=seed, dt=DT)
            readings = np.column_stack([meas["left_encoder"],
                                        meas["right_encoder"],
                                        meas["gyro"]])
            truth = np.column_stack([run["x"], run["y"], run["heading"],
                                     run["speed"], run["turn_rate"]])

            means, covs, innov, S = UKF(Q, R, measure=h).run(
                readings, truth[0].copy(), start_cov, DT)

            speed_err.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
            turn_err.append(np.sqrt(np.mean((means[:, 4] - run["turn_rate"]) ** 2)))
            nis_all.append(nis(innov, S))
        elapsed = time.time() - t0

        scores[label] = (np.array(speed_err), np.array(turn_err))
        print("%-24s %10.4f %10.4f %9.3f %9.1f"
              % (label, np.mean(speed_err), np.mean(turn_err),
                 np.concatenate(nis_all).mean(), elapsed))

    print("%-24s %10s %10s %9.1f" % ("should be", "", "", 3.0))

    # Same twenty runs went through both networks, so compare them run by
    # run. A mean difference smaller than its own spread is not a result.
    print("\nResidual minus plain, run by run:")
    for i, name in enumerate(["speed", "turn "]):
        diff = scores["residual network"][i] - scores["plain network"][i]
        print("  %s  deeper wins %2d of 20   mean %+.5f   spread %.5f"
              % (name, np.sum(diff < 0), diff.mean(), diff.std()))

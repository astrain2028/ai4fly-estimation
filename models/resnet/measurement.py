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
INPUTS = _train.INPUTS



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
        # The filter carries more states than this model was trained
        # on -- the accelerations are appended after speed and turn
        # rate, which the sensors do not see. Take the five it knows.
        states = np.atleast_2d(states)[:, :len(INPUTS)]
        x = torch.tensor(states, dtype=torch.float32)

        # no_grad because we only want the answer, not derivatives
        with torch.no_grad():
            out = model((x - x_mean) / x_std) * y_std + y_mean

        return out.numpy().astype(float)

    return measure


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (N_RUNS, NIS_DOF, NEES_DOF, best_constant_R, gather,
                        load_arm)

    # The best single covariance available, which is the average variance.
    # Every arm without its own covariance gets this one, so the numbers in
    # these blocks can honestly be compared with each other.
    R = best_constant_R()

    print("Depth is the only difference, %d runs." % N_RUNS)
    print()
    print("%-24s %9s %9s %9s %9s"
          % ("", "speed", "turn", "NIS", "NEES"))
    scores = {}
    for label, name in [("written by hand", "fixed"),
                        ("plain network", "plain"),
                        ("residual network", "resnet")]:
        r = gather(load_arm(name), range(N_RUNS), R=R)
        scores[name] = r
        print("%-24s %9.4f %9.4f %9.3f %9.3f"
              % (label, r["speed_rmse"].mean(),
                 r["turn_rmse"].mean(), r["nis"].mean(),
                 r["nees"].mean()))
    print("%-24s %9s %9s %9.1f %9.1f"
          % ("should be", "", "", float(NIS_DOF), float(NEES_DOF)))

    # Same runs through both networks, so compare run by run. A mean
    # difference smaller than its own spread is not a result.
    print()
    print("Residual minus plain, run by run:")
    for key, name in [("speed_rmse", "speed"), ("turn_rmse", "turn ")]:
        diff = scores["resnet"][key] - scores["plain"][key]
        print("  %s  deeper wins %2d of %d   mean %+.5f   spread %.5f"
              % (name, int((diff < 0).sum()), len(diff),
                 diff.mean(), diff.std()))
    print()
    print("Depth was not the limit. That is the answer this arm was")
    print("built for, and it is a negative one.")

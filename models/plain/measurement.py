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
make_model = _train.make_model
INPUTS = _train.INPUTS



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
        # The filter carries more states than this model was trained
        # on -- the accelerations are appended after speed and turn
        # rate, which the sensors do not see. Take the five it knows.
        states = np.atleast_2d(states)[:, :len(INPUTS)]
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
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (N_RUNS, NIS_DOF, NEES_DOF, best_constant_R, gather,
                        load_arm)

    # The best single covariance available, which is the average variance.
    # Every arm without its own covariance gets this one, so the numbers in
    # these blocks can honestly be compared with each other.
    R = best_constant_R()

    print("Learned mean against the hand-written one, %d runs." % N_RUNS)
    print()
    print("%-24s %9s %9s %9s %9s"
          % ("", "speed", "turn", "NIS", "NEES"))
    for label, name in [("written by hand", "fixed"), ("learned network", "plain")]:
        r = gather(load_arm(name), range(N_RUNS), R=R)
        print("%-24s %9.4f %9.4f %9.3f %9.3f"
              % (label, r["speed_rmse"].mean(),
                 r["turn_rmse"].mean(), r["nis"].mean(),
                 r["nees"].mean()))
    print("%-24s %9s %9s %9.1f %9.1f"
          % ("should be", "", "", float(NIS_DOF), float(NEES_DOF)))

    print()
    print("The hand-written model is the function the simulator itself")
    print("uses, so it is not an approximation of the truth -- it is the")
    print("truth. A learned mean can tie it at best. What a learned model")
    print("can offer instead is a covariance, and this arm has none.")

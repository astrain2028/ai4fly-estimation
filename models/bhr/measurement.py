"""
Uses the heteroscedastic model as the filter's measurement model.

Unlike the plain network, this one supplies its own noise. It returns both
the expected readings and a covariance for each sample point, and the filter
uses those instead of a fixed R.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

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
split_outputs = _train.split_outputs
to_mean_and_var = _train.to_mean_and_var
OUTPUTS = _train.OUTPUTS
INPUTS = _train.INPUTS



def load_measurement_model(path=None, use_laplace=True):
    """Load the trained model and return a function the filter can call.

    The returned function takes states, shape (n, 5), and gives back
    readings, shape (n, 3), together with a covariance for each state,
    shape (n, 3, 3). Those covariances are diagonal: the model treats the
    three sensors as independent, which they are here.

    If laplace.py has been run, its posterior is loaded and the covariance
    gains a second term -- the model's uncertainty about its own weights, on
    top of the sensor noise it predicts. See laplace.py for what that is and
    why it is not the same quantity.
    """
    if path is None:
        path = Path(__file__).parent / "bhr_model.pt"
    saved = torch.load(path, weights_only=False)

    model = make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    posteriors = None
    if use_laplace:
        fitted = Path(__file__).parent / "bhr_laplace.npz"
        if fitted.exists():
            posteriors = np.load(fitted)["posteriors"]

    # The network is split in two so it can be run once and used twice: the
    # last layer needs the hidden features to produce an answer, and the
    # posterior needs the same features to say how sure it is. Calling the
    # whole network and then calling most of it again -- which is what
    # laplace.py's own demo does -- doubles the cost for nothing.
    trunk, head = model[:-1], model[-1]

    def measure(states):
        # The filter carries more states than this model was trained
        # on -- the accelerations are appended after speed and turn
        # rate, which the sensors do not see. Take the five it knows.
        states = np.atleast_2d(states)[:, :len(INPUTS)]
        x = torch.tensor(states, dtype=torch.float32)

        with torch.no_grad():
            hidden = trunk((x - x_mean) / x_std)
            eta1, eta2 = split_outputs(head(hidden), False)
            mean_s, var_s = to_mean_and_var(eta1, eta2)

        variances_s = var_s.numpy().astype(np.float64)

        if posteriors is not None:
            # features with a trailing 1, so the last layer's bias is treated
            # as uncertain like every other weight
            phi = np.concatenate(
                [hidden.numpy().astype(np.float64), np.ones((len(states), 1))],
                axis=1)
            for j in range(len(OUTPUTS)):
                spread_eta1 = np.einsum("ni,ij,nj->n", phi, posteriors[j], phi)
                # eta1 is the natural parameter; the mean is eta1 * variance,
                # so its spread carries through squared
                variances_s[:, j] += variances_s[:, j] ** 2 * spread_eta1

        # back to real units. A mean scales by the spread used to standardise
        # it; a variance scales by the square of that spread.
        readings = (mean_s * y_std + y_mean).numpy().astype(float)
        variances = variances_s * (y_std.numpy() ** 2)

        R = np.zeros((len(states), len(OUTPUTS), len(OUTPUTS)))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    measure.has_laplace = posteriors is not None
    return measure


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (N_RUNS, NIS_DOF, NEES_DOF, best_constant_R, gather,
                        load_arm)

    # The best single covariance available, which is the average variance.
    # Every arm without its own covariance gets this one, so the numbers in
    # these blocks can honestly be compared with each other.
    R = best_constant_R()

    measure = load_measurement_model()
    print("Heteroscedastic model, Laplace %s, %d runs."
          % ("fitted" if measure.has_laplace else "NOT fitted", N_RUNS))
    print()
    print("%-24s %9s %9s %9s %9s"
          % ("", "speed", "turn", "NIS", "NEES"))
    for label, name in [("written by hand", "fixed"), ("heteroscedastic", "bhr")]:
        r = gather(load_arm(name), range(N_RUNS), R=R)
        print("%-24s %9.4f %9.4f %9.3f %9.3f"
              % (label, r["speed_rmse"].mean(),
                 r["turn_rmse"].mean(), r["nis"].mean(),
                 r["nees"].mean()))
    print("%-24s %9s %9s %9.1f %9.1f"
          % ("should be", "", "", float(NIS_DOF), float(NEES_DOF)))

    print()
    print("The covariance here is never tuned -- it comes from the model")
    print("and changes at every step. On healthy, correctly calibrated")
    print("data that buys nothing over the best constant, which is what R")
    print("above is. Where it earns its keep is measured in experiments/.")

"""
Uses the five-model ensemble as the filter's measurement model.

The covariance handed to the filter is the sum of two things: the noise the
members claim, and the amount they disagree with each other. The filter only
ever sees the total, but the two parts are kept separate and reachable
through `measure.decompose(states)`, because which one grew is the whole
question a fault experiment is asking.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
import torch


def _bhr():
    """The members are bhr models, so load that arm's definitions by path."""
    spec = importlib.util.spec_from_file_location(
        "bhr_train_for_ensemble_measure", ROOT / "models" / "bhr" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _bhr()
INPUTS = bhr.INPUTS


def load_measurement_model(path=None):
    """Load all five members and return a function the filter can call.

    Gives back readings, shape (n, 3), and a covariance per state,
    shape (n, 3, 3) -- the same interface as the single heteroscedastic arm,
    so the filter cannot tell the difference.
    """
    if path is None:
        path = Path(__file__).parent / "ensemble_model.pt"
    saved = torch.load(path, weights_only=False)

    models = []
    for state in saved["weights"]:
        model = bhr.make_model()
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]
    n_out = len(saved["outputs"])

    def both_parts(states):
        """Every member's answer, in real units."""
        # The filter carries more states than this model was trained
        # on -- the accelerations are appended after speed and turn
        # rate, which the sensors do not see. Take the five it knows.
        states = np.atleast_2d(states)[:, :len(INPUTS)]
        x = torch.tensor(states, dtype=torch.float32)
        xs = (x - x_mean) / x_std

        means, variances = [], []
        with torch.no_grad():
            for model in models:
                eta1, eta2 = bhr.split_outputs(model(xs), False)
                m, v = bhr.to_mean_and_var(eta1, eta2)
                means.append((m * y_std + y_mean).numpy())
                variances.append((v * y_std ** 2).numpy())

        means = np.array(means)
        variances = np.array(variances)
        return (means.mean(axis=0),        # the answer
                variances.mean(axis=0),    # aleatoric: claimed sensor noise
                means.var(axis=0))         # epistemic: disagreement

    def measure(states):
        readings, aleatoric, epistemic = both_parts(states)
        total = aleatoric + epistemic

        R = np.zeros((len(readings), n_out, n_out))
        for k in range(len(readings)):
            R[k] = np.diag(total[k])
        return readings.astype(float), R

    def decompose(states):
        """The two parts separately, for analysis rather than filtering."""
        readings, aleatoric, epistemic = both_parts(states)
        return {"readings": readings,
                "aleatoric": aleatoric,
                "epistemic": epistemic}

    measure.decompose = decompose
    measure.members = len(models)
    return measure


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (N_RUNS, NIS_DOF, NEES_DOF, gather, two_moment,
                        load_arm, make_run)

    measure = load_arm("ensemble")
    single = load_arm("bhr")

    print("Ensemble of %d against a single heteroscedastic model, %d runs\n"
          % (measure.members, N_RUNS))
    print("%-22s %10s %10s %9s %9s %10s"
          % ("", "speed err", "turn err", "NIS", "NEES", "ms/step"))

    for label, fn in [("single model", single), ("ensemble", measure)]:
        r = gather(fn, range(N_RUNS))
        print("%-22s %10.4f %10.4f %9.3f %9.3f %10.3f"
              % (label, r["speed_rmse"].mean(), r["turn_rmse"].mean(),
                 r["nis"].mean(), r["nees"].mean(), r["ms_per_step"]))
    print("%-22s %10s %10s %9.1f %9.1f"
          % ("should be", "", "", float(NIS_DOF), float(NEES_DOF)))

    # What the extra four networks actually bought
    _, meas, truth = make_run(0)
    parts = measure.decompose(truth)
    print("\nOn states the members were trained around:")
    print("  %-15s %12s %12s %10s"
          % ("", "sensor noise", "model doubt", "doubt %"))
    for i, name in enumerate(["left_encoder", "right_encoder", "gyro"]):
        a, e = parts["aleatoric"][:, i].mean(), parts["epistemic"][:, i].mean()
        print("  %-15s %12.6f %12.6f %9.2f%%"
              % (name, a, e, 100 * e / (a + e)))

    print("\n  Small doubt here is correct -- these are ordinary states.")
    print("  The cost of measuring it is the ms/step column above.")

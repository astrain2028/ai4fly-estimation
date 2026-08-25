"""
The health-conditioned measurement model: the thesis, in one file.

Every other arm learns h(x) -- what the sensors should read given where the
robot is and how fast it is going. This one learns

    h(x, m)  and  R(x, m)

where m is how degraded each sensor is. At m = 0 it must reproduce the
healthy relationship. As m rises it reproduces whatever that degradation does
to the reading, with intermediate values interpolating.

WHY THIS IS THE WHOLE POINT

A model taking only the state cannot respond to a fault. The fault arrives in
the measurement; the state looks entirely ordinary; the model returns the
same answer it always did. That is not a shortcoming of any particular
network, it is arithmetic, and it was measured directly: the heteroscedastic
arm's predicted spread moved from 0.1505 to 0.1506 while the true noise went
up fourfold.

Giving the model m as an input is what creates the path. Then the filter can
carry m as a state, spread its sample points over it, and see the predicted
readings change -- and a predicted reading that changes with health is
exactly what lets the update move health.

WHAT TO EXPECT, AND WHAT NOT TO

The filter estimates health through the covariance between its sample points'
health and their predicted readings. That covariance is non-zero only where
the predicted reading actually depends on health.

    a bias fault              shifts the reading      -> h depends on m
    noise inflation           leaves the reading      -> h does NOT depend
                              centred as before          on m; only R does

So a bias should be estimable and pure noise inflation should not, under an
update that uses only the first moment of the innovation. Both modes are in
the training data so that this can be measured rather than argued about. A
negative result on the second is a real property of the formulation, and it
points at what a filter would need instead -- an update that reads the size
of its innovations, not only their direction.

Heteroscedastic throughout, in the natural parameterisation, for the reason
bhr/train.py explains: under a plain mean-and-variance parameterisation the
gradient on the mean is divided by the predicted variance, so the model can
talk itself into ignoring the data.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_faulted.csv"

import numpy as np
import pandas as pd
import torch


def _bhr():
    """Reuse the heteroscedastic machinery rather than copying it."""
    spec = importlib.util.spec_from_file_location(
        "bhr_train_for_health", ROOT / "models" / "bhr" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _bhr()

VEHICLE = ["x", "y", "heading", "speed", "turn_rate"]
HEALTH = ["severity_left", "severity_right", "severity_gyro"]
INPUTS = VEHICLE + HEALTH
OUTPUTS = ["left_encoder", "right_encoder", "gyro"]

EPOCHS = 80
WARMUP = 15
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN = 64


def make_model():
    """Eight inputs now: five vehicle states and three health levels."""
    return torch.nn.Sequential(
        torch.nn.Linear(len(INPUTS), HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 2 * len(OUTPUTS)),
    )


def load_and_split(path=DATA, val_fraction=0.2, seed=0):
    """Hold out whole runs, and hold out healthy and faulted ones alike.

    Splitting at random would be adequate on average and occasionally leave
    one condition badly represented in validation. Sampling within each fault
    mode keeps both sides looking like the whole.
    """
    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)

    val_runs = []
    per_run = df.groupby("run")["fault_mode"].first()
    for mode in per_run.unique():
        # copy() because a pandas index hands back a read-only view
        runs = per_run[per_run == mode].index.to_numpy().copy()
        rng.shuffle(runs)
        val_runs.extend(runs[:max(1, int(len(runs) * val_fraction))])

    is_val = df["run"].isin(val_runs)
    train, val = df[~is_val], df[is_val]
    print("train: %d rows from %d runs" % (len(train), train["run"].nunique()))
    print("val:   %d rows from %d runs" % (len(val), val["run"].nunique()))
    return train, val


def main():
    torch.manual_seed(0)
    if not DATA.exists():
        print("No %s -- run robot/make_faulted.py first." % DATA.name)
        return

    train_df, val_df = load_and_split()

    x = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    y = torch.tensor(train_df[OUTPUTS].values, dtype=torch.float32)
    xv = torch.tensor(val_df[INPUTS].values, dtype=torch.float32)
    yv = torch.tensor(val_df[OUTPUTS].values, dtype=torch.float32)

    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)

    model = make_model()
    print("\nTraining")
    bhr.train(model, (x - x_mean) / x_std, (y - y_mean) / y_std,
              (xv - x_mean) / x_std, (yv - y_mean) / y_std)

    with torch.no_grad():
        eta1, eta2 = bhr.split_outputs(model((xv - x_mean) / x_std), False)
        mean_s, var_s = bhr.to_mean_and_var(eta1, eta2)
    pred_mean = (mean_s * y_std + y_mean).numpy()
    pred_std = (var_s.sqrt() * y_std).numpy()

    print("\nDoes the model use the health input at all?")
    print("Held-out rows, split by whether the left encoder was degraded.\n")
    print("  %-22s %10s %12s %12s"
          % ("", "rows", "mean error", "claimed sd"))
    severity = val_df["severity_left"].values
    for label, rows in [("left healthy", severity == 0),
                        ("left degraded", severity > 0)]:
        if not rows.any():
            continue
        err = pred_mean[rows, 0] - val_df["left_encoder"].values[rows]
        print("  %-22s %10d %12.4f %12.4f"
              % (label, rows.sum(), err.mean(), pred_std[rows, 0].mean()))

    print("\nBy fault mode, on the degraded rows only:\n")
    print("  %-22s %10s %12s %12s"
          % ("", "rows", "mean shift", "claimed sd"))
    modes = val_df["fault_mode"].values
    for mode in ["bias", "noise_inflation"]:
        rows = (severity > 0) & (modes == mode)
        if not rows.any():
            continue
        shift = (pred_mean[rows, 0]
                 - val_df["left_spin"].values[rows]).mean()
        print("  %-22s %10d %12.4f %12.4f"
              % (mode, rows.sum(), shift, pred_std[rows, 0].mean()))

    print("\n  A bias should move the mean column and a noise fault should")
    print("  move the spread column. If both move only one, the model has")
    print("  learned to treat every fault the same way, and health will not")
    print("  be separable by mode.")

    torch.save({"weights": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "inputs": INPUTS, "outputs": OUTPUTS},
               str(Path(__file__).parent / "health_model.pt"))
    print("\nSaved health_model.pt")


if __name__ == "__main__":
    main()

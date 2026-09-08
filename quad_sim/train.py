"""
The health-conditioned measurement model, on a map that is actually nonlinear.

WHAT IS DIFFERENT FROM THE GROUND ROBOT

Everything about the formulation is the same: learn h(x, m) and R(x, m) where m
is how degraded each device is, carry m as filter state, and let the covariance
between a sigma point's health and its predicted reading move the estimate.

What differs is that here the learning has something to do. On the robot the
measurement map is exactly linear, and an ordinary least-squares fit beat the
trained network on every channel -- so whatever the learned arms bought there,
it was not the ability to represent a nonlinear function. linearity.py measures
the same thing here and finds a linear fit leaves twelve times the noise floor
unexplained on the magnetometer channels.

So this is where the two halves of the claim can finally be separated:

    a learned MAP earns its place when h is nonlinear
    a learned COVARIANCE earns its place when the noise varies with state

The robot could only test the second. This tests both.

THE ARCHITECTURE IS WIDER, AND THAT IS NOT A FREE CHOICE

The robot's model has 64 hidden units for a target that is a three-by-two
matrix, which is enormous overkill -- and the deeper residual control was
measurably worse there, because a ReLU stack approximates a straight line with
kinks that have to cancel. Capacity was never the binding constraint.

Here the target is trigonometric in three angles and the input is twelve wide,
so capacity plausibly is binding. 128 units rather than 64. That is a real
difference between the two problems rather than a knob, but it should still be
checked rather than assumed, which is what a capacity control is for.

HETEROSCEDASTIC IN NATURAL PARAMETERS

Reused wholesale from models/bhr. The gradient pathology Seitzer et al.
describe matters more here than in ordinary regression: under a plain
mean-and-variance parameterisation the gradient on the mean is divided by the
predicted variance, so a model can lower its loss by declaring data noisy and
giving up on predicting it. For a fault model the faulted samples are precisely
the inconvenient ones, so that escape route would let it learn "broken sensors
are just noisy" and never learn to predict the shifted reading at all.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data" / "quad_faulted.csv"

import numpy as np
import pandas as pd
import torch

from make_dataset import SEVERITY_COLUMNS, STATES
from faults import REFERENCE
from sensors import CHANNELS, ideal_readings


def _bhr():
    """Reuse the heteroscedastic machinery rather than copying it."""
    spec = importlib.util.spec_from_file_location(
        "bhr_train_for_quad", ROOT / "models" / "bhr" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _bhr()

INPUTS = STATES + SEVERITY_COLUMNS      # 6 states + 6 health levels
OUTPUTS = CHANNELS                      # 9 channels

EPOCHS = 120
WARMUP = 20
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN = 128


def make_model():
    """Twelve inputs, eighteen outputs -- two numbers per channel."""
    return torch.nn.Sequential(
        torch.nn.Linear(len(INPUTS), HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 2 * len(OUTPUTS)),
    )


def load_and_split(path=DATA, val_fraction=0.2, seed=0):
    """Hold out whole flights, stratified on which device broke and how.

    Splitting at random would be adequate on average and occasionally leave a
    device badly represented in validation. Stratifying on the pair (which
    device, which modes) keeps both sides looking like the whole.

    Peak severity rather than the first row's: half of faulted runs begin
    healthy and degrade partway through, and reading row zero would file every
    one of them under healthy.
    """
    df = pd.read_csv(path)
    rng = np.random.default_rng(seed)

    peak = df.groupby("run")[SEVERITY_COLUMNS].max()
    kind = []
    for column in SEVERITY_COLUMNS:
        kind.append((peak[column] > 0).astype(int).astype(str))
    label = kind[0]
    for extra in kind[1:]:
        label = label + extra

    val_runs = []
    for group in label.unique():
        runs = label[label == group].index.to_numpy().copy()
        rng.shuffle(runs)
        val_runs.extend(runs[:max(1, int(len(runs) * val_fraction))])

    is_val = df["run"].isin(val_runs)
    train, val = df[~is_val], df[is_val]
    print("train: %d rows from %d flights"
          % (len(train), train["run"].nunique()))
    print("val:   %d rows from %d flights" % (len(val), val["run"].nunique()))
    return train, val


def main():
    torch.manual_seed(0)
    if not DATA.exists():
        print("No %s -- run make_dataset.py first." % DATA.name)
        return 1

    train_df, val_df = load_and_split()

    x = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    y = torch.tensor(train_df[OUTPUTS].values, dtype=torch.float32)
    xv = torch.tensor(val_df[INPUTS].values, dtype=torch.float32)
    yv = torch.tensor(val_df[OUTPUTS].values, dtype=torch.float32)

    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)

    model = make_model()
    print("\n%d inputs, %d outputs, %d weights"
          % (len(INPUTS), 2 * len(OUTPUTS),
             sum(p.numel() for p in model.parameters())))

    print("\nTraining")
    bhr.EPOCHS, bhr.WARMUP = EPOCHS, WARMUP
    bhr.OUTPUTS = OUTPUTS
    bhr.train(model, (x - x_mean) / x_std, (y - y_mean) / y_std,
              (xv - x_mean) / x_std, (yv - y_mean) / y_std)

    with torch.no_grad():
        eta1, eta2 = bhr.split_outputs(model((xv - x_mean) / x_std), False)
        mean_s, var_s = bhr.to_mean_and_var(eta1, eta2)
    pred_mean = (mean_s * y_std + y_mean).numpy()
    pred_std = (var_s.sqrt() * y_std).numpy()

    # The comparison that matters: against the best linear model, on the same
    # held-out rows. On the ground robot least squares won.
    design = np.hstack([train_df[INPUTS].values,
                        np.ones((len(train_df), 1))])
    weights, *_ = np.linalg.lstsq(design, train_df[OUTPUTS].values,
                                  rcond=None)
    linear = np.hstack([val_df[INPUTS].values,
                        np.ones((len(val_df), 1))]) @ weights

    print("\n\nAGAINST THE BEST LINEAR MODEL, HELD-OUT FLIGHTS\n")
    print("  %-10s %12s %12s %12s" % ("channel", "linear", "network",
                                      "network better"))
    print("  " + "-" * 50)
    truth = val_df[OUTPUTS].values
    wins = 0
    for j, name in enumerate(OUTPUTS):
        rl = np.sqrt(np.mean((linear[:, j] - truth[:, j]) ** 2))
        rn = np.sqrt(np.mean((pred_mean[:, j] - truth[:, j]) ** 2))
        wins += rn < rl
        print("  %-10s %12.5f %12.5f %11.1f%%"
              % (name, rl, rn, 100 * (rl - rn) / rl))
    print("  " + "-" * 50)
    print("\n  network ahead on %d of %d channels" % (wins, len(OUTPUTS)))
    print("\n  On the ground robot this table goes the other way on every")
    print("  channel, by three to seven per cent, because the map there is a")
    print("  matrix and a ReLU stack only approximates one.")

    print("\n\nDOES EACH HEALTH INPUT DRIVE THE RIGHT OUTPUT?\n")
    print("  Held-out rows, accelerometer, by what the flight carried.\n")
    print("  %-16s %8s %10s %12s %10s %12s"
          % ("", "rows", "severity", "mean shift", "wanted", "claimed sd"))

    bias = val_df["severity_bias_accel"].values
    noise = val_df["severity_noise_accel"].values

    # The reading the same attitude would have produced with nothing broken.
    # An earlier version compared against the faulted reading in the dataframe,
    # which makes the column a residual that is near zero however the model
    # behaves -- it cannot distinguish a bias that was corrected from one that
    # was ignored. The correction has to be measured against the healthy
    # reading it is supposed to move away from.
    states = val_df[STATES].values
    clean = ideal_readings(states[:, 0], states[:, 1], states[:, 2],
                           states[:, 3], states[:, 4], states[:, 5])

    for label, rows in [("healthy", (bias == 0) & (noise == 0)),
                        ("bias only", (bias > 0) & (noise == 0)),
                        ("noise only", (bias == 0) & (noise > 0)),
                        ("both", (bias > 0) & (noise > 0))]:
        if not rows.any():
            continue
        shift = (pred_mean[rows, :3] - clean[rows, :3]).mean()
        wanted = bias[rows].mean() * REFERENCE["accel"]
        level = np.maximum(bias[rows], noise[rows]).mean()
        print("  %-16s %8d %10.2f %12.4f %10.4f %12.4f"
              % (label, rows.sum(), level, shift, wanted,
                 pred_std[rows, :3].mean()))

    print("\n  The bias row should move the mean-shift column and leave the")
    print("  spread alone; the noise row should do the opposite. If they do")
    print("  not separate, splitting health into two numbers bought nothing.")

    torch.save({"weights": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "inputs": INPUTS, "outputs": OUTPUTS},
               str(HERE / "quad_health.pt"))
    print("\nSaved quad_health.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

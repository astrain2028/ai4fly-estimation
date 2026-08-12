"""
Heteroscedastic model: predicts each sensor reading AND how noisy it is.

The plain model gives one number per sensor. This one gives two: the
expected reading, and the spread around it. That second number is what a
filter needs for R, and it changes from moment to moment -- more noise when
the wheels spin fast, less when they crawl.

WHY NOT JUST ADD A SECOND OUTPUT

The obvious way is to predict mu and var directly and train on

    loss = 0.5 * ( log(var) + (y - mu)^2 / var )

That breaks. Differentiating with respect to mu gives

    d(loss)/d(mu) = -(y - mu) / var

so the pull on the mean is divided by the predicted variance. Wherever the
model guesses a large variance, the mean stops being corrected, the error
stays large, and that large error then justifies an even larger variance.
The model talks itself into ignoring part of the data.

The fix is to predict a different pair of numbers -- the "natural
parameters" of a Gaussian:

    eta1 = mu / var
    eta2 = -1 / (2 var)

The same distribution, written differently. In these coordinates the loss
has no such feedback, and the mean keeps being fitted everywhere. Converting
back is one line each way.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_data.csv"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUTS = ["x", "y", "heading", "speed", "turn_rate"]
OUTPUTS = ["left_encoder", "right_encoder", "gyro"]
TRUE_NOISE = ["left_noise", "right_noise", "gyro_noise"]

EPOCHS = 80
WARMUP = 15          # epochs spent fitting the mean before the spread moves
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN = 64


def load_and_split(path=DATA, val_fraction=0.2, seed=0):
    """Hold out whole runs, never individual rows -- each run shares one
    gyro bias, so splitting rows would leak it across the divide."""
    df = pd.read_csv(path)
    runs = df["run"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    val_runs = runs[:int(len(runs) * val_fraction)]
    is_val = df["run"].isin(val_runs)
    print("train: %d rows from %d runs" % ((~is_val).sum(), df[~is_val]["run"].nunique()))
    print("val:   %d rows from %d runs" % (is_val.sum(), df[is_val]["run"].nunique()))
    return df[~is_val], df[is_val]


def make_model():
    """Two numbers per sensor instead of one, so six outputs for three
    sensors. The first three are eta1, the last three eta2."""
    return nn.Sequential(
        nn.Linear(len(INPUTS), HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 2 * len(OUTPUTS)),
    )


def split_outputs(raw, freeze_spread=False):
    """Turn the network's six raw numbers into (eta1, eta2).

    eta2 must be negative, since it is -1/(2*var) and variance is positive.
    softplus is always positive, so negating it guarantees the sign.

    During warmup eta2 is pinned to -0.5, which means variance = 1. The
    model is then just fitting a mean, with nothing to hide behind.
    """
    n = len(OUTPUTS)
    eta1 = raw[:, :n]
    if freeze_spread:
        eta2 = torch.full_like(eta1, -0.5)
    else:
        eta2 = -torch.nn.functional.softplus(raw[:, n:]) - 1e-6
    return eta1, eta2


def to_mean_and_var(eta1, eta2):
    """Back to the numbers we actually want."""
    var = -0.5 / eta2
    mean = eta1 * var
    return mean, var


def natural_loss(eta1, eta2, y):
    """Negative log likelihood, written in natural parameters.

    log p(y) = eta1*y + eta2*y^2 + eta1^2/(4*eta2) - 0.5*log(pi / -eta2)
    """
    log_p = (eta1 * y + eta2 * y ** 2
             + eta1 ** 2 / (4 * eta2)
             - 0.5 * torch.log(np.pi / (-eta2)))
    return -log_p.mean()


def train(model, x_train, y_train, x_val, y_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n = len(x_train)

    for epoch in range(1, EPOCHS + 1):
        warming_up = epoch <= WARMUP
        order = torch.randperm(n)
        total, batches = 0.0, 0

        for start in range(0, n, BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            eta1, eta2 = split_outputs(model(x_train[rows]), warming_up)
            loss = natural_loss(eta1, eta2, y_train[rows])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1

        if epoch % 20 == 0 or epoch == 1 or epoch == WARMUP:
            with torch.no_grad():
                e1, e2 = split_outputs(model(x_val), False)
                val = natural_loss(e1, e2, y_val).item()
            note = "  (warmup: spread frozen)" if warming_up else ""
            print("  epoch %3d   train %.4f   val %.4f%s"
                  % (epoch, total / batches, val, note))


def main():
    torch.manual_seed(0)
    train_df, val_df = load_and_split()

    x_train = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[OUTPUTS].values, dtype=torch.float32)
    x_val = torch.tensor(val_df[INPUTS].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[OUTPUTS].values, dtype=torch.float32)

    x_mean, x_std = x_train.mean(0), x_train.std(0)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    xs_train = (x_train - x_mean) / x_std
    xs_val = (x_val - x_mean) / x_std
    ys_train = (y_train - y_mean) / y_std
    ys_val = (y_val - y_mean) / y_std

    model = make_model()
    print("\nTraining")
    train(model, xs_train, ys_train, xs_val, ys_val)

    with torch.no_grad():
        eta1, eta2 = split_outputs(model(xs_val), False)
        mean_s, var_s = to_mean_and_var(eta1, eta2)
    # back to real units: a mean scales by y_std, a variance by y_std squared
    pred_mean = (mean_s * y_std + y_mean).numpy()
    pred_std = (var_s.sqrt() * y_std).numpy()

    print("\nMean accuracy on held-out runs (compare with the plain model)")
    for i, name in enumerate(OUTPUTS):
        err = np.sqrt(np.mean((pred_mean[:, i] - val_df[name].values) ** 2))
        print("   %-15s off by %.4f" % (name, err))

    print("\nDoes the predicted spread match the real noise?")
    print("   %-15s %10s %10s" % ("", "predicted", "actual"))
    for i, (name, truth) in enumerate(zip(OUTPUTS, TRUE_NOISE)):
        actual = val_df[truth].values
        if "encoder" in name:      # encoders also carry tick rounding
            tick = (2 * np.pi / 1024) / 0.02
            actual = np.sqrt(actual ** 2 + (tick ** 2) / 12)
        print("   %-15s %10.4f %10.4f" % (name, pred_std[:, i].mean(), actual.mean()))

    print("\nDoes the spread track the state, or is it just an average?")
    spin = np.abs(val_df["left_spin"].values)
    slow = spin < np.percentile(spin, 25)
    fast = spin > np.percentile(spin, 75)
    print("   left encoder, slow wheels: predicted %.4f   actual %.4f"
          % (pred_std[slow, 0].mean(),
             np.sqrt(val_df["left_noise"].values[slow] ** 2
                     + ((2 * np.pi / 1024) / 0.02) ** 2 / 12).mean()))
    print("   left encoder, fast wheels: predicted %.4f   actual %.4f"
          % (pred_std[fast, 0].mean(),
             np.sqrt(val_df["left_noise"].values[fast] ** 2
                     + ((2 * np.pi / 1024) / 0.02) ** 2 / 12).mean()))
    print("   A single fixed number cannot do both. That is the whole point.")

    torch.save({"weights": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "inputs": INPUTS, "outputs": OUTPUTS},
               str(Path(__file__).parent / "bhr_model.pt"))
    print("\nSaved bhr_model.pt")


if __name__ == "__main__":
    main()

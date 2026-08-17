"""
Trains a deeper network, built from residual blocks, on the same job as the
plain baseline.

WHY THIS ARM EXISTS

The plain network is slightly worse than the formula written by hand, even
though it is approximating a relationship we know exactly. Two explanations
fit: the network is too small to represent it, or the leftover error is a
floor that no network gets under. Those call for different responses, so it
is worth telling them apart.

This arm changes one thing and nothing else -- same data, same split, same
loss, same optimiser, same seed, same number of epochs. Only the shape of the
network differs. If the error drops, the baseline was short on capacity. If
it does not, the floor is real and adding capacity is not the answer.

Input:  x, y, heading, speed, turn_rate
Output: left encoder, right encoder, gyro

Like the plain arm, this one predicts a single number per sensor and says
nothing about noise, so the filter still needs a fixed R.
"""

import sys
from pathlib import Path

# The simulation lives in robot/. Find it relative to THIS file, so the
# script works no matter which directory you run it from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_data.csv"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUTS = ["x", "y", "heading", "speed", "turn_rate"]
OUTPUTS = ["left_encoder", "right_encoder", "gyro"]

# Matched to the plain arm so the comparison is fair.
EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN = 64

BLOCKS = 4          # each block holds two layers, so this is 8 hidden layers


def load_and_split(path=DATA, val_fraction=0.2, seed=0):
    """Load the data and hold out whole runs for validation.

    Splitting by run matters. Every row in a run shares that run's gyro
    bias, so if rows from one run land on both sides, the network sees the
    bias during training and the validation score comes out too good.
    """
    df = pd.read_csv(path)
    runs = df["run"].unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    n_val = int(len(runs) * val_fraction)
    val_runs = runs[:n_val]

    is_val = df["run"].isin(val_runs)
    train = df[~is_val]
    val = df[is_val]
    print("train: %d rows from %d runs" % (len(train), train["run"].nunique()))
    print("val:   %d rows from %d runs" % (len(val), val["run"].nunique()))
    return train, val


def to_tensors(frame):
    x = torch.tensor(frame[INPUTS].values, dtype=torch.float32)
    y = torch.tensor(frame[OUTPUTS].values, dtype=torch.float32)
    return x, y


class ResidualBlock(nn.Module):
    """Two layers, plus a shortcut carrying the input straight through.

    A normal layer has to produce the whole answer. This block only has to
    produce a *correction*, which is added to what came in. That matters when
    you stack many of them: a block with nothing useful to add can settle on
    outputting near zero, and the input passes through unchanged. Depth that
    is not needed costs little instead of getting in the way.

    Without the shortcut, deep stacks train badly -- the signal used to adjust
    the early layers has to travel back through every later layer and shrinks
    on the way. The shortcut gives it a direct path.
    """

    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


def make_model():
    """5 inputs, a stack of residual blocks, 3 outputs."""
    return nn.Sequential(
        nn.Linear(len(INPUTS), HIDDEN),
        nn.ReLU(),
        *[ResidualBlock(HIDDEN) for _ in range(BLOCKS)],
        nn.Linear(HIDDEN, len(OUTPUTS)),
    )


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def train(model, x_train, y_train, x_val, y_val):
    """Standard training loop: predict, measure error, adjust, repeat."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    n = len(x_train)

    for epoch in range(1, EPOCHS + 1):
        # shuffle the rows each epoch so batches differ
        order = torch.randperm(n)

        total_loss = 0.0
        batches = 0
        for start in range(0, n, BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]

            predictions = model(x_train[rows])
            loss = loss_function(predictions, y_train[rows])

            optimizer.zero_grad()   # clear last step's gradients
            loss.backward()         # work out which way to adjust weights
            optimizer.step()        # adjust them

            total_loss += loss.item()
            batches += 1

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = loss_function(model(x_val), y_val).item()
            print("  epoch %3d   train loss %.4f   val loss %.4f"
                  % (epoch, total_loss / batches, val_loss))


def main():
    torch.manual_seed(0)

    train_df, val_df = load_and_split()
    x_train, y_train = to_tensors(train_df)
    x_val, y_val = to_tensors(val_df)

    # Put every column on a similar scale, using the training set only so
    # nothing about the validation runs leaks in.
    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)

    x_train_scaled = (x_train - x_mean) / x_std
    x_val_scaled = (x_val - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std
    y_val_scaled = (y_val - y_mean) / y_std

    model = make_model()
    print("\n%d residual blocks, %d hidden units, %d weights"
          % (BLOCKS, HIDDEN, count_weights(model)))
    print("(the plain baseline has 4,867)")

    print("\nTraining")
    train(model, x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled)

    # Predictions, converted back to real units
    with torch.no_grad():
        predictions = model(x_val_scaled) * y_std + y_mean
    predictions = predictions.numpy()

    # The best any predictor can do is the size of the noise, because noise
    # is random and cannot be predicted. Three separate things add to it,
    # and they combine as the square root of the sum of squares.
    tick = (2 * np.pi / 1024) / 0.02          # one encoder tick per time step
    quantisation = tick / np.sqrt(12)          # spread of rounding to ticks
    bias_spread = val_df.groupby("run")["gyro_bias"].first().std()

    floors = {
        "left_encoder": np.sqrt(val_df["left_noise"].mean() ** 2 + quantisation ** 2),
        "right_encoder": np.sqrt(val_df["right_noise"].mean() ** 2 + quantisation ** 2),
        # the gyro has no ticks, but each run's bias is unknowable in advance
        "gyro": np.sqrt(val_df["gyro_noise"].mean() ** 2 + bias_spread ** 2),
    }

    print("\nHow far off is each prediction, on held-out runs")
    print("  compared against what the sensor actually read:\n")
    print("  %-15s %10s %10s" % ("", "off by", "best possible"))
    for i, name in enumerate(OUTPUTS):
        error = np.sqrt(np.mean((predictions[:, i] - val_df[name].values) ** 2))
        print("  %-15s %10.4f %10.4f" % (name, error, floors[name]))

    print("\nSame predictions, compared against the noiseless truth")
    truth_columns = ["left_spin", "right_spin", "turn_rate"]
    for name, truth in zip(OUTPUTS, truth_columns):
        error = np.sqrt(np.mean((predictions[:, OUTPUTS.index(name)]
                                 - val_df[truth].values) ** 2))
        print("  %-15s off by %.4f" % (name, error))
    print("  This is the number that answers the question. The plain")
    print("  network's version of it is what the extra depth has to beat.")

    torch.save({"weights": model.state_dict(),
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "inputs": INPUTS, "outputs": OUTPUTS},
               str(Path(__file__).parent / "resnet_model.pt"))
    print("\nSaved resnet_model.pt (weights and scaling)")


if __name__ == "__main__":
    main()

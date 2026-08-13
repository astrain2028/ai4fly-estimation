"""
Gaussian process measurement model.

A GP does not learn weights. It keeps a set of training points and answers
new questions by asking "which points have I seen that look like this one?"
Nearby points get a lot of say, distant ones get almost none.

That gives something the other models have to be built to provide: if a new
input sits far from everything in the training set, no stored point speaks
for it, and the GP says so by reporting a large spread. Uncertainty about
its own knowledge comes free.

WHAT IT DOES NOT GIVE

A standard GP has one noise number, learned once and applied everywhere. It
cannot say "readings are noisier when the wheels spin fast" -- that is
exactly the thing the heteroscedastic model exists to do. So the GP covers
one half of what this project wants and not the other.

KEEPING IT SMALL

Exact GP maths needs an n-by-n matrix, and n here is 80,000. That matrix
would have 6.4 billion entries. So we keep a random handful of training
points instead -- a few hundred -- and do the exact maths on those. Crude,
but it is real GP regression, and it is enough to see how the method
behaves.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_data.csv"

import numpy as np
import pandas as pd

INPUTS = ["x", "y", "heading", "speed", "turn_rate"]
OUTPUTS = ["left_encoder", "right_encoder", "gyro"]
TRUE_NOISE = ["left_noise", "right_noise", "gyro_noise"]

N_KEPT = 1500         # how many training points to hold on to


def rbf_kernel(A, B, lengthscale):
    """How similar is every point in A to every point in B?

    1.0 when two points are identical, falling towards 0 as they separate.
    `lengthscale` sets how far apart counts as far.
    """
    # squared distance between every pair, without a Python loop
    sq = (np.sum(A ** 2, axis=1)[:, None]
          + np.sum(B ** 2, axis=1)[None, :]
          - 2 * A @ B.T)
    sq = np.maximum(sq, 0.0)          # tiny negatives from rounding
    return np.exp(-0.5 * sq / lengthscale ** 2)


class SimpleGP:
    """One GP per output. Fit once, then answer questions."""

    def __init__(self, lengthscale=1.0, noise=0.1):
        self.lengthscale = lengthscale
        self.noise = noise

    def fit(self, X, y):
        self.X = X
        K = rbf_kernel(X, X, self.lengthscale) + self.noise ** 2 * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        # alpha holds "how much each stored point pulls the answer"
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
        return self

    def log_marginal_likelihood(self, y):
        """How well these settings explain the training data.

        Used to choose the lengthscale and noise, instead of guessing.
        """
        fit_term = -0.5 * y @ self.alpha
        complexity = -np.sum(np.log(np.diag(self.L)))
        return fit_term + complexity - 0.5 * len(y) * np.log(2 * np.pi)

    def predict(self, X_new):
        """Returns the expected value and the spread around it."""
        k = rbf_kernel(X_new, self.X, self.lengthscale)
        mean = k @ self.alpha

        # how much of the answer is pinned down by stored points
        v = np.linalg.solve(self.L, k.T)
        explained = np.sum(v ** 2, axis=0)

        # 1.0 is the spread with no information at all. Subtracting what the
        # stored points explain leaves the leftover doubt -- big far from
        # the data, small near it. This is the epistemic part.
        epistemic = np.maximum(1.0 - explained, 0.0)

        # the noise floor, the same number everywhere
        total = epistemic + self.noise ** 2
        return mean, total, epistemic


def choose_settings(X, y, lengthscales, noises):
    """Pick the lengthscale and noise that best explain the data."""
    best = (-np.inf, None, None)
    for ls in lengthscales:
        for nz in noises:
            gp = SimpleGP(ls, nz).fit(X, y)
            score = gp.log_marginal_likelihood(y)
            if score > best[0]:
                best = (score, ls, nz)
    return best[1], best[2], best[0]


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(DATA)

    runs = df["run"].unique()
    rng.shuffle(runs)
    val_runs = runs[:int(len(runs) * 0.2)]
    is_val = df["run"].isin(val_runs)
    train_df, val_df = df[~is_val], df[is_val]
    print("train: %d rows from %d runs" % (len(train_df), train_df["run"].nunique()))
    print("val:   %d rows from %d runs" % (len(val_df), val_df["run"].nunique()))

    X_all = train_df[INPUTS].values
    x_mean, x_std = X_all.mean(0), X_all.std(0)

    keep = rng.choice(len(train_df), N_KEPT, replace=False)
    X = (X_all[keep] - x_mean) / x_std
    X_val = (val_df[INPUTS].values - x_mean) / x_std
    print("\nkeeping %d of %d training points" % (N_KEPT, len(train_df)))
    print("(exact GP maths on all of them would need a %d x %d matrix)"
          % (len(train_df), len(train_df)))

    print("\n%-15s %11s %8s %10s %10s"
          % ("", "lengthscale", "noise", "mean err", "predicted"))
    for i, (name, truth_col) in enumerate(zip(OUTPUTS, TRUE_NOISE)):
        y_all = train_df[name].values
        y_mean, y_std = y_all.mean(), y_all.std()
        y = (y_all[keep] - y_mean) / y_std

        # The grid must be wide enough that the best value is inside it,
        # not at an edge. A first pass used a narrower range, the search
        # picked the largest value offered, and the right encoder ended up
        # reporting more than twice its real noise.
        ls, nz, _ = choose_settings(X, y, [1, 2, 4, 8, 16, 32],
                                    [0.003, 0.01, 0.03, 0.05, 0.1])
        gp = SimpleGP(ls, nz).fit(X, y)
        mean_s, total_s, epi_s = gp.predict(X_val)

        pred_mean = mean_s * y_std + y_mean
        pred_std = np.sqrt(total_s) * y_std

        err = np.sqrt(np.mean((pred_mean - val_df[name].values) ** 2))
        print("%-15s %11.2f %8.3f %10.4f %10.4f"
              % (name, ls, nz, err, pred_std.mean()))

        if i == 0:                      # look closer at one channel
            left_epi, left_std, left_ls, left_nz = epi_s, pred_std, ls, nz

    print("\nCheck: does the doubt grow away from the stored points?")
    order = np.argsort(left_epi)
    for frac, label in [(0.05, "closest to stored data"),
                        (0.50, "typical"),
                        (0.95, "furthest away")]:
        idx = order[int(frac * len(order))]
        print("   %-24s leftover doubt %.5f" % (label, left_epi[idx]))
    print("   That growth is the part a plain network has no way to produce.")

    print("\nCheck: does the noise floor follow the state?")
    spin = np.abs(val_df["left_spin"].values)
    slow = spin < np.percentile(spin, 25)
    fast = spin > np.percentile(spin, 75)
    tick_var = ((2 * np.pi / 1024) / 0.02) ** 2 / 12
    actual_slow = np.sqrt(val_df["left_noise"].values[slow] ** 2 + tick_var).mean()
    actual_fast = np.sqrt(val_df["left_noise"].values[fast] ** 2 + tick_var).mean()
    print("   slow wheels: GP says %.4f   actually %.4f"
          % (left_std[slow].mean(), actual_slow))
    print("   fast wheels: GP says %.4f   actually %.4f"
          % (left_std[fast].mean(), actual_fast))
    print("   The GP barely moves, because its noise is one fixed number.")
    print("   It knows when it is out of its depth, not when the sensor is noisy.")


if __name__ == "__main__":
    main()

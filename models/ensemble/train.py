"""
Five heteroscedastic models, trained separately, answering together.

WHY FIVE OF THE SAME THING

One heteroscedastic model says how noisy a reading should be. What it cannot
say is whether it has any business answering at all -- shown a state unlike
anything in its training data, it still returns a confident number.

Train the same model five times from different random starting weights and
they agree closely wherever the data pinned them down, and disagree where it
did not. That disagreement is the missing signal.

    total spread = average of what they each claim
                 + how much they disagree with each other
                   \___ sensor noise ___/  \___ model doubt ___/

The first part is aleatoric: real noise in the sensor, irreducible. The
second is epistemic: the model's own ignorance, which more data would fix.

WHAT IT COSTS

Five networks means five times the inference, five times the memory, and
five times the training. On a companion computer sharing cycles with a
flight stack, that is the whole problem. This arm is here as the honest
upper bound -- the best epistemic signal available -- so that a cheaper
method has something real to be measured against.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
import torch

MEMBERS = 5


def _bhr():
    """Load the heteroscedastic trainer by path.

    Every member of this ensemble IS the bhr arm, trained again from a
    different start. Importing it rather than copying it means the two arms
    cannot drift apart -- if the loss changes there, it changes here.
    """
    spec = importlib.util.spec_from_file_location(
        "bhr_train_for_ensemble", ROOT / "models" / "bhr" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _bhr()


def main():
    # The split is fixed, so every member sees the same data. Only the
    # starting weights and the batch order differ. That is what makes their
    # disagreement mean something: it is disagreement about the same
    # evidence, not about different evidence.
    train_df, val_df = bhr.load_and_split()

    x_train = torch.tensor(train_df[bhr.INPUTS].values, dtype=torch.float32)
    y_train = torch.tensor(train_df[bhr.OUTPUTS].values, dtype=torch.float32)
    x_val = torch.tensor(val_df[bhr.INPUTS].values, dtype=torch.float32)
    y_val = torch.tensor(val_df[bhr.OUTPUTS].values, dtype=torch.float32)

    x_mean, x_std = x_train.mean(0), x_train.std(0)
    y_mean, y_std = y_train.mean(0), y_train.std(0)
    xs_train = (x_train - x_mean) / x_std
    xs_val = (x_val - x_mean) / x_std
    ys_train = (y_train - y_mean) / y_std
    ys_val = (y_val - y_mean) / y_std

    weights = []
    for member in range(MEMBERS):
        print("\n--- member %d of %d ---" % (member + 1, MEMBERS))
        torch.manual_seed(member)          # the only thing that differs
        model = bhr.make_model()
        bhr.train(model, xs_train, ys_train, xs_val, ys_val)
        weights.append(model.state_dict())

    # What each member predicts on the held-out runs
    means, variances = [], []
    for state in weights:
        model = bhr.make_model()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            eta1, eta2 = bhr.split_outputs(model(xs_val), False)
            m, v = bhr.to_mean_and_var(eta1, eta2)
        means.append((m * y_std + y_mean).numpy())
        variances.append((v * y_std ** 2).numpy())

    means = np.array(means)              # (members, rows, sensors)
    variances = np.array(variances)

    aleatoric = variances.mean(axis=0)   # what they each claim
    epistemic = means.var(axis=0)        # how much they disagree

    print("\nWhere does the uncertainty come from, on held-out runs?")
    print("  %-15s %12s %12s %10s" % ("", "sensor noise", "model doubt", "doubt %"))
    for i, name in enumerate(bhr.OUTPUTS):
        a, e = aleatoric[:, i].mean(), epistemic[:, i].mean()
        print("  %-15s %12.6f %12.6f %9.2f%%"
              % (name, a, e, 100 * e / (a + e)))

    print("\n  On data like the training set the doubt should be small: the")
    print("  members were shown these conditions and agree about them. The")
    print("  number to watch is what happens when they are not.")

    print("\nMean accuracy of the ensemble (compare with a single bhr model)")
    combined = means.mean(axis=0)
    for i, name in enumerate(bhr.OUTPUTS):
        err = np.sqrt(np.mean((combined[:, i] - val_df[name].values) ** 2))
        print("   %-15s off by %.4f" % (name, err))

    torch.save({"weights": weights, "members": MEMBERS,
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std,
                "inputs": bhr.INPUTS, "outputs": bhr.OUTPUTS},
               str(Path(__file__).parent / "ensemble_model.pt"))
    print("\nSaved ensemble_model.pt (%d members)" % MEMBERS)


if __name__ == "__main__":
    main()

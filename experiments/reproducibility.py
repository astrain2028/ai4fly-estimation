"""
Does retraining from source reproduce the published model?

Motivation
----------

No trained weights are tracked in this repository. The generators and the
training scripts are, on the reasoning that those are the artefacts worth
keeping. That makes reproducibility load-bearing rather than a courtesy:
every table in the README rests on a retrain from a fixed seed producing the
same model, and until this file existed nobody had checked that it does.

Two questions, and they are different
-------------------------------------

Determinism asks whether training twice from the same seed gives the same
weights. It can fail through an unseeded random draw, a data loader with its
own generator, or a nondeterministic kernel. It is cheap to test, because a
few epochs expose any divergence.

Fidelity asks whether a fresh retrain matches the model actually saved on
disk. Determinism is necessary for it but not sufficient: the saved model may
have been trained on a dataset or with settings that have since changed, and
then the published numbers describe a model the current code no longer
produces. This one needs the full training schedule and takes minutes.

Scope
-----

The health-conditioned model is tested, because combined, layered and doubt
all load its weights; if it does not reproduce, none of them do. The saved
model is never overwritten -- training happens in memory, and the file is only
read.

    python experiments/reproducibility.py          determinism only
    python experiments/reproducibility.py --full   and fidelity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loader import load_module

health = load_module(ROOT / "models" / "health" / "train.py", "health_train_repro")
SAVED = ROOT / "models" / "health" / "health_model.pt"

SHORT_EPOCHS = 3


def tensors(frame, columns):
    return torch.tensor(frame[columns].values, dtype=torch.float32)


def train_once(data, epochs):
    """One training run from seed zero, in memory. Returns the state dict."""
    x, y, xv, yv, x_mean, x_std, y_mean, y_std = data
    torch.manual_seed(0)
    model = health.make_model()

    saved_epochs = health.bhr.EPOCHS
    health.bhr.EPOCHS = epochs
    try:
        health.bhr.train(model, (x - x_mean) / x_std, (y - y_mean) / y_std,
                         (xv - x_mean) / x_std, (yv - y_mean) / y_std)
    finally:
        health.bhr.EPOCHS = saved_epochs
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def largest_difference(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def main():
    full = "--full" in sys.argv

    train_df, val_df = health.load_and_split()
    x, y = tensors(train_df, health.INPUTS), tensors(train_df, health.OUTPUTS)
    xv, yv = tensors(val_df, health.INPUTS), tensors(val_df, health.OUTPUTS)
    data = (x, y, xv, yv, x.mean(0), x.std(0), y.mean(0), y.std(0))

    print("\nDETERMINISM\n")
    print("Two runs of %d epochs from seed zero." % SHORT_EPOCHS)
    first = train_once(data, SHORT_EPOCHS)
    second = train_once(data, SHORT_EPOCHS)
    gap = largest_difference(first, second)
    print("  largest difference in any weight: %.3e" % gap)
    print("  %s" % ("identical" if gap == 0.0 else
                    "NOT identical -- training draws on an unseeded source"))

    if not full:
        print("\n(pass --full to also check the saved model; it retrains at")
        print(" the full schedule and takes several minutes)")
        return 0 if gap == 0.0 else 1

    print("\n\nFIDELITY\n")
    if not SAVED.exists():
        print("  no %s to compare against" % SAVED.name)
        return 1

    print("A full retrain of %d epochs, against %s."
          % (health.EPOCHS, SAVED.relative_to(ROOT)))
    fresh = train_once(data, health.EPOCHS)
    saved = torch.load(SAVED, weights_only=False)["weights"]
    gap = largest_difference(fresh, saved)
    print("  largest difference in any weight: %.3e" % gap)

    # Weight identity is the strict test. The practical one is whether the
    # two models predict the same readings, since that is all the filter sees.
    probe = (xv[:2000] - data[4]) / data[5]
    outputs = []
    for weights in (fresh, saved):
        model = health.make_model()
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            outputs.append(model(probe).numpy())
    print("  largest difference in any output:  %.3e"
          % np.abs(outputs[0] - outputs[1]).max())

    if gap == 0.0:
        print("\n  The saved model is exactly what this code produces. Every")
        print("  table built on it can be regenerated from the repository.")
    else:
        print("\n  The saved model is NOT what this code produces. Either the")
        print("  dataset or the training settings changed after it was saved,")
        print("  and the published numbers describe a model the current code")
        print("  does not reproduce. Retrain before quoting them.")
    return 0 if gap == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())

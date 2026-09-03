"""
A last-layer posterior for the health-conditioned model.

WHY THIS IS A DIFFERENT PROPOSITION FROM THE SAME THING ON bhr

models/bhr/laplace.py answers "have I seen a state like this before". Its
model takes five vehicle states, so its doubt grows when the robot goes
somewhere unfamiliar -- faster than training, turning harder. Useful, and
measured: the doubt climbs sharply past the training envelope while staying
flat inside it.

That signal is structurally blind to a broken sensor, and the reason is worth
being precise about. A fault arrives in the MEASUREMENT. The state is
entirely ordinary -- the robot is driving at a speed it has driven a thousand
times -- so nothing about the input is unfamiliar and no novelty score over
inputs can see it. This is the same arithmetic as the covariance argument for
why health has to be an input at all, one level up.

The health model is the one arm where that stops being true, because its
inputs are not only the vehicle state:

    [x, y, heading, speed, turn_rate,
     bias_left, bias_right, bias_gyro, noise_left, noise_right, noise_gyro]

The last six are estimated by the filter and fed back in. Training covers
severities from 0.25 to 3.0 and nothing constrains the filter to stay there.
Faced with a fault it cannot represent -- a frozen encoder, say -- the update
will move health wherever best explains the residual, and that may be a long
way outside anything the model was fitted on.

So the question this file makes askable is not "is this fault familiar",
which the model cannot answer, but "is this HEALTH LEVEL familiar", which it
can. A filter driving health to 6.0 is asking a question the model has no
basis for, and that is detectable by exactly the machinery that cannot see
the fault directly.

Whether it actually happens is an empirical matter and the self-test below
measures it. Both answers are worth having. If health leaves its training
range on unseen faults, this is a fault-class detector for the price of a
matrix multiply. If it does not -- if the model finds a comfortable
in-distribution explanation for a frozen sensor -- that is a sharper negative
result than "it does badly", and it argues for a trivial frozen-reading test
instead of anything learned.

WHAT IS REUSED

Everything general lives in models/bhr/laplace.py: the feature extraction,
the Gram matrix, the evidence-based choice of prior precision, and the
curvature identity that makes this cheap in the first place -- the second
derivative of the loss with respect to eta1 is just the predicted variance.
Only the model and the dataset differ, so only those are supplied here.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_faulted.csv"

import numpy as np
import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load(ROOT / "models" / "bhr" / "laplace.py", "bhr_laplace_for_doubt")
health = _load(ROOT / "models" / "health" / "train.py", "health_train_for_doubt")

INPUTS, OUTPUTS = health.INPUTS, health.OUTPUTS
TAU_GRID = base.TAU_GRID


def fit(model_path=None, data_path=DATA):
    """Fit the last-layer posterior for the trained health model.

    Fitted on the training runs only. Using the validation runs would make
    the model look confident about inputs it was never fitted on, which is
    the exact failure this is built to detect.
    """
    if model_path is None:
        model_path = ROOT / "models" / "health" / "health_model.pt"
    saved = torch.load(model_path, weights_only=False)

    model = health.make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_std = saved["y_std"]

    train_df, val_df = health.load_and_split(data_path)

    x = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    xs = (x - x_mean) / x_std
    phi = base.features(model, xs)

    with torch.no_grad():
        eta1, eta2 = health.bhr.split_outputs(model(xs), False)
        _, var = health.bhr.to_mean_and_var(eta1, eta2)
    var = var.numpy().astype(np.float64)

    last = model[-1]
    W = last.weight.detach().numpy().astype(np.float64)
    b = last.bias.detach().numpy().astype(np.float64)

    n_features = phi.shape[1]
    print("features %d, rows %d" % (n_features, len(phi)))
    print("\nChoosing the prior precision by evidence\n")
    print("  %-12s" % "tau" + "".join("%14s" % n for n in OUTPUTS))

    scores = {name: [] for name in OUTPUTS}
    for tau in TAU_GRID:
        row = []
        for j, name in enumerate(OUTPUTS):
            weights = np.concatenate([W[j], [b[j]]])
            _, log_det = base.fit_posterior(phi, var[:, j], tau)
            score = base.log_evidence(0.0, weights, tau, log_det, n_features)
            scores[name].append(score)
            row.append(score)
        print("  %-12.1f" % tau + "".join("%14.1f" % s for s in row))

    best_tau, posteriors = [], []
    for j, name in enumerate(OUTPUTS):
        tau = TAU_GRID[int(np.argmax(scores[name]))]
        best_tau.append(tau)
        A_inv, _ = base.fit_posterior(phi, var[:, j], tau)
        posteriors.append(A_inv)

    print("\n  chosen: " + ", ".join("%s tau=%g" % (n, t)
                                     for n, t in zip(OUTPUTS, best_tau)))
    if any(t in (TAU_GRID[0], TAU_GRID[-1]) for t in best_tau):
        print("  WARNING: a chosen tau sits at the edge of the grid, so the")
        print("  real best may be outside it. Widen TAU_GRID and refit.")

    # What the doubt looks like on data the model was fitted on. The arm needs
    # this to say whether a doubt it sees at run time is large or ordinary,
    # and measuring it here means it is never a guessed constant.
    reference = base.epistemic_variance(model, posteriors, xs).mean(axis=0)
    print("\n  typical in-distribution doubt: "
          + ", ".join("%s %.3e" % (n, r) for n, r in zip(OUTPUTS, reference)))

    np.savez(Path(__file__).parent / "doubt_laplace.npz",
             posteriors=np.array(posteriors),
             tau=np.array(best_tau),
             reference=reference,
             n_features=n_features)
    print("\nSaved doubt_laplace.npz")
    return model, posteriors, reference, (x_mean, x_std, y_std), val_df


if __name__ == "__main__":
    model, posteriors, reference, scaling, val_df = fit()
    x_mean, x_std, y_std = scaling

    print("\n\nDOES THE DOUBT GROW WHERE THE FILTER MIGHT ACTUALLY GO?\n")
    print("The five vehicle states are held at an ordinary cruise and only")
    print("the health input is moved. Training covers 0.25 to 3.0, so this")
    print("is the question the filter asks when it cannot explain a fault.\n")
    print("  %-22s %18s %12s" % ("bias_left fed in", "doubt (left enc)",
                                 "vs typical"))

    for level in [0.0, 1.0, 3.0, 5.0, 8.0, 15.0]:
        probe = np.zeros((200, len(INPUTS)))
        probe[:, 3] = 0.8                      # an unremarkable speed
        probe[:, 4] = np.linspace(-0.5, 0.5, 200)
        probe[:, 5] = level                    # severity_bias_left
        p = torch.tensor(probe, dtype=torch.float32)
        doubt = base.epistemic_variance(
            model, posteriors, (p - x_mean) / x_std)[:, 0].mean()
        print("  %-22.1f %18.6e %11.1fx"
              % (level, doubt, doubt / reference[0]))

    print("\n  If this climbs steeply past 3.0 the arm has a signal for")
    print("  'the filter is asking me something I was never taught'. If it")
    print("  stays flat, the model extrapolates confidently into health")
    print("  levels it has never seen, and no epistemic term will help.")

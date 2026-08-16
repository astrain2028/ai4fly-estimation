"""
The Bayesian half of the heteroscedastic model.

WHAT IS MISSING WITHOUT IT

The trained model gives one number for the noise on each sensor. That number
answers "how noisy is this reading", and it is honest wherever the training
data pinned it down. It has nothing at all to say about "have I ever seen a
state like this before". Shown something unfamiliar, the model returns a
confident answer with no hint that it is guessing.

The ensemble arm gets that missing signal by training five models and
watching them disagree, which costs five times the inference. This file gets
it from one model, after training, for about the price of a small matrix
multiply.

THE IDEA

Training finds one set of weights: the ones that best explain the data. But
other weights nearby explain it almost as well, and the model has no
principled reason to prefer its own. That spread of nearly-as-good weights IS
the model's uncertainty about itself.

Laplace's approximation says: pretend that spread is a Gaussian, centred on
the weights we found, with a width set by how sharply the loss curves upward
around them. Sharp curvature means the data pinned those weights down, so the
spread is narrow. Flat curvature means many weights would have done, so it is
wide.

    p(w | data)  ~=  Normal( w_trained , A^-1 )

where A is the curvature of the loss. Then a prediction is no longer one
number but a distribution, and its spread is the epistemic part.

ONLY THE LAST LAYER

Curvature for every weight in the network would be a matrix of 5,000 by 5,000
and is not worth it. Treating only the final layer as uncertain keeps almost
all of the signal: the layers before it are a feature extractor, and the last
layer is the part that actually commits to an answer. That makes A a 65 by 65
matrix -- 64 hidden units plus a bias -- which inverts instantly.

THE PART THAT IS NICE ABOUT NATURAL PARAMETERS

For the loss this model trains on, the curvature with respect to eta1 works
out to something very simple. Writing the negative log likelihood in natural
parameters,

    -log p = -( eta1*y + eta2*y^2 + eta1^2/(4*eta2) - 0.5*log(pi/-eta2) )

and differentiating twice with respect to eta1:

    d^2(-log p) / d(eta1)^2  =  -1 / (2*eta2)  =  variance

The curvature is just the predicted variance. So a row the model thinks is
noisy contributes little curvature -- it pins the weights down weakly, which
is exactly right, because a noisy observation is weak evidence. That falls
straight out of the parameterisation rather than having to be argued for.

PUTTING IT TOGETHER

For each output j, with features f(x) from the penultimate layer:

    A_j     = sum over rows of  variance_n * f_n f_n^T  +  tau * I
    epi_j(x) = f(x)^T A_j^-1 f(x)          <- spread of eta1
    Var[mu]  = v^2 * epi_j(x)              <- carried through to the mean

and the covariance handed to the filter becomes

    R = aleatoric + epistemic
      = (v + v^2 * epi) * y_std^2

tau is the prior precision -- how much to distrust large weights before
seeing any data. It is chosen here by maximising the evidence rather than
guessed, because guessing it is the usual way this method goes wrong.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))
DATA = ROOT / "data" / "robot_data.csv"

import numpy as np
import pandas as pd
import torch

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "bhr_train_for_laplace", Path(__file__).parent / "train.py")
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)

INPUTS, OUTPUTS = _train.INPUTS, _train.OUTPUTS

# How strongly to distrust large weights before seeing data. Searched over
# rather than picked -- the evidence below says which one the data prefers.
TAU_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def features(model, x):
    """Everything the network computes before the final layer, plus a 1.

    The trailing 1 lets the final layer's bias be treated as just another
    weight, so it gets a posterior like the rest instead of being assumed
    known exactly.
    """
    with torch.no_grad():
        hidden = model[:-1](x)
    ones = torch.ones(len(hidden), 1)
    return torch.cat([hidden, ones], dim=1).numpy().astype(np.float64)


def fit_posterior(phi, curvature, tau):
    """A^-1 for one output, and the log determinant that the evidence needs.

    A = sum_n curvature_n * phi_n phi_n^T + tau * I

    The sum is a weighted Gram matrix, so it is one matrix multiply over all
    80,000 rows rather than a loop.
    """
    weighted = phi * curvature[:, None]
    A = weighted.T @ phi
    A[np.diag_indices_from(A)] += tau

    L = np.linalg.cholesky(A)
    A_inv = np.linalg.inv(A)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return A_inv, log_det


def log_evidence(fit_term, weights, tau, log_det, n_features):
    """How well this prior precision explains the data.

        log Z ~= log p(data | w) - (tau/2)|w|^2 + (d/2) log tau - 0.5 log|A|

    The last two terms are the trade-off that makes this a real choice rather
    than "smaller tau always wins": a loose prior explains the data better but
    pays for the volume of weight space it had to search.
    """
    return (fit_term
            - 0.5 * tau * float(weights @ weights)
            + 0.5 * n_features * np.log(tau)
            - 0.5 * log_det)


def fit(model_path=None, data_path=DATA):
    """Fit the last-layer posterior for a trained heteroscedastic model."""
    if model_path is None:
        model_path = Path(__file__).parent / "bhr_model.pt"
    saved = torch.load(model_path, weights_only=False)

    model = _train.make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    # The posterior must be fitted on the training runs only. Using the
    # validation runs would make the model look confident about states it was
    # never actually fitted on, which is the exact failure this is meant to
    # detect.
    train_df, val_df = _train.load_and_split(data_path)

    x = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    xs = (x - x_mean) / x_std
    phi = features(model, xs)

    with torch.no_grad():
        eta1, eta2 = _train.split_outputs(model(xs), False)
        _, var = _train.to_mean_and_var(eta1, eta2)
    var = var.numpy().astype(np.float64)

    last = model[-1]
    W = last.weight.detach().numpy().astype(np.float64)
    b = last.bias.detach().numpy().astype(np.float64)

    print("features %d (64 hidden + 1 bias), rows %d" % (phi.shape[1], len(phi)))
    print("\nChoosing the prior precision by evidence\n")
    print("  %-12s" % "tau" + "".join("%14s" % n for n in OUTPUTS))

    n_features = phi.shape[1]
    scores = {name: [] for name in OUTPUTS}
    for tau in TAU_GRID:
        row = []
        for j, name in enumerate(OUTPUTS):
            weights = np.concatenate([W[j], [b[j]]])
            _, log_det = fit_posterior(phi, var[:, j], tau)
            # The fit term is the same for every tau (the weights do not
            # move), so it cancels in the comparison and is set to zero.
            score = log_evidence(0.0, weights, tau, log_det, n_features)
            scores[name].append(score)
            row.append(score)
        print("  %-12.1f" % tau + "".join("%14.1f" % s for s in row))

    best_tau, posteriors = [], []
    for j, name in enumerate(OUTPUTS):
        tau = TAU_GRID[int(np.argmax(scores[name]))]
        best_tau.append(tau)
        A_inv, _ = fit_posterior(phi, var[:, j], tau)
        posteriors.append(A_inv)

    print("\n  chosen: " + ", ".join("%s tau=%g" % (n, t)
                                     for n, t in zip(OUTPUTS, best_tau)))
    if any(t in (TAU_GRID[0], TAU_GRID[-1]) for t in best_tau):
        print("  WARNING: a chosen tau sits at the edge of the grid, so the")
        print("  real best may be outside it. Widen TAU_GRID and refit.")

    np.savez(Path(__file__).parent / "bhr_laplace.npz",
             posteriors=np.array(posteriors),
             tau=np.array(best_tau),
             n_features=n_features)
    print("\nSaved bhr_laplace.npz")

    return model, posteriors, (x_mean, x_std, y_mean, y_std), val_df


def epistemic_variance(model, posteriors, states_scaled):
    """f(x)^T A^-1 f(x) for each output -- the spread on eta1."""
    phi = features(model, states_scaled)
    out = np.zeros((len(phi), len(posteriors)))
    for j, A_inv in enumerate(posteriors):
        # the diagonal of phi A^-1 phi^T, without forming the full matrix
        out[:, j] = np.einsum("ni,ij,nj->n", phi, A_inv, phi)
    return out


if __name__ == "__main__":
    import time

    model, posteriors, scaling, val_df = fit()
    x_mean, x_std, y_mean, y_std = scaling

    print("\n\nDoes the doubt stay small on data like the training set?\n")
    x_val = torch.tensor(val_df[INPUTS].values, dtype=torch.float32)
    xs_val = (x_val - x_mean) / x_std

    with torch.no_grad():
        eta1, eta2 = _train.split_outputs(model(xs_val), False)
        _, var_s = _train.to_mean_and_var(eta1, eta2)
    var_s = var_s.numpy().astype(np.float64)
    epi_eta1 = epistemic_variance(model, posteriors, xs_val)
    epi_mean = var_s ** 2 * epi_eta1          # carried through to the mean

    ys = y_std.numpy().astype(np.float64)
    print("  %-15s %14s %14s %10s" % ("", "sensor noise", "model doubt", "doubt %"))
    for j, name in enumerate(OUTPUTS):
        a = (var_s[:, j] * ys[j] ** 2).mean()
        e = (epi_mean[:, j] * ys[j] ** 2).mean()
        print("  %-15s %14.8f %14.8f %9.3f%%"
              % (name, a, e, 100 * e / (a + e)))

    print("\nDoes it grow where the model has not been?\n")
    # Speeds far past anything in training. The states are otherwise
    # ordinary, so anything that moves is coming from the input being
    # unfamiliar rather than from the sensors being noisy.
    print("  %-22s %16s %12s" % ("speed fed in", "doubt (left enc)", "vs normal"))
    base = None
    for speed in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        probe = np.zeros((200, 5))
        probe[:, 3] = speed
        probe[:, 4] = np.linspace(-1, 1, 200)
        p = torch.tensor(probe, dtype=torch.float32)
        e = epistemic_variance(model, posteriors, (p - x_mean) / x_std)[:, 0].mean()
        if base is None:
            base = e
        print("  %-22.1f %16.8f %11.1fx" % (speed, e, e / base))

    print("\n  Training speeds run to about 1.5 m/s. If the doubt climbs well")
    print("  past that and stays flat inside it, the model knows where it has")
    print("  been -- which is the entire point of this file.")

    print("\nWhat it costs\n")
    probe = torch.tensor(np.zeros((11, 5)), dtype=torch.float32)   # sigma points
    t0 = time.perf_counter()
    for _ in range(200):
        epistemic_variance(model, posteriors, probe)
    per_call = 1000 * (time.perf_counter() - t0) / 200
    print("  %.4f ms per filter step, on top of the forward pass." % per_call)
    print("  The ensemble buys the same signal with four extra networks.")

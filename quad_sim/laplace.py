"""
A last-layer posterior for the quadcopter's health-conditioned model.

The same method as models/doubt/laplace.py, which fits one for the ground
robot's health model: treat the final layer's weights as uncertain, take the
curvature of the loss around the trained ones, and read the model's doubt
about a prediction off the spread that implies. Everything general --
features, the evidence, the curvature identity that makes the curvature on
eta1 equal the predicted variance -- is in models/bhr/laplace.py and is used
from there.

Two things differ from the robot, and both are about size. The network is
twice as wide, 128 hidden units rather than 64, so each posterior is 129 by
129, from 325,000 training rows. The Gram matrices are accumulated in chunks
rather than formed from one feature matrix that would need a third of a
gigabyte, and each is computed once and reused across the prior-precision
grid rather than rebuilt for every tau.

Runs, not rows
--------------

Laplace counts every row as independent evidence, and these are not: 1,000
rows per flight at 50 Hz, sharing a per-flight gyro bias the model is not
told about. Counted that way the posterior is far too tight -- an independent
check found the run-level spread 10 to 19 times larger on the accelerometer
and magnetometer and 86 to 134 times on the gyro. So the file carries both:
``posteriors``, the ordinary Laplace one, and ``posteriors_run``, the
cluster-robust sandwich from models/bhr/laplace.py run_posterior, which is
what layered uses.

What the inputs are, and so what this can see
---------------------------------------------

Six attitude states and six health levels, all estimated by the filter.
Training covers health from 0 to 3.0. quad_sim/health_readout.py found the
magnetometer entries sitting at 3 to 4.8 on healthy flights, past that range,
so on this vehicle the epistemic term has something to respond to even
without a fault -- whether that helps or hurts is what
experiments/epistemic.py measures.

    python quad_sim/laplace.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from loader import load_module

train = load_module(HERE / "train.py", "quad_train_for_laplace")
base = load_module(ROOT / "models" / "bhr" / "laplace.py", "bhr_laplace_for_quad")

INPUTS, OUTPUTS = train.INPUTS, train.OUTPUTS
TAU_GRID = base.TAU_GRID
CHUNK = 50_000
OUT = HERE / "quad_laplace.npz"


def load_model(path=None):
    """The trained quad model and its input and output scaling."""
    path = HERE / "quad_health.pt" if path is None else path
    saved = torch.load(path, weights_only=False)
    model = train.make_model()
    model.load_state_dict(saved["weights"])
    model.eval()
    return model, saved


def predicted_variance(model, xs):
    """The model's own variance per output, in scaled units."""
    train.bhr.OUTPUTS = OUTPUTS          # bhr reads the channel count from here
    with torch.no_grad():
        eta1, eta2 = train.bhr.split_outputs(model(xs), False)
        _, var = train.bhr.to_mean_and_var(eta1, eta2)
    return var.numpy().astype(np.float64)


def grams(model, xs, ys, run_index):
    """Per output: the Gram matrix sum var * phi phi', and per-run score sums.

    ``ys`` are the targets in the model's scaled units and ``run_index`` the
    flight each row came from, numbered from zero; the per-run sums of the
    score (mean - y) * phi are what the run-level posterior needs.
    """
    n_features = model[-1].in_features + 1
    n_runs = int(run_index.max()) + 1
    G = np.zeros((len(OUTPUTS), n_features, n_features))
    S = np.zeros((len(OUTPUTS), n_runs, n_features))
    train.bhr.OUTPUTS = OUTPUTS
    for start in range(0, len(xs), CHUNK):
        chunk = xs[start:start + CHUNK]
        phi = base.features(model, chunk)
        with torch.no_grad():
            eta1, eta2 = train.bhr.split_outputs(model(chunk), False)
            mean, var = train.bhr.to_mean_and_var(eta1, eta2)
        var = var.numpy().astype(np.float64)
        residual = mean.numpy().astype(np.float64) - ys[start:start + CHUNK]
        runs = run_index[start:start + CHUNK]
        for j in range(len(OUTPUTS)):
            G[j] += (phi * var[:, j][:, None]).T @ phi
            S[j] += base.run_sums(phi, residual[:, j], runs, n_runs)
    return G, S


def mean_leverage(model, xs, posteriors):
    """phi' C_j phi averaged over rows, per output: the in-distribution doubt."""
    total = np.zeros(len(posteriors))
    for start in range(0, len(xs), CHUNK):
        phi = base.features(model, xs[start:start + CHUNK])
        total += np.array([((phi @ C) * phi).sum(axis=1).sum()
                           for C in posteriors])
    return total / len(xs)


def fit(model_path=None):
    """Choose tau per output by evidence and write the posteriors."""
    model, saved = load_model(model_path)
    train_df, _ = train.load_and_split()
    x = torch.tensor(train_df[INPUTS].values, dtype=torch.float32)
    xs = (x - saved["x_mean"]) / saved["x_std"]
    ys = ((train_df[OUTPUTS].values - saved["y_mean"].numpy())
          / saved["y_std"].numpy()).astype(np.float64)
    run_index = np.unique(train_df["run"].values, return_inverse=True)[1]

    G, S = grams(model, xs, ys, run_index)
    n_features = G.shape[1]
    W = model[-1].weight.detach().numpy().astype(np.float64)
    b = model[-1].bias.detach().numpy().astype(np.float64)

    print("features %d, rows %d\n" % (n_features, len(xs)))
    print("Choosing the prior precision by evidence\n")
    print("  %-10s" % "tau" + "".join("%11s" % n for n in OUTPUTS))
    scores = np.zeros((len(TAU_GRID), len(OUTPUTS)))
    for i, tau in enumerate(TAU_GRID):
        for j in range(len(OUTPUTS)):
            A = G[j] + tau * np.eye(n_features)
            log_det = 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(A))))
            weights = np.concatenate([W[j], [b[j]]])
            scores[i, j] = base.log_evidence(0.0, weights, tau, log_det,
                                             n_features)
        print("  %-10g" % tau + "".join("%11.1f" % s for s in scores[i]))

    best = [TAU_GRID[int(np.argmax(scores[:, j]))] for j in range(len(OUTPUTS))]
    posteriors = np.array([np.linalg.inv(G[j] + best[j] * np.eye(n_features))
                           for j in range(len(OUTPUTS))])
    print("\n  chosen tau: " + ", ".join("%s %g" % (n, t)
                                         for n, t in zip(OUTPUTS, best)))
    if any(t in (TAU_GRID[0], TAU_GRID[-1]) for t in best):
        print("  WARNING: a chosen tau is at the edge of the grid; the best")
        print("  may lie outside it. Widen TAU_GRID and refit.")

    # The doubt on data the model was fitted on, so a run-time value can be
    # read as ordinary or not without a guessed constant.
    reference = mean_leverage(model, xs, posteriors)
    posteriors_run = np.array([base.run_posterior(posteriors[j], S[j])
                               for j in range(len(OUTPUTS))])
    reference_run = mean_leverage(model, xs, posteriors_run)

    np.savez(OUT, posteriors=posteriors, tau=np.array(best),
             reference=reference, posteriors_run=posteriors_run,
             reference_run=reference_run, n_features=n_features)
    print("\n  typical in-distribution doubt on eta1: "
          + ", ".join("%.2e" % r for r in reference))
    print("  run-level, %d flights, as a multiple of that: "
          % S.shape[1] + ", ".join("%.1fx" % (r / r0) for r, r0
                                   in zip(reference_run, reference)))
    print("\nSaved %s" % OUT.relative_to(ROOT))
    return model, saved, posteriors, reference


if __name__ == "__main__":
    model, saved, posteriors, reference = fit()
    x_mean, x_std = saved["x_mean"], saved["x_std"]

    print("\n\nDOES THE DOUBT GROW WHERE THE FILTER GOES ON THIS VEHICLE?\n")
    print("Level attitude, gentle rates, one health input moved. Training")
    print("covers 0 to 3.0; the magnetometer bias entry reaches 4.8 on a")
    print("healthy flight (quad_sim/health_readout.py).\n")
    column = INPUTS.index("severity_bias_mag")
    mag = [OUTPUTS.index(c) for c in OUTPUTS if c.startswith("mag")]
    print("  %-22s %22s" % ("bias_mag fed in", "doubt on mag, vs typical"))
    for level in [0.0, 1.0, 3.0, 4.8, 8.0, 15.0]:
        probe = np.zeros((200, len(INPUTS)))
        probe[:, 3:6] = np.linspace(-0.3, 0.3, 200)[:, None]
        probe[:, column] = level
        p = (torch.tensor(probe, dtype=torch.float32) - x_mean) / x_std
        doubt = base.epistemic_variance(model, list(posteriors), p)
        print("  %-22.1f %21.1fx" % (level, (doubt[:, mag].mean(axis=0)
                                            / reference[mag]).mean()))
    print("\n  Rising past 3.0 is the premise: the filter can drive a health")
    print("  entry somewhere the model was never fitted, and this is the only")
    print("  term that can say so.")

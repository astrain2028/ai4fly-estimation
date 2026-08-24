"""
Test 3: does the model know what it does not know, and is that worth paying
for?

The other two tests ask whether a learned model gets the right answer. This
one asks whether it can tell when it is guessing.

HOW THE QUESTION IS POSED

A model trained on every state the robot ever visits has no unfamiliar inputs
to be uncertain about, so the question cannot be asked of it. Here the model
is trained on a restricted envelope -- slower speeds and gentler turns -- and
then run on the full range. The withheld region has perfectly healthy
sensors. Nothing is broken out there. The only thing wrong is that the model
has never been shown it.

That separation matters. Elsewhere in this project a rise in uncertainty
could always be blamed on a sensor. Here it cannot.

WHAT IS BEING COMPARED

    plain + constant R      no mechanism for doubt at all
    heteroscedastic         predicts noise, but not its own ignorance
    heteroscedastic + Laplace   adds a posterior over the last layer
    ensemble of five        the same signal from disagreement, at five times
                            the inference

The ensemble is the honest upper bound. If Laplace reaches a comparable
separation at one forward pass, that is the argument. If it does not, the
ensemble is simply better and the cost is what it costs.

WHAT WOULD COUNT AS FAILING

Three ways, all worth reporting.

If the epistemic term does not rise outside the envelope, it does not work.
If it rises but the filter's consistency outside is no better for it, then it
is a signal nobody is acting on -- still useful as a flag, a much weaker
claim than calibrated uncertainty. And if the term rises everywhere rather
than specifically outside, it is measuring something other than novelty.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from common import NEES_DOF, NIS_DOF, P0, Q, best_constant_R, two_moment
from common import NEES_STATES, stack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import make_dataset
import sensors
from trajectories import DT, random_run
from ukf import UKF, nis as nis_of, nees as nees_of

from heteroscedasticity import train_heteroscedastic


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _load(ROOT / "models" / "bhr" / "train.py", "bhr_train_env")
lap = _load(ROOT / "models" / "bhr" / "laplace.py", "bhr_laplace_env")

# The envelope the model is allowed to learn from. The full data runs to
# 1.6 m/s and 1.05 rad/s, so this withholds about a fifth of it.
#
# The size of the hole matters more than it looks. An earlier version used
# 1.0 and 0.5, which withheld 56 per cent, and the network extrapolated so
# far outside that its predicted readings were meaningless -- the filter lost
# lock entirely and reported a speed error of 21 m/s on a robot that does
# 1.6. Every number downstream was then describing a diverged filter rather
# than an uncertain model, which answers no question worth asking.
#
# What this test needs is a model that is extrapolating but still roughly
# right, so that the epistemic term has something to be right or wrong about.
MAX_SPEED = 1.3
MAX_TURN = 0.7

TRAIN_RUNS = 60
EVAL_RUNS = 10
MEMBERS = 5
TAU_GRID = [0.1, 1.0, 10.0, 100.0]


class quiet:
    def __enter__(self):
        import io
        self._real, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *_):
        sys.stdout = self._real


def inside(speed, turn):
    """Is this state one the restricted model was trained on?"""
    return (speed <= MAX_SPEED) & (np.abs(turn) <= MAX_TURN)


def train_with_laplace(frame):
    """A heteroscedastic model plus a posterior over its last layer.

    Returns two callables sharing one network: one that reports only the
    noise it predicts, and one that adds its uncertainty about its own
    weights. Sharing the network is what makes the comparison fair -- any
    difference between them is the posterior and nothing else.
    """
    torch.manual_seed(0)
    x = torch.tensor(frame[bhr.INPUTS].values, dtype=torch.float32)
    y = torch.tensor(frame[bhr.OUTPUTS].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    xs, ys = (x - x_mean) / x_std, (y - y_mean) / y_std

    model = bhr.make_model()
    bhr.train(model, xs, ys, xs, ys)
    model.eval()

    # Curvature with respect to eta1 is the predicted variance -- see
    # laplace.py for why that falls out of the natural parameterisation.
    phi = lap.features(model, xs)
    with torch.no_grad():
        eta1, eta2 = bhr.split_outputs(model(xs), False)
        _, var = bhr.to_mean_and_var(eta1, eta2)
    var = var.numpy().astype(np.float64)

    last = model[-1]
    W = last.weight.detach().numpy().astype(np.float64)
    b = last.bias.detach().numpy().astype(np.float64)

    posteriors = []
    for j in range(len(bhr.OUTPUTS)):
        weights = np.concatenate([W[j], [b[j]]])
        best = (-np.inf, None)
        for tau in TAU_GRID:
            A_inv, log_det = lap.fit_posterior(phi, var[:, j], tau)
            score = lap.log_evidence(0.0, weights, tau, log_det, phi.shape[1])
            if score > best[0]:
                best = (score, A_inv)
        posteriors.append(best[1])

    trunk, head = model[:-1], model[-1]

    def make(use_laplace):
        def measure(states):
            states = np.atleast_2d(states)[:, :len(bhr.INPUTS)]
            t = torch.tensor(states, dtype=torch.float32)
            with torch.no_grad():
                hidden = trunk((t - x_mean) / x_std)
                e1, e2 = bhr.split_outputs(head(hidden), False)
                mean_s, var_s = bhr.to_mean_and_var(e1, e2)
            v = var_s.numpy().astype(np.float64)

            if use_laplace:
                f = np.concatenate([hidden.numpy().astype(np.float64),
                                    np.ones((len(states), 1))], axis=1)
                for j in range(len(bhr.OUTPUTS)):
                    spread = np.einsum("ni,ij,nj->n", f, posteriors[j], f)
                    v[:, j] += v[:, j] ** 2 * spread

            readings = (mean_s * y_std + y_mean).numpy().astype(float)
            variances = v * (y_std.numpy() ** 2)
            R = np.zeros((len(states), 3, 3))
            for k in range(len(states)):
                R[k] = np.diag(variances[k])
            return readings, R
        return measure

    def doubt(states):
        """The epistemic part alone, for the left encoder."""
        states = np.atleast_2d(states)[:, :len(bhr.INPUTS)]
        t = torch.tensor(states, dtype=torch.float32)
        with torch.no_grad():
            hidden = trunk((t - x_mean) / x_std)
        f = np.concatenate([hidden.numpy().astype(np.float64),
                            np.ones((len(states), 1))], axis=1)
        return np.einsum("ni,ij,nj->n", f, posteriors[0], f)

    return make(False), make(True), doubt


def train_ensemble(frame):
    """Five heteroscedastic models; doubt comes from their disagreement."""
    models, scaling = [], None
    x = torch.tensor(frame[bhr.INPUTS].values, dtype=torch.float32)
    y = torch.tensor(frame[bhr.OUTPUTS].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    xs, ys = (x - x_mean) / x_std, (y - y_mean) / y_std

    for member in range(MEMBERS):
        torch.manual_seed(member)
        model = bhr.make_model()
        bhr.train(model, xs, ys, xs, ys)
        model.eval()
        models.append(model)

    def parts(states):
        states = np.atleast_2d(states)[:, :len(bhr.INPUTS)]
        t = (torch.tensor(states, dtype=torch.float32) - x_mean) / x_std
        ms, vs = [], []
        with torch.no_grad():
            for model in models:
                e1, e2 = bhr.split_outputs(model(t), False)
                m, v = bhr.to_mean_and_var(e1, e2)
                ms.append((m * y_std + y_mean).numpy())
                vs.append((v * y_std ** 2).numpy())
        ms, vs = np.array(ms), np.array(vs)
        return ms.mean(0), vs.mean(0), ms.var(0)

    def measure(states):
        readings, aleatoric, epistemic = parts(states)
        total = aleatoric + epistemic
        R = np.zeros((len(readings), 3, 3))
        for k in range(len(readings)):
            R[k] = np.diag(total[k])
        return readings.astype(float), R

    def doubt(states):
        return parts(states)[2][:, 0]

    return measure, doubt


def evaluate(measure, R, doubt=None, n_runs=EVAL_RUNS):
    """Filter full-range runs and split every step by whether it was inside
    the training envelope."""
    rows = {"in": {"nis": [], "nees": [], "doubt": []},
            "out": {"nis": [], "nees": [], "doubt": []}}
    err = []

    for seed in range(300, 300 + n_runs):
        run = random_run(seed, duration=20.0)
        meas = sensors.read_sensors(run, seed=seed, dt=DT)
        accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
        turn_accel = np.diff(run["turn_rate"],
                             append=run["turn_rate"][-1]) / DT
        truth = np.column_stack([run["x"], run["y"], run["heading"],
                                 run["speed"], run["turn_rate"],
                                 accel, turn_accel])

        means, covs, innov, S = UKF(Q, R, measure=measure).run(
            stack(meas), truth[0].copy(), P0, DT)

        v = nis_of(innov, S)
        w = nees_of(means, covs, truth, states=NEES_STATES)
        d = doubt(means) if doubt is not None else np.zeros(len(means))

        # Classified on the TRUE state, not the estimate. A filter that has
        # drifted would otherwise get to decide which bucket it lands in.
        mask = inside(truth[:, 3], truth[:, 4])
        for key, sel in [("in", mask), ("out", ~mask)]:
            if sel.any():
                rows[key]["nis"].append(v[sel])
                rows[key]["nees"].append(w[sel])
                rows[key]["doubt"].append(d[sel])
        err.append(np.sqrt(np.mean((means[:, 3] - truth[:, 3]) ** 2)))

    out = {"speed": float(np.mean(err))}
    for key in ("in", "out"):
        for field in ("nis", "nees", "doubt"):
            out["%s_%s" % (key, field)] = (np.concatenate(rows[key][field])
                                           if rows[key][field]
                                           else np.array([np.nan]))
    return out


def main():
    print("TEST 3 -- KNOWING WHAT YOU DO NOT KNOW\n")
    print("Trained on speed <= %.1f m/s and |turn| <= %.1f rad/s."
          % (MAX_SPEED, MAX_TURN))
    print("Evaluated on the full range, with healthy sensors throughout.")
    print("Anything outside that envelope is unfamiliar, not broken.\n")

    full = make_dataset.build(n_runs=TRAIN_RUNS)
    keep = inside(full["speed"].values, full["turn_rate"].values)
    restricted = full[keep]
    print("training rows: %d of %d (%.0f%% withheld)\n"
          % (len(restricted), len(full), 100 * (1 - keep.mean())))

    R_const = best_constant_R(full)

    with quiet():
        aleatoric_only, with_laplace, laplace_doubt = train_with_laplace(
            restricted)
        ensemble, ensemble_doubt = train_ensemble(restricted)

    arms = [
        ("heteroscedastic", aleatoric_only, None),
        ("+ Laplace (1 pass)", with_laplace, laplace_doubt),
        ("ensemble of 5", ensemble, ensemble_doubt),
    ]

    print("%-22s %8s %18s %18s"
          % ("", "speed", "NIS in / out", "NEES in / out"))
    print("-" * 70)
    results = {}
    for label, measure, doubt in arms:
        r = evaluate(measure, R_const, doubt)
        results[label] = r
        print("%-22s %8.4f   %7.3f / %-7.3f   %7.3f / %-7.3f"
              % (label, r["speed"], r["in_nis"].mean(), r["out_nis"].mean(),
                 r["in_nees"].mean(), r["out_nees"].mean()))
    print("-" * 70)
    print("%-22s %8s   %7.1f / %-7.1f   %7.1f / %-7.1f"
          % ("target", "", NIS_DOF, NIS_DOF, NEES_DOF, NEES_DOF))

    print("\n\nDOES THE DOUBT RISE WHERE THE MODEL HAS NOT BEEN?\n")
    print("%-22s %12s %12s %10s" % ("", "inside", "outside", "ratio"))
    for label in ("+ Laplace (1 pass)", "ensemble of 5"):
        r = results[label]
        a, b = np.median(r["in_doubt"]), np.median(r["out_doubt"])
        print("%-22s %12.3e %12.3e %9.2fx" % (label, a, b, b / a))

    print("\nBoth numbers are medians: these distributions have long tails and")
    print("a handful of extreme steps would otherwise carry the average.")

    print("\n\nWHAT THE POSTERIOR BOUGHT\n")
    a = results["heteroscedastic"]
    b = results["+ Laplace (1 pass)"]
    print("Outside the envelope, NIS %.3f without the posterior and %.3f with."
          % (a["out_nis"].mean(), b["out_nis"].mean()))
    print("Target is %.1f. Closer is better, and if these are the same then" % NIS_DOF)
    print("the term is a flag rather than a correction -- a weaker claim, but")
    print("not a useless one.")


if __name__ == "__main__":
    main()

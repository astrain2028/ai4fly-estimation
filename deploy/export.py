"""
The trained model without PyTorch, for the vehicle it has to run on.

Motivation
----------

A vehicle-side runtime should not need a deep-learning framework to evaluate
a two-layer network. The health model is 5,318 parameters, 24 KB on disk, and
PyTorch is the reason the inference process needs a quarter of a gigabyte.

Peak resident memory for the same model over the same fifty steps:

    inference through PyTorch      224 MB
    inference in numpy              28 MB

Of the 28 MB, 24 KB is the model; the remainder is the interpreter and numpy.

Latency, and a hypothesis this file was written to test
-------------------------------------------------------

The full health-conditioned filter step costs about 1.5 ms. The network is
evaluated at 27 sigma points per step, about 140,000 multiply-accumulates,
which is microseconds of arithmetic -- so it was expected that most of the
1.5 ms was framework dispatch, and that removing PyTorch would recover it.

The self-test below measures the network alone, and the expectation was
wrong. Through PyTorch it takes about 0.25 ms per step, and in numpy about
0.12 ms. The network is therefore under a fifth of the step, and removing the
framework recovers about a tenth of it. The remainder is the unscented
transform itself: robot/ukf.py builds sigma points, rebuilds means and
covariances, and accumulates the cross-covariance in Python loops that call
np.outer once per point. That is where a faster implementation would have to
look, and the framework is not.

The memory result stands on its own and is the reason to deploy this path.

Model structure
---------------

Three matrix multiplies and two clamps:

    h = relu( (x - x_mean)/x_std @ W1' + b1 )
    h = relu( h @ W2' + b2 )
    out =     h @ W3' + b3

then the natural parameters split off the back of it, which is arithmetic
rather than machinery:

    eta1 = out[:, :n]
    eta2 = -softplus(out[:, n:]) - 1e-6      negative by construction
    var  = -0.5 / eta2
    mean = eta1 * var

Nothing in that needs autograd, a device, or a graph. The training does; the
flying does not.

Scope
-----

It is not a reimplementation. The weights are the trained ones, unmodified,
and the self-test below checks that the two paths agree to float32 rounding
rather than assuming it. A separate implementation that merely looked right
would be a second thing to keep in step with the first.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import time

import numpy as np


def _timeit(fn, states, repeats=50):
    """Milliseconds per call, averaged over a burst."""
    started = time.perf_counter()
    for _ in range(repeats):
        fn(states)
    return 1000.0 * (time.perf_counter() - started) / repeats


def export(model_path, out_path):
    """Read a trained .pt and write a .npz with no torch dependency.

    Runs offline, on a machine that has torch. The result is what ships.
    """
    import torch

    saved = torch.load(model_path, weights_only=False)
    weights = {k: v.numpy() for k, v in saved["weights"].items()}
    np.savez(out_path, **weights,
             x_mean=saved["x_mean"].numpy(), x_std=saved["x_std"].numpy(),
             y_mean=saved["y_mean"].numpy(), y_std=saved["y_std"].numpy(),
             n_outputs=np.array(len(saved["outputs"])))
    return out_path


def _softplus(x):
    """log(1 + exp(x)), without overflowing for large x."""
    return np.logaddexp(0.0, x)


def load(npz_path, take=None, n_vehicle=None):
    """A measurement model in numpy alone.

    `take` picks the state entries the model reads, and `n_vehicle` says how
    many of those come before the health entries -- health is clipped at zero
    on the way in, because the model was never shown a negative degradation
    and returns the same answer below zero as at it.
    """
    z = np.load(npz_path)
    layers = []
    index = 0
    while "%d.weight" % index in z:
        layers.append((z["%d.weight" % index], z["%d.bias" % index]))
        index += 2                      # ReLU sits between, carrying no weights

    x_mean, x_std = z["x_mean"], z["x_std"]
    y_mean, y_std = z["y_mean"], z["y_std"]
    n = int(z["n_outputs"])

    def measure(states):
        states = np.atleast_2d(np.asarray(states, dtype=float))
        picked = states if take is None else states[:, take]
        if n_vehicle is not None:
            picked = picked.copy()
            picked[:, n_vehicle:] = np.maximum(picked[:, n_vehicle:], 0.0)

        h = (picked - x_mean) / x_std
        for W, b in layers[:-1]:
            h = np.maximum(h @ W.T + b, 0.0)
        W, b = layers[-1]
        out = h @ W.T + b

        eta1 = out[:, :n]
        eta2 = -_softplus(out[:, n:]) - 1e-6
        var = -0.5 / eta2
        mean = eta1 * var

        readings = mean * y_std + y_mean
        variances = var * y_std ** 2

        R = np.zeros((len(states), n, n))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    return measure


if __name__ == "__main__":
    import importlib.util

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    out_dir = Path(__file__).parent / "weights"
    out_dir.mkdir(exist_ok=True)

    targets = [
        ("health", ROOT / "models" / "health" / "health_model.pt",
         ROOT / "models" / "health" / "measurement.py", 13, 5),
        ("quad", ROOT / "quad_sim" / "quad_health.pt",
         ROOT / "quad_sim" / "measurement.py", 12, 6),
    ]

    print("EXPORTING TRAINED MODELS TO NUMPY\n")
    for name, pt, module_path, n_states, n_vehicle in targets:
        if not pt.exists():
            print("  %-8s no weights yet -- train it first" % name)
            continue

        npz = out_dir / ("%s.npz" % name)
        export(pt, npz)
        print("  %-8s %s -> %s  (%d KB)"
              % (name, pt.name, npz.name, npz.stat().st_size // 1024))

    print("\n\nDO THE TWO PATHS AGREE?\n")
    print("  %-8s %16s %16s" % ("", "readings", "variances"))

    for name, pt, module_path, n_states, n_vehicle in targets:
        npz = out_dir / ("%s.npz" % name)
        if not npz.exists():
            continue

        module = _load(module_path, "exported_%s" % name)
        reference = module.load_measurement_model()
        if hasattr(reference, "base"):        # unwrap a Layered arm
            reference = reference.base
        take = getattr(module, "TAKE", None)
        numpy_only = load(npz, take=take, n_vehicle=n_vehicle)

        rng = np.random.default_rng(0)
        states = rng.normal(size=(2 * n_states + 1, n_states))
        states[:, n_vehicle if take is None else take[n_vehicle]:] = np.abs(
            states[:, n_vehicle if take is None else take[n_vehicle]:])

        got_r, got_R = numpy_only(states)
        want_r, want_R = reference(states)
        print("  %-8s %16.3e %16.3e"
              % (name, np.abs(got_r - want_r).max(),
                 np.abs(got_R - want_R).max()))

    print("\n  Differences at float32 rounding are the two paths agreeing.")
    print("  Anything larger is a port that looked right and was not.")

    print("\n\nWHAT IT COSTS\n")
    npz = out_dir / "health.npz"
    if npz.exists():
        module = _load(ROOT / "models" / "health" / "measurement.py",
                       "timing_health")
        torch_path = module.load_measurement_model()
        numpy_path = load(npz, take=module.TAKE, n_vehicle=5)
        states = np.zeros((27, 13))
        states[:, 3] = 0.8

        for label, fn in [("through PyTorch", torch_path),
                          ("numpy only", numpy_path)]:
            for _ in range(20):
                fn(states)
            best = min(min(_timeit(fn, states) for _ in range(5))
                       for _ in range(3))
            print("  %-18s %8.3f ms per filter step" % (label, best))

        print("\n  Twenty-seven sigma points through three matrix multiplies.")
        print("  The arithmetic is the same in both rows.")

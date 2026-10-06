"""
layered's second term: what the model does not know, in numpy alone.

One implementation, three callers -- the ground robot's layered arm, the
quadcopter's, and the vehicle-side runtime -- so the term cannot mean
something different in the simulator from what it means in flight. It needs
only the network's weights as arrays and a fitted last-layer posterior;
from_torch reads them out of a trained .pt, and deploy/export.py writes the
same arrays into the file that ships.

What it computes
----------------

For each channel j, with phi the last hidden layer and a trailing 1:

    leverage_j  = phi' C_j phi                  spread on eta1
    R_epi_j     = var_j^2 * leverage_j * y_std_j^2

C_j is the run-level posterior when one has been fitted (models/bhr/
laplace.py run_posterior), and the ordinary Laplace one otherwise. The
carry-through from eta1 to the reading is the one models/doubt uses: the mean
is eta1 times the predicted variance.

Two things it does beyond that
------------------------------

**The multiplier is taken where the model is valid.** var_j comes from the
network's own variance head, and that head extrapolates badly: push a health
input to 20 and the predicted variance runs to its ceiling, about 10^8 times its
healthy value, and R_epi -- which goes as its square -- with it. That is the
variance head breaking, not the model admitting doubt. So var_j is evaluated
with the health inputs held to the training range, CEILING, while leverage is
evaluated at the real input, which is where the novelty is.

**It reports how unusual the input is.** ratio_j is leverage_j divided by its
average over the training rows. It is 1 on ordinary inputs and grows where
the model has not been, and unlike R_epi it does not depend on the variance
head at all. That is the number to watch in flight: deploy/mavlink_sink.py
sends it per device.
"""

import numpy as np

CEILING = 3.0          # the largest health level either training set contains


def _softplus(x):
    return np.logaddexp(0.0, x)


def wrap_angle(angle):
    """An angle, or array of them, in radians, into (-pi, pi].

    The quadcopter's model takes yaw as a raw angle, and the filter's yaw
    keeps counting past a half turn -- the training data spans -3.6 to 4.0
    rad, a real flight of a few minutes spans many turns, and the unscented
    transform's sigma points reach further still while heading is uncertain.
    The readings depend on heading only through its sine and cosine, so
    wrapping is exact, and it keeps every query where the model was trained.
    """
    return np.pi - np.mod(np.pi - np.asarray(angle, dtype=float), 2 * np.pi)


class Epistemic:
    """Epistemic variance per channel, and the input's novelty ratio."""

    def __init__(self, layers, x_mean, x_std, y_std, posteriors, reference,
                 take=None, n_vehicle=None, ceiling=CEILING, wrap=()):
        self.layers = [(np.asarray(W, float), np.asarray(b, float))
                       for W, b in layers]
        self.x_mean = np.asarray(x_mean, float)
        self.x_std = np.asarray(x_std, float)
        self.y_var = np.asarray(y_std, float) ** 2
        self.posteriors = np.asarray(posteriors, float)
        self.reference = np.maximum(np.asarray(reference, float), 1e-300)
        self.take = take
        self.n_vehicle = n_vehicle
        self.ceiling = ceiling
        self.wrap = list(wrap)              # input angles to hold in (-pi, pi]
        self.n = len(self.posteriors)
        self.ratio = np.ones(self.n)        # of the last call, mean over points

        # A posterior fitted to one network and used with another would give
        # numbers that look plausible and mean nothing. The dimensions are the
        # cheapest check that the pair belongs together: one posterior per
        # channel, each as wide as the last hidden layer plus the bias.
        W_last = self.layers[-1][0]
        if (self.posteriors.shape[1:] != (W_last.shape[1] + 1,) * 2
                or 2 * self.n != W_last.shape[0]):
            raise ValueError(
                "posterior %s does not fit a network whose last layer is %s; "
                "refit it for this model" % (self.posteriors.shape,
                                             W_last.shape))

    def _forward(self, picked):
        """Last hidden layer and predicted variance, in scaled units."""
        h = (picked - self.x_mean) / self.x_std
        for W, b in self.layers[:-1]:
            h = np.maximum(h @ W.T + b, 0.0)
        W, b = self.layers[-1]
        out = h @ W.T + b
        var = -0.5 / (-_softplus(out[:, self.n:]) - 1e-6)
        return h, var

    def __call__(self, states):
        states = np.atleast_2d(np.asarray(states, dtype=float))
        picked = (states if self.take is None else states[:, self.take]).copy()
        picked[:, self.wrap] = wrap_angle(picked[:, self.wrap])
        if self.n_vehicle is not None:
            picked[:, self.n_vehicle:] = np.maximum(picked[:, self.n_vehicle:],
                                                    0.0)
        h, var = self._forward(picked)
        phi = np.hstack([h, np.ones((len(h), 1))])
        # phi' C_j phi for every point and channel: a batched matrix multiply
        # and a row-wise dot. A single four-index einsum computes the same
        # thing as an unvectorised loop, about ten times slower.
        leverage = (np.matmul(phi[None], self.posteriors) * phi[None]).sum(-1).T

        # The variance multiplier from health inputs held to the training
        # range. On almost every step none is above it and the forward pass
        # already made gives the same answer, so the second is run only when
        # one is.
        if (self.n_vehicle is not None and self.ceiling is not None
                and np.any(picked[:, self.n_vehicle:] > self.ceiling)):
            held = picked.copy()
            held[:, self.n_vehicle:] = np.minimum(held[:, self.n_vehicle:],
                                                  self.ceiling)
            _, var = self._forward(held)

        self.ratio = (leverage / self.reference).mean(axis=0)
        return var ** 2 * leverage * self.y_var

    @classmethod
    def from_arrays(cls, arrays, prefix="", **kwargs):
        """From a mapping holding the network and posterior as numpy arrays.

        Accepts either a saved state_dict (keys '0.weight', '0.bias', ...) or
        deploy/export.py's npz, which uses the same keys. The posterior is
        read from 'laplace_posteriors' and 'laplace_reference'.
        """
        layers, index = [], 0
        while "%d.weight" % index in arrays:
            layers.append((arrays["%d.weight" % index],
                           arrays["%d.bias" % index]))
            index += 2
        return cls(layers, arrays["x_mean"], arrays["x_std"], arrays["y_std"],
                   arrays["laplace_posteriors"], arrays["laplace_reference"],
                   **kwargs)

    @classmethod
    def from_torch(cls, model_path, laplace_path, **kwargs):
        """From a trained .pt and a fitted posterior. Needs torch to read."""
        import torch
        saved = torch.load(model_path, weights_only=False)
        arrays = {k: v.numpy() for k, v in saved["weights"].items()}
        for key in ("x_mean", "x_std", "y_std"):
            arrays[key] = saved[key].numpy()
        posterior, reference = read_posterior(laplace_path)
        arrays["laplace_posteriors"] = posterior
        arrays["laplace_reference"] = reference
        return cls.from_arrays(arrays, **kwargs)


def read_posterior(laplace_path):
    """The run-level posterior and its reference if fitted, else Laplace's."""
    z = np.load(laplace_path)
    if "posteriors_run" in z.files:
        return z["posteriors_run"], z["reference_run"]
    return z["posteriors"], z["reference"]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    vehicles = [
        ("ground robot", ROOT / "models" / "health" / "health_model.pt",
         ROOT / "models" / "doubt" / "doubt_laplace.npz", 13,
         [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12], 5, 7),
        ("quadcopter", ROOT / "quad_sim" / "quad_health.pt",
         ROOT / "quad_sim" / "quad_laplace.npz", 12, None, 6, 6),
    ]

    print("THE EPISTEMIC TERM, ON ORDINARY AND UNFAMILIAR INPUTS\n")
    ok = True
    for name, pt, npz, n_states, take, n_vehicle, first_health in vehicles:
        if not (pt.exists() and npz.exists()):
            print("  %-13s no model or posterior yet -- train and fit first"
                  % name)
            continue
        term = Epistemic.from_torch(pt, npz, take=take, n_vehicle=n_vehicle)
        # An ordinary moment, as the repository's Laplace probes use: the
        # robot cruising at 0.8 m/s through gentle turns, the quadcopter near
        # level with gentle rates. An all-zero state is not ordinary for the
        # robot -- it never stands still in training -- and reads as novel.
        states = np.zeros((25, n_states))
        sweep = np.linspace(-0.3, 0.3, 25)
        if name == "ground robot":
            states[:, 3], states[:, 4] = 0.8, sweep
        else:
            states[:, 3:6] = sweep[:, None]
        R = term(states)
        ordinary = float(term.ratio.max())
        states[:, first_health] = 15.0              # far past training
        R_far = term(states)
        far = float(term.ratio.max())
        print("  %-13s novelty %.2f ordinarily, %.1f with a health entry at 15;"
              " R_epi grows %.0fx" % (name, ordinary, far,
                                      (R_far / R).max()))
        ok = ok and np.all(R >= 0) and far > ordinary
    print("\n  Novelty should sit near a few ordinarily and rise well past it off")
    print("  the training range, and the term should never be negative.")
    sys.exit(0 if ok else 1)

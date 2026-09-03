"""
The health model, its own doubt, and a multiplier that listens to it.

WHAT THIS ADDS TO models/combined

The combined arm carries two mechanisms: a learned covariance conditioned on
health, and an adaptive multiplier on top of it for the fault modes health
cannot see. It works, and it has one clear weakness -- everything about how
the multiplier behaves is a constant somebody chose. It may not fall below
1.0, it moves at 5 per cent per update, and it averages over a hundred steps
before moving at all. Those were tuned once, against the steady state, and
they are the same numbers whether the model is on ground it knows well or
ground it has never seen.

That is the wrong shape for the problem. When the learned model is reliable
the multiplier SHOULD be sluggish, because moving it would be second-guessing
a model that is right. When the learned model has no idea, the multiplier is
the only thing left and its caution is pure lag.

So the two constants become one signal. The model says how much it trusts
itself; the multiplier moves in proportion to how little that is.

WHY THE HEALTH MODEL IS THE ONE WHERE THIS CAN WORK

An epistemic term measures novelty in the model's INPUTS. On every other arm
those are the five vehicle states, and a broken sensor does not change them
-- the robot drives normally and only the reading is wrong -- so no amount of
Laplace or ensembling sees a fault. Measured on this model's inputs, though,
health is among them, and the filter estimates health and feeds it back. A
fault the model cannot represent drives that estimate wherever best explains
the residual, and the probe in laplace.py says the doubt responds: 1.9x
typical at the training edge of 3.0, 8.6x at 8.0, 30.7x at 15.0.

So the question the arm can ask is not "is this fault familiar", which is
unanswerable, but "is this health level familiar", which is not. Those are
different questions and only the second has ever been available.

TWO USES, AND THEY ARE NOT THE SAME USE

The epistemic variance is added to R, which is the ordinary thing to do with
it and makes the filter appropriately unsure where the model is guessing.

It also sets how fast the multiplier may move, which is the part that is not
ordinary. That is a hand-off: as the learned mechanism loses confidence, the
classical one is allowed to take over, and the trigger is the model's own
admission rather than a threshold somebody picked.

WHAT WOULD FALSIFY THE IDEA

If the filter never drives health outside its training range on faults the
model cannot represent, the doubt never rises, the multiplier never speeds
up, and this arm is models/combined with extra arithmetic. The probe says the
doubt WOULD respond if health went there; whether the filter takes it there
is a separate question and is measured below.
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = _load(ROOT / "models" / "health" / "measurement.py", "health_for_doubt")
health_train = _load(ROOT / "models" / "health" / "train.py", "health_train_for_doubt_m")
base = _load(ROOT / "models" / "bhr" / "laplace.py", "bhr_laplace_for_doubt_m")

N_STATES = health.N_STATES
HEALTH_STATES = health.HEALTH_STATES
TAKE = health.TAKE
filter_settings = health.filter_settings

WINDOW = 100          # innovations averaged over, as in models/combined
BLEND = 0.05          # how fast the multiplier moves when the model is sure
LIMITS = (1.0, 25.0)

# How much faster the multiplier may move when the model doubts itself. A
# doubt ratio of 1 is ordinary and leaves the blend at BLEND; the cap stops a
# single strange step from throwing the multiplier across its whole range.
DOUBT_GAIN = 1.0
MAX_BLEND = 0.50


class Doubtful:
    """Health-conditioned readings, plus what the model does not know."""

    def __init__(self, base_measure, model, posteriors, reference, scaling,
                 window=WINDOW, blend=BLEND, limits=LIMITS):
        self.base = base_measure
        self.model = model
        self.posteriors = posteriors
        self.reference = np.maximum(np.asarray(reference, dtype=float), 1e-12)
        self.x_mean, self.x_std, self.y_std = scaling

        self.scale = np.ones(3)
        self.history = deque(maxlen=window)
        self.window = window
        self.blend = blend
        self.limits = limits
        self.ratio = 1.0
        self.trace = []

    def _doubt(self, states):
        """Epistemic variance on the predicted reading, per sigma point.

        The curvature identity in bhr/laplace.py gives the spread on eta1;
        carrying it through to the mean multiplies by the predicted variance
        squared, and the y_std term puts it back in the sensors' own units.
        """
        picked = states[:, TAKE].copy()
        picked[:, 5:] = np.clip(picked[:, 5:], 0.0, None)
        x = torch.tensor(picked, dtype=torch.float32)
        scaled = (x - self.x_mean) / self.x_std

        with torch.no_grad():
            eta1, eta2 = health_train.bhr.split_outputs(self.model(scaled), False)
            _, var_s = health_train.bhr.to_mean_and_var(eta1, eta2)
        var_s = var_s.numpy().astype(np.float64)

        epi_eta1 = base.epistemic_variance(self.model, self.posteriors, scaled)
        ys = self.y_std.numpy().astype(np.float64)
        return var_s ** 2 * epi_eta1 * ys ** 2, epi_eta1

    def __call__(self, states):
        states = np.atleast_2d(states)
        readings, R = self.base(states)
        epistemic, raw = self._doubt(states)

        # How unusual this is, against what the model saw in training. Kept
        # as one number rather than three: the multiplier's speed is a
        # statement about the model as a whole, and a single bad channel is
        # already handled by the per-channel multiplier itself.
        self.ratio = float(np.mean(raw.mean(axis=0) / self.reference))

        for k in range(len(states)):
            R[k] = R[k] + np.diag(epistemic[k])
        return readings, R * self.scale[None, :, None]

    def observe(self, innovation, S):
        """Adjust the multiplier, at a speed the model's doubt sets."""
        innovation = np.asarray(innovation, dtype=float)
        diag = np.maximum(np.diag(np.asarray(S, dtype=float)), 1e-12)
        self.history.append(innovation ** 2 / diag)

        if len(self.history) == self.window:
            ratio = np.maximum(np.mean(np.array(self.history), axis=0), 1e-6)
            # The one line this arm exists for. Sure of itself, and the
            # multiplier crawls; lost, and it is allowed to move.
            blend = min(self.blend * (1.0 + DOUBT_GAIN * (self.ratio - 1.0)),
                        MAX_BLEND)
            blend = max(blend, self.blend)
            self.scale = np.clip(self.scale * ratio ** blend,
                                 self.limits[0], self.limits[1])

        self.trace.append((self.ratio, self.scale.copy()))

    def constrain(self, mean):
        mean = np.array(mean, dtype=float)
        mean[HEALTH_STATES] = np.maximum(mean[HEALTH_STATES], 0.0)
        return mean

    def reset(self):
        self.scale = np.ones(3)
        self.history.clear()
        self.ratio = 1.0
        self.trace = []


def load_measurement_model(path=None):
    """The health arm, its Laplace posterior, and a doubt-driven multiplier."""
    if path is None:
        path = ROOT / "models" / "health" / "health_model.pt"
    saved = torch.load(path, weights_only=False)

    model = health_train.make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    stored = np.load(Path(__file__).parent / "doubt_laplace.npz")
    posteriors = [p for p in stored["posteriors"]]

    return Doubtful(health.load_measurement_model(path), model, posteriors,
                    stored["reference"],
                    (saved["x_mean"], saved["x_std"], saved["y_std"]))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import Q, P0, best_constant_R
    from faults import apply_fault
    from sensors import read_sensors
    from trajectories import DT, random_run
    from ukf import UKF

    measure = load_measurement_model()
    combined = _load(ROOT / "models" / "combined" / "measurement.py",
                     "combined_for_doubt").load_measurement_model()
    Q_use, P_use = filter_settings(Q, P0)
    R_const = best_constant_R()

    SEEDS = range(700, 704)

    def go(arm, mode, severity):
        """One arm over several runs; error, and where health ended up."""
        errors, peak_health, doubts = [], [], []
        for seed in SEEDS:
            run = random_run(seed, duration=20.0)
            meas = read_sensors(run, seed=seed, dt=DT)
            if severity > 0:
                meas = apply_fault(meas, "left_encoder", mode, severity,
                                   seed=seed, dt=DT)
            readings = np.column_stack([meas["left_encoder"],
                                        meas["right_encoder"], meas["gyro"]])
            start = np.zeros(N_STATES)
            start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                         run["speed"][0], run["turn_rate"][0]]

            arm.reset()
            means, _, _, _ = UKF(Q_use, R_const, measure=arm).run(
                readings, start, P_use, DT)
            errors.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
            half = len(means) // 2
            peak_health.append(means[half:][:, HEALTH_STATES].max())
            if hasattr(arm, "trace") and arm.trace and isinstance(
                    arm.trace[0], tuple):
                doubts.append(np.mean([t[0] for t in arm.trace[half:]]))
        return (float(np.mean(errors)), float(np.mean(peak_health)),
                float(np.mean(doubts)) if doubts else float("nan"))

    print("DOES THE FILTER ACTUALLY GO SOMEWHERE THE MODEL HAS NOT BEEN?\n")
    print("Training covers health from 0.25 to 3.0. The two right-hand modes")
    print("are ones the model has never seen in any form.\n")
    print("  %-18s %8s %12s %12s %10s"
          % ("mode", "severity", "peak health", "doubt", "speed"))
    print("  " + "-" * 64)

    for mode, severity in [("none", 0.0), ("bias", 2.0),
                           ("noise_inflation", 2.0), ("stuck", 0.5),
                           ("scale_error", 2.0)]:
        error, peak, doubt = go(measure, mode, severity)
        print("  %-18s %8.2f %12.2f %12.1fx %10.4f"
              % (mode, severity, peak, doubt, error))

    print("  " + "-" * 64)
    print("\n  A peak health well past 3.0 on the unfamiliar modes is the")
    print("  whole premise: the filter asks the model something it was never")
    print("  taught, and the doubt column is whether the model notices.")

    print("\n\nAGAINST THE ARM IT IS TRYING TO IMPROVE ON\n")
    print("  %-18s %8s %12s %12s"
          % ("mode", "severity", "combined", "doubt-driven"))
    print("  " + "-" * 54)
    for mode, severity in [("none", 0.0), ("bias", 2.0),
                           ("noise_inflation", 2.0), ("stuck", 0.5),
                           ("scale_error", 2.0)]:
        was, _, _ = go(combined, mode, severity)
        now, _, _ = go(measure, mode, severity)
        print("  %-18s %8.2f %12.4f %12.4f" % (mode, severity, was, now))
    print("  " + "-" * 54)
    print("\n  The first two rows are what the health model handles well and")
    print("  should not move. The last two are the ones it cannot represent,")
    print("  and are where letting the multiplier off its leash has to pay.")

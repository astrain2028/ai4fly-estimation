"""
Three sources of ignorance, added rather than multiplied.

WHAT IS WRONG WITH models/combined

The combined arm scales the learned covariance by a multiplier that
covariance matching estimates online:

    R_used = c * R_model(x, m)

It works, and every awkward thing about it follows from that one choice of
algebra. A multiplier cannot decompose, so there is no way to ask which
mechanism claimed which part of the covariance -- and without that, nothing
stops the two from correcting the same error twice. The floor at 1.0 exists
to prevent it, and the floor is an empirical fudge: it was arrived at by
noticing that letting the multiplier fall settles at 0.72 on healthy data,
which fixes NIS from 1.98 to 3.08 and makes the estimate worse, 0.0074 to
0.0097.

That is a real measurement and the floor is the right call given the form.
But it is a patch on the form.

VARIANCES ADD

    R_total  =  R_aleatoric(x, m)  +  R_epistemic(x, m)  +  R_unmodelled

Three terms, three kinds of not-knowing, each estimated by the only mechanism
that can see it:

    R_aleatoric    how noisy the sensor is, given the state and how degraded
                   it is. The learned heteroscedastic head.
    R_epistemic    how unsure the model is about that claim. The last-layer
                   posterior, over inputs that include the six health states
                   the filter estimates and feeds back. Optional -- present
                   only if models/doubt has been fitted.
    R_unmodelled   whatever neither of them accounted for.

and the third is estimated the way Mehra estimated it in 1970: as the part of
the observed innovation covariance that the filter's own prediction does not
already explain.

    R_unm  <-  max( 0,  E[nu nu'] - spread - R_ale - R_epi )

where spread is the sigma-point scatter of the predicted measurements, which
is S minus whatever R went in. Everything on the right is already computed
every step.

WHAT THAT BUYS

The floor stops being a fudge. It becomes max(0, .), because a variance that
has not been accounted for cannot be negative. Same behaviour, derived rather
than tuned.

The gain schedule stops being needed. models/doubt speeds the multiplier up
when the model is unsure; here, a model that is unsure has already inflated
R_epi, so S is already large, so the residual goes to zero on its own and the
adaptive layer correctly stands down. Note the sign is opposite to doubt's,
and this one falls out of the algebra instead of being chosen.

And it becomes attributable. At any step you can say how much of the
covariance came from sensor noise, from model ignorance, and from something
nobody modelled -- three diagnostic channels where a multiplier gives one
opaque number.

CONSTANTS REMOVED

    combined:  WINDOW, BLEND, LIMITS low, LIMITS high        4
    doubt:     the two above plus DOUBT_GAIN, MAX_BLEND      6
    here:      FORGET                                        1

The window goes because the innovation covariance is tracked by exponential
forgetting rather than a fixed buffer, which is the ordinary way to do it and
avoids a hard edge at the buffer boundary.
"""

import importlib.util
import sys
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


health = _load(ROOT / "models" / "health" / "measurement.py", "health_for_layered")

N_STATES = health.N_STATES
HEALTH_STATES = health.HEALTH_STATES
TAKE = health.TAKE
filter_settings = health.filter_settings

# How fast the running innovation covariance forgets. 0.01 gives a time
# constant near a hundred steps, two seconds at 50 Hz, which is where
# combined's fixed window sat -- so the comparison is about the algebra and
# not about one arm being allowed to react faster than the other.
FORGET = 0.01


class Layered:
    """Aleatoric, epistemic and unmodelled variance, summed.

    Nothing here is specific to the ground robot. The only thing the class
    needs to know about the vehicle is which entries of the state are health,
    so that constrain can keep them non-negative, and how many channels there
    are. Both are arguments, which is what lets quad_sim reuse the algebra
    rather than copy it -- and a copy would be two implementations of the same
    estimator free to drift apart.
    """

    def __init__(self, base, epistemic=None, health_states=None, n_channels=3):
        self.base = base
        self.epistemic = epistemic          # None, or a callable like doubt's
        self.health_states = (HEALTH_STATES if health_states is None
                              else list(health_states))
        self.n_channels = n_channels
        self.reset()

    def reset(self):
        self.covariance = None              # running E[nu nu'], diagonal
        self.unmodelled = np.zeros(self.n_channels)
        self.last_R = np.zeros(self.n_channels)  # what went into the last update
        self.trace = []

    def __call__(self, states):
        states = np.atleast_2d(states)
        readings, R = self.base(states)

        if self.epistemic is not None:
            extra = self.epistemic(states)
            for k in range(len(states)):
                R[k] = R[k] + np.diag(extra[k])

        # The average R the filter will actually use. The UKF weights the
        # sigma points when it averages and this does not, which is a
        # second-order error: R varies little across points that differ only
        # by the spread of the current estimate. It is used to recover the
        # measurement scatter in observe, not to set the covariance itself.
        self.last_R = np.diag(R.mean(axis=0)).copy() + self.unmodelled

        for k in range(len(states)):
            R[k] = R[k] + np.diag(self.unmodelled)
        return readings, R

    def observe(self, innovation, S):
        """Estimate the variance nothing has accounted for.

        S is the scatter of the predicted measurements plus the R that went
        in, so subtracting that R recovers the scatter alone. What the
        innovations turn out to be, minus what the filter already predicted
        for reasons it can name, is what is left over.
        """
        innovation = np.asarray(innovation, dtype=float)
        observed = innovation ** 2

        if self.covariance is None:
            self.covariance = observed.copy()
        else:
            self.covariance += FORGET * (observed - self.covariance)

        predicted = np.diag(np.asarray(S, dtype=float))
        scatter = np.maximum(predicted - self.last_R, 0.0)

        # A variance nobody has modelled cannot be negative. That is the whole
        # of the constraint, and it replaces combined's floor at 1.0.
        target = np.maximum(self.covariance - scatter
                            - (self.last_R - self.unmodelled), 0.0)
        self.unmodelled += FORGET * (target - self.unmodelled)
        self.trace.append(self.unmodelled.copy())

    def constrain(self, mean):
        mean = np.array(mean, dtype=float)
        mean[self.health_states] = np.maximum(mean[self.health_states], 0.0)
        return mean


def load_measurement_model(path=None, with_epistemic=False):
    """The health arm, plus an additive residual, optionally plus doubt.

    The epistemic term is off by default. It needs models/doubt/laplace.py to
    have been fitted, and the point of this arm is the algebra of the third
    term rather than the presence of the second.
    """
    base = health.load_measurement_model(path)
    epistemic = None

    if with_epistemic:
        doubt = _load(ROOT / "models" / "doubt" / "measurement.py",
                      "doubt_for_layered")
        wrapped = doubt.load_measurement_model(path)

        def epistemic(states):
            return wrapped._doubt(states)[0]

    return Layered(base, epistemic)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import Q, P0, best_constant_R
    from faults import apply_fault
    from sensors import read_sensors
    from trajectories import DT, random_run
    from ukf import UKF, nis as nis_of

    combined = _load(ROOT / "models" / "combined" / "measurement.py",
                     "combined_for_layered").load_measurement_model()
    layered = load_measurement_model()
    Q_use, P_use = filter_settings(Q, P0)
    R_const = best_constant_R()

    def go(arm, mode, severity):
        errors, nis_all = [], []
        for seed in range(2000, 2008):
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
            if hasattr(arm, "reset"):
                arm.reset()
            means, _, innov, S = UKF(Q_use, R_const, measure=arm).run(
                readings, start, P_use, DT)
            errors.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
            nis_all.append(nis_of(innov, S).mean())
        return float(np.mean(errors)), float(np.mean(nis_all))

    print("ADDITIVE RESIDUAL AGAINST MULTIPLICATIVE SCALING\n")
    print("Same runs and same seeds as the bakeoff. combined carries four")
    print("tuned constants for its multiplier; this carries one.\n")
    print("  %-20s %11s %9s %11s %9s"
          % ("", "combined", "NIS", "layered", "NIS"))
    print("  " + "-" * 64)

    for mode, severity in [("none", 0.0), ("bias", 3.0),
                           ("noise_inflation", 3.0), ("drift", 3.0),
                           ("scale_error", 3.0)]:
        was, was_nis = go(combined, mode, severity)
        now, now_nis = go(layered, mode, severity)
        print("  %-20s %11.4f %9.2f %11.4f %9.2f"
              % ("%s %.1f" % (mode, severity), was, was_nis, now, now_nis))

    print("  " + "-" * 64)
    print("\n  NIS should sit near 3. The claim is not that this is more")
    print("  accurate -- it is that it reaches similar behaviour with one")
    print("  constant instead of four, and that the floor is arithmetic")
    print("  rather than a number somebody chose.")

"""
Multiple-model adaptive estimation: the classical answer to this problem.

WHY THIS ARM EXISTS

Every other arm in this project is a measurement model, and the comparison
between them is which map and which covariance to learn. MMAE is not that. It
is a different filter architecture, and it is the one a reviewer familiar with
fault detection will reach for first, so the health-augmented state has to be
measured against it rather than argued past.

The idea is older and simpler than anything learned here. Rather than
estimating how degraded a sensor is, enumerate the ways it could be broken,
run one filter per hypothesis, and let each filter's own innovations say how
plausible its hypothesis is. Maybeck applied it to sensor and actuator
failures in aircraft flight control; Hanlon and Maybeck refined the test to
use the correlation of the residual rather than its magnitude.

    w_i  <-  w_i * p(innovation | hypothesis i)

normalised, and the reported estimate is the probability-weighted blend. A
filter whose assumed fault matches reality predicts its measurements well, so
its weight rises; the rest fall.

WHAT IT COSTS AND WHAT THAT BUYS

Seven hypotheses means seven filters, each with its own state, covariance and
sigma points. That is the comparison the README makes: a bank costs N filters
where augmenting health costs one wider filter, fifteen sigma points against
twenty-seven. Until this file existed that claim was asserted rather than
measured, which made it the only comparative claim in the project not backed
by a run.

WHERE THE BANK COMES FROM, AND WHY THAT IS THE HARD PART

MMAE needs a discrete set of hypotheses and the fault space here is
continuous: three channels, two modes, severity anywhere from nothing to
severe. So the bank has to be chosen, and the choice decides the answer. A
bank placed at the severities used for testing wins; a coarse one loses. That
is a knob, and turning it to suit a conclusion is exactly what this project
has tried to avoid elsewhere.

So the bank is fixed on grounds that do not mention the test: healthy, plus
one hypothesis per channel per mode at a single mid-range severity. Seven
filters. Testing then happens BELOW, AT and ABOVE that severity, and the
interesting question is what happens at the two that the bank does not sit on.

That is the real comparison. Not "does a bank work" -- it does, on the fault
it was built for -- but whether a continuous health state is worth having
where the fault is not one of the ones somebody wrote down.

TWO THINGS THAT WOULD MAKE THIS A STRAWMAN IF LEFT OUT

Weights are floored. Left alone, a hypothesis whose likelihood is tiny for a
few hundred steps has its weight driven to zero in floating point and can
never recover, so the bank stops being able to respond to a fault that
arrives later. Maybeck's own treatment puts a lower bound on the weights for
this reason, and without it the onset comparison would be measuring an
implementation bug rather than the method.

The reported estimate is the weighted blend and not the single most likely
filter. Picking the argmax throws away the spread between hypotheses, which
is most of what the bank knows when it is genuinely unsure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np

import faults
from ukf import UKF, expected_readings

# One severity for the whole bank, chosen mid-range and without reference to
# what the tests use. Everything about the result depends on this number, so
# it is a constant with a name rather than a literal buried in a loop.
BANK_SEVERITY = 1.5

# No hypothesis may fall below this share of the total. See the docstring:
# without it a bank that has ruled something out has ruled it out forever.
WEIGHT_FLOOR = 1e-3

MODES = ["bias", "noise_inflation"]


def hypothesis_model(channel, mode, severity, base_R):
    """The measurement model one member of the bank believes in.

    A bias makes the sensor read high, so the filter that assumes it should
    predict a reading that is high by the same amount -- then its innovation
    is near zero when it is right. Noise inflation leaves the reading centred
    and widens it, so that hypothesis changes R and not the prediction. The
    two fault modes want opposite changes here for the same reason they want
    opposite responses from the learned model.
    """
    if channel is None:
        return expected_readings

    amount = severity * faults.REFERENCE[faults.CHANNELS[channel]]

    if mode == "bias":
        offset = np.zeros(3)
        offset[channel] = amount

        def measure(states):
            return expected_readings(states) + offset
        return measure

    inflated = base_R.copy()
    inflated[channel, channel] += amount ** 2

    def measure(states):
        readings = expected_readings(states)
        return readings, np.tile(inflated, (len(readings), 1, 1))
    return measure


def make_bank(base_R, severity=BANK_SEVERITY):
    """Healthy, plus one hypothesis per channel per mode. Seven filters."""
    bank = [("healthy", hypothesis_model(None, None, 0.0, base_R))]
    for channel, name in enumerate(faults.CHANNELS):
        for mode in MODES:
            label = "%s %s" % (name.replace("_encoder", ""),
                               mode.split("_")[0])
            bank.append((label,
                         hypothesis_model(channel, mode, severity, base_R)))
    return bank


def log_density(innovation, S):
    """How plausible this innovation is under this filter's own claim.

    In logs because the product of a thousand small likelihoods underflows
    long before the run ends.
    """
    _, logdet = np.linalg.slogdet(S)
    quadratic = innovation @ np.linalg.solve(S, innovation)
    return -0.5 * (quadratic + logdet + len(innovation) * np.log(2 * np.pi))


class MMAE:
    """A bank of filters and a posterior over which one is right.

    Presents the same `run` as UKF so an experiment can use either, but it is
    a filter and not a measurement model -- it cannot be passed as `measure`
    to a UKF, because the thing it replaces is the UKF.
    """

    def __init__(self, Q, R, severity=BANK_SEVERITY, floor=WEIGHT_FLOOR):
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.bank = make_bank(self.R, severity)
        self.labels = [label for label, _ in self.bank]
        self.floor = floor
        self.filters = [UKF(self.Q, self.R, measure=m) for _, m in self.bank]
        self.weights = np.full(len(self.bank), 1.0 / len(self.bank))

    def reset(self):
        self.weights = np.full(len(self.bank), 1.0 / len(self.bank))

    def run(self, readings, start_mean, start_cov, dt):
        """Filter a run, returning what UKF.run returns.

        Every hypothesis starts from the same estimate and they diverge from
        there, each conditioned on its own assumption about the sensors.
        """
        n_steps, n_channels = readings.shape
        n_states = len(start_mean)
        n_hyp = len(self.bank)

        means = np.zeros((n_steps, n_states))
        covs = np.zeros((n_steps, n_states, n_states))
        innovations = np.zeros((n_steps, n_channels))
        innovation_covs = np.zeros((n_steps, n_channels, n_channels))
        self.weight_trace = np.zeros((n_steps, n_hyp))

        m = [np.asarray(start_mean, dtype=float).copy() for _ in range(n_hyp)]
        P = [np.asarray(start_cov, dtype=float).copy() for _ in range(n_hyp)]
        w = self.weights.copy()

        for k in range(n_steps):
            nus, Ss, logs = [], [], np.zeros(n_hyp)
            for i, one in enumerate(self.filters):
                m[i], P[i] = one.predict(m[i], P[i], dt)
                m[i], P[i], nu, S = one.update(m[i], P[i], readings[k])
                nus.append(nu)
                Ss.append(S)
                logs[i] = log_density(nu, S)

            # Bayes in logs, then back. Subtracting the largest before
            # exponentiating is what keeps this from underflowing to all
            # zeros once the run is long enough for the weights to separate.
            log_w = np.log(w) + logs
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()
            w = np.maximum(w, self.floor)
            w /= w.sum()

            # The mixture's first two moments. A weighted average of the
            # means, and a covariance that carries both each filter's own
            # spread and how far the filters disagree with each other -- the
            # second term is the part that says "the bank is not sure yet".
            mean = (w[:, None] * np.array(m)).sum(axis=0)
            cov = np.zeros((n_states, n_states))
            for i in range(n_hyp):
                offset = m[i] - mean
                cov = cov + w[i] * (P[i] + np.outer(offset, offset))

            nu_mix = (w[:, None] * np.array(nus)).sum(axis=0)
            S_mix = np.zeros((n_channels, n_channels))
            for i in range(n_hyp):
                offset = nus[i] - nu_mix
                S_mix = S_mix + w[i] * (Ss[i] + np.outer(offset, offset))

            means[k], covs[k] = mean, cov
            innovations[k], innovation_covs[k] = nu_mix, S_mix
            self.weight_trace[k] = w

        self.weights = w
        return means, covs, innovations, innovation_covs


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import Q, P0, best_constant_R
    from faults import apply_fault
    from sensors import read_sensors
    from trajectories import DT, random_run

    R_const = best_constant_R()
    SEEDS = range(500, 504)
    SEVERITIES = [0.5, 1.5, 3.0]

    def score(bank, severity):
        """Speed error on a left-encoder bias, and which hypothesis won."""
        errors, picked = [], []
        for seed in SEEDS:
            run = random_run(seed, duration=20.0)
            meas = read_sensors(run, seed=seed, dt=DT)
            if severity > 0:
                meas = apply_fault(meas, "left_encoder", "bias", severity,
                                   seed=seed, dt=DT)
            readings = np.column_stack([meas["left_encoder"],
                                        meas["right_encoder"], meas["gyro"]])
            start = np.zeros(7)
            start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                         run["speed"][0], run["turn_rate"][0]]

            bank.reset()
            means, _, _, _ = bank.run(readings, start, P0, DT)
            errors.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))

            # Which hypothesis the bank settled on, over the second half.
            picked.append(bank.weight_trace[len(readings) // 2:].mean(axis=0))
        return float(np.mean(errors)), np.mean(picked, axis=0)

    def plain(severity):
        """The same runs through an ordinary filter, as the reference."""
        errors = []
        for seed in SEEDS:
            run = random_run(seed, duration=20.0)
            meas = read_sensors(run, seed=seed, dt=DT)
            if severity > 0:
                meas = apply_fault(meas, "left_encoder", "bias", severity,
                                   seed=seed, dt=DT)
            readings = np.column_stack([meas["left_encoder"],
                                        meas["right_encoder"], meas["gyro"]])
            start = np.zeros(7)
            start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                         run["speed"][0], run["turn_rate"][0]]
            means, _, _, _ = UKF(Q, R_const).run(readings, start, P0, DT)
            errors.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
        return float(np.mean(errors))

    bank = MMAE(Q, R_const)

    print("A BANK OF %d HYPOTHESES, PLACED AT SEVERITY %.1f\n"
          % (len(bank.bank), BANK_SEVERITY))
    print("Left encoder bias. The bank sits at one severity; the tests are")
    print("below it, on it, and above it. Speed error in m/s.\n")
    print("  %-10s %14s %14s %10s" % ("severity", "analytic", "MMAE", "change"))
    print("  " + "-" * 52)

    for severity in [0.0] + SEVERITIES:
        reference = plain(severity)
        got, _ = score(bank, severity)
        print("  %-10.1f %14.4f %14.4f %9.0f%%"
              % (severity, reference, got,
                 100.0 * (got / reference - 1.0)))

    print("  " + "-" * 52)
    print("\n  A bank matched to the fault should do well at 1.5 and less")
    print("  well either side of it. If it is flat across the row, the")
    print("  severity mismatch costs nothing and the argument for a")
    print("  continuous health state is weaker than claimed.")

    print("\n\nDOES IT NAME THE RIGHT SENSOR?\n")
    print("Weights over the second half of each run, left encoder biased.\n")
    print("  %-10s" % "severity"
          + "".join("%14s" % l[:13] for l in bank.labels))
    print("  " + "-" * (10 + 14 * len(bank.labels)))
    for severity in SEVERITIES:
        _, weights = score(bank, severity)
        print("  %-10.1f" % severity
              + "".join("%14.3f" % v for v in weights))
    print("  " + "-" * (10 + 14 * len(bank.labels)))
    print("\n  The 'left bias' column should dominate. Weight landing on the")
    print("  healthy hypothesis means the fault went unnoticed; weight on the")
    print("  wrong channel means it was noticed and misattributed, which is")
    print("  the failure the health formulation has to beat.")

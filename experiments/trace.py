"""
What the filter's internals do, step by step, when a fault arrives.

Every other experiment here reports one number per run. That is the right
unit for comparing arms and the wrong one for seeing a mechanism work: an
average cannot show a health estimate climbing to meet a bias, or the
covariance-matching term staying flat while it does, or the opposite
happening under a noise fault. This records those quantities at every step.

Setup
-----

The left encoder runs healthy for ONSET seconds, then degrades to severity 3
and stays there, over runs of DURATION seconds -- twice the bakeoff's,
because a health estimate for a fault that arrives mid-run takes longer than
twenty seconds to settle. Two scenarios, one per fault family:

    bias             first moment: the reading shifts
    noise_inflation  second moment: the reading spreads

Two arms: the analytic model with the best constant R, which has no way to
respond, and layered, the arm that is deployed. For layered the state
carries the left encoder's two health entries, and the measurement model
carries the two covariance terms, so four internals are logged:

    bias_left     health estimate, first-moment entry
    noise_left    health estimate, second-moment entry
    r_ale_left    the learned covariance for that channel at the estimate
    r_unm_left    the covariance-matching residual for that channel

Eight seeds, the bakeoff's, every one kept. Logged at 10 Hz rather than the
filter's 50 to keep the file small; every quantity here moves on a timescale
of seconds.

    python experiments/trace.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from common import P0, Q, best_constant_R, load_arm, make_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from faults import apply_fault
from trajectories import DT
from ukf import UKF, nis as nis_of

SEEDS = range(2000, 2008)
DURATION = 40.0        # twice the bakeoff's runs: health takes time to settle
ONSET = 8.0            # seconds of healthy running before the fault
SEVERITY = 3.0
CHANNEL = "left_encoder"
SCENARIOS = ["bias", "noise_inflation"]
EVERY = 5              # log one step in five: 10 Hz

BIAS_LEFT, NOISE_LEFT = 7, 10


def one_run(name, arm, seed, mode):
    """Filter one run with the fault arriving at ONSET. One row per logged step."""
    run, meas, truth = make_run(seed, duration=DURATION)
    n = len(meas[CHANNEL])
    t = np.arange(n) * DT
    profile = np.where(t >= ONSET, SEVERITY, 0.0)
    meas = apply_fault(meas, CHANNEL, mode, profile, seed=seed, dt=DT)
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])

    if hasattr(arm, "reset"):
        arm.reset()
    n_states = getattr(arm, "n_states", 7)
    if n_states > 7:
        Q_use, P_use = arm.filter_settings(Q, P0)
        start = np.zeros(n_states)
        start[:7] = truth[0, :7]
    else:
        Q_use, P_use, start = Q, P0, truth[0].copy()

    means, covs, innov, S = UKF(Q_use, best_constant_R(), measure=arm).run(
        readings, start, P_use, DT)
    nis = nis_of(innov, S)

    rows = {"scenario": mode, "arm": name, "seed": seed,
            "t": t, "severity": profile,
            "speed_error": means[:, 3] - truth[:, 3],
            "speed_sd": np.sqrt(covs[:, 3, 3]), "nis": nis}

    if n_states > 7:
        # The learned covariance at the estimate, and the residual term as it
        # stood after each step's update.
        _, R_model = arm.base(means)
        rows["bias_left"] = means[:, BIAS_LEFT]
        rows["noise_left"] = means[:, NOISE_LEFT]
        rows["r_ale_left"] = R_model[:, 0, 0]
        rows["r_unm_left"] = np.array(arm.trace)[:, 0]

    frame = pd.DataFrame(rows)
    return frame.iloc[::EVERY]


def main():
    arms = [("analytic + best const R", load_arm("fixed")),
            ("layered", load_arm("layered"))]

    print("A FAULT ARRIVING MID-RUN, STEP BY STEP\n")
    print("Left encoder healthy for %.0f s, then severity %.1f, in %.0f s"
          " runs. %d seeds.\n" % (ONSET, SEVERITY, DURATION, len(SEEDS)))

    frames = []
    for mode in SCENARIOS:
        for name, arm in arms:
            for seed in SEEDS:
                frames.append(one_run(name, arm, seed, mode))
            print("  %-16s %-26s done" % (mode, name))
    frame = pd.concat(frames, ignore_index=True)

    # The mechanism in four numbers per scenario: layered's internals over the
    # last five seconds, where the fault has had time to be absorbed.
    late = frame[(frame["arm"] == "layered") & (frame["t"] >= frame["t"].max() - 5)]
    print("\nlayered, last 5 s, median over seeds and steps:\n")
    print("  %-16s %10s %10s %12s %12s"
          % ("scenario", "bias_left", "noise_left", "r_ale_left", "r_unm_left"))
    for mode in SCENARIOS:
        part = late[late["scenario"] == mode]
        print("  %-16s %10.3f %10.3f %12.5f %12.5f"
              % (mode, part["bias_left"].median(), part["noise_left"].median(),
                 part["r_ale_left"].median(), part["r_unm_left"].median()))
    print("\n  Under bias, the first-moment entry should carry the fault and")
    print("  the residual stay small. Under noise, the reverse: the health")
    print("  entry cannot see a fault that does not move the mean, and the")
    print("  residual has to.")

    path = ROOT / "results" / "trace.csv"
    frame.to_csv(path, index=False, float_format="%.5g")
    print("\nWrote %d rows to %s" % (len(frame), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

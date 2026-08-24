"""
Every arm with a sensor going bad.

THE QUESTION

When a sensor degrades, the filter should stop trusting it. "Trust" here is
literally R: the bigger R is for a channel, the less that channel moves the
estimate. So the test is simple to state -- as a sensor gets noisier, does
the arm's claimed noise for that channel follow the real one?

An arm that tracks it stays consistent and keeps estimating well. An arm that
does not carries on trusting a broken sensor, and the damage shows up in NIS
climbing far above 3.

WHAT COUNTS AS PASSING

The claimed spread should stay near the true spread as severity rises. That
ratio is the column to read. NIS is the supporting evidence: an arm that
keeps its claimed noise honest keeps NIS near 3 as well, because NIS is
exactly the question "was I as surprised as I said I would be".

A NOTE ON WHAT THIS CAN AND CANNOT SHOW

The learned arms take the STATE as their input. A fault is added to the
READING. So for a given state the learned models return the same answer
whether the sensor is fine or ruined -- the fault can only reach them
second-hand, through the filter's state estimate drifting. Do not expect
them to respond, and read a flat row as a fact about where the fault was
injected rather than a fact about the model.

That is not a defect in this script. It is the reason the state has to carry
a health variable, and this experiment is what shows it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import (ARMS, LABELS, SLOW, NIS_DOF, R_TUNED, available,
                    filter_once, load_arm, make_run, stack)
from faults import REFERENCE, apply_fault

CHANNELS = ["left_encoder", "right_encoder", "gyro"]

SEVERITIES = [0.0, 0.5, 1.0, 2.0, 4.0]
N_RUNS = 8
SLOW_RUNS = 2


def true_spread(channel, mode, severity):
    """What the channel's real spread becomes at this severity.

    Only defined for noise_inflation, where extra noise adds in quadrature
    with the noise already there. The other modes move the mean rather than
    the spread, so there is no single number to compare against and this
    returns None.
    """
    if mode != "noise_inflation":
        return None
    return REFERENCE[channel] * np.sqrt(1.0 + severity ** 2)


def claimed_spread(measure, states, channel):
    """What the arm says that channel's spread is, averaged over the run.

    An arm supplying its own covariance is asked directly. An arm without one
    is using the hand-tuned constant, so that is what it claims.
    """
    index = CHANNELS.index(channel)
    out = measure(states)
    if isinstance(out, tuple):
        _, R = out
        return float(np.sqrt(R[:, index, index]).mean())
    return float(np.sqrt(R_TUNED[index, index]))


def sweep(name, channel, mode, severities, n_runs):
    """One arm, one fault, every severity."""
    measure = load_arm(name)
    rows = []

    for severity in severities:
        claimed, nis_all, speed = [], [], []

        for seed in range(n_runs):
            run, meas, truth = make_run(seed)
            broken = apply_fault(meas, channel, mode, severity,
                                 seed=1000 + seed)

            result = filter_once(measure, broken, truth)
            nis_all.append(result["nis"])
            speed.append(result["speed_rmse"])

            # Ask the arm what it believed, at the states the filter actually
            # visited. For an adaptive arm this is its adapted R; for a
            # learned arm it is R at the estimated state.
            claimed.append(claimed_spread(measure, result["means"], channel))

        rows.append({
            "severity": severity,
            "claimed": float(np.mean(claimed)),
            "nis": float(np.concatenate(nis_all).mean()),
            "speed": float(np.mean(speed)),
        })
    return rows


def main():
    ready, missing = available()
    channel, mode = "left_encoder", "noise_inflation"

    print("DEGRADATION -- %s, %s" % (channel, mode))
    print("The one mode a model predicting only a mean cannot express:")
    print("the reading stays centred correctly and only gets noisier.\n")

    truth_row = [true_spread(channel, mode, s) for s in SEVERITIES]
    print("%-22s %-8s" % ("", "") + "".join("%9.1f" % s for s in SEVERITIES))
    print("%-22s %-8s" % ("true spread", "") + "".join("%9.4f" % t for t in truth_row))
    print("-" * 78)

    for name in ARMS:
        if name not in ready:
            continue
        n = SLOW_RUNS if name in SLOW else N_RUNS
        rows = sweep(name, channel, mode, SEVERITIES, n)

        print("%-22s %-8s" % (LABELS[name], "claimed")
              + "".join("%9.4f" % r["claimed"] for r in rows))
        print("%-22s %-8s" % ("", "ratio")
              + "".join("%9.2f" % (r["claimed"] / t)
                        for r, t in zip(rows, truth_row)))
        print("%-22s %-8s" % ("", "NIS")
              + "".join("%9.2f" % r["nis"] for r in rows))
        print()

    print("-" * 78)
    print("ratio near 1.00 across the row means the arm tracked the fault.")
    print("NIS near %.1f across the row means the filter stayed honest." % NIS_DOF)

    # ---- does the ensemble's model doubt notice? ----
    if "ensemble" in ready:
        measure = load_arm("ensemble")
        print("\n\nDOES MODEL DOUBT RESPOND?  (ensemble only)")
        print("Disagreement between the five members, on the same sweep.\n")
        print("%-22s" % "severity" + "".join("%9.1f" % s for s in SEVERITIES))

        aleatoric_row, epistemic_row = [], []
        for severity in SEVERITIES:
            run, meas, truth = make_run(0)
            broken = apply_fault(meas, channel, mode, severity, seed=1000)
            result = filter_once(measure, broken, truth)
            parts = measure.decompose(result["means"])
            aleatoric_row.append(parts["aleatoric"][:, 0].mean())
            epistemic_row.append(parts["epistemic"][:, 0].mean())

        print("%-22s" % "sensor noise" + "".join("%9.5f" % a for a in aleatoric_row))
        print("%-22s" % "model doubt" + "".join("%9.5f" % e for e in epistemic_row))
        print("%-22s" % "doubt, vs healthy"
              + "".join("%8.2fx" % (e / epistemic_row[0]) for e in epistemic_row))

    # ---- other fault modes, one severity ----
    print("\n\nOTHER FAULT MODES  (%s, severity 2.0)\n" % channel)
    modes = ["bias", "drift", "scale_error", "stuck", "dropout"]
    print("%-22s" % "" + "".join("%12s" % m for m in modes))
    print("-" * 78)

    for name in ARMS:
        if name not in ready or name in SLOW:
            continue
        measure = load_arm(name)
        cells = []
        for mode_name in modes:
            rows = sweep(name, channel, mode_name, [2.0], N_RUNS)
            cells.append(rows[0]["nis"])
        print("%-22s" % LABELS[name] + "".join("%12.2f" % c for c in cells))

    print("-" * 78)
    print("NIS, where %d is correct. These are the filter noticing, which is" % NIS_DOF)
    print("not the same as any arm knowing which sensor to blame.")


if __name__ == "__main__":
    main()

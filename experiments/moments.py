"""
The thesis, stated as a prediction about faults nobody trained on.

WHAT IS BEING CLAIMED

A Kalman update is a first-moment operation. It multiplies the innovation by
a gain and adds, so it reads which way the innovation points and is deaf to
anything that changes only its size. Covariance matching is a second-moment
operation: it squares the innovation and averages, which destroys the sign, so
it reads size and is deaf to direction.

Now sort the faults the same way. A bias moves the mean of the measurement
distribution and leaves the spread alone -- a first-moment perturbation. Noise
inflation moves the spread and leaves the mean exactly where it was -- second
moment. So the fault taxonomy and the method taxonomy are the same taxonomy,
indexed by moment order, and the pairing between them is forced rather than
lucky.

If that is right it is not a statement about bias and noise inflation. It is a
statement about any fault, including ones the model has never been trained on,
because moment order is a property of what the fault does to the readings and
not of what appeared in the training set.

THE PREDICTION, WRITTEN DOWN BEFORE THE RUN

    bias              first moment    trained on. health should win.
    drift             first moment    NOT trained on -- a bias that grows
                                      during the run. health should still win
                                      if moment order is what matters.
    scale_error       first moment    NOT trained on -- a bias proportional to
                                      the signal. same expectation.
    noise_inflation   second moment   trained on. covariance matching wins,
                                      and health cannot move at all: dh/dm is
                                      zero so the update has nothing to
                                      multiply.
    stuck             neither         NOT trained on. the reading freezes at a
                                      plausible value. it does not shift the
                                      mean predictably nor widen the spread,
                                      so neither mechanism has a signal.
    dropout           neither         NOT trained on. same, by a different
                                      route -- samples are lost and the last
                                      is held.

WHAT EACH OUTCOME WOULD MEAN

If drift and scale_error pattern with bias despite never appearing in
training, the taxonomy is doing real work: it predicted transfer to unseen
faults from structure alone, and the claim is about moment order.

If they do not, the claim is narrower -- something closer to "the two modes we
happened to train on" -- and the paper has to say so. That would be a worse
result and a more honest one, and it is better found here than in review.

If stuck and dropout defeat everything alike, that is the boundary of the
whole approach and belongs in the paper as a limitation rather than being left
for a reviewer to find.

WHY ONLY FOUR ARMS

The learned arms that take only the vehicle state -- plain, resnet, gp, bhr,
ensemble -- cannot respond to any fault at all, whatever its moment order,
because the fault arrives in the measurement and their inputs do not move.
They would add five identical rows. The question here is about mechanisms, so
the arms are the ones that have one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from bakeoff import RESULTS, SEEDS, load_arm, score
from common import LABELS

ROOT = Path(__file__).resolve().parents[1]

ARMS = ["fixed", "adaptive", "health", "combined"]

# Severity means different things to different modes. bias, drift,
# noise_inflation and scale_error take an amount, measured in multiples of the
# channel's own healthy spread. stuck and dropout take a fraction of the run,
# so 3.0 is not a severity they can express and asking for one would silently
# clamp. Each mode gets a ladder it can actually walk.
LADDERS = {
    "bias": [0.5, 1.5, 3.0],
    "drift": [0.5, 1.5, 3.0],
    "scale_error": [0.5, 1.5, 3.0],
    "noise_inflation": [0.5, 1.5, 3.0],
    "stuck": [0.1, 0.3, 0.5],
    "dropout": [0.1, 0.3, 0.5],
}

MOMENT = {
    "bias": "first", "drift": "first", "scale_error": "first",
    "noise_inflation": "second", "stuck": "neither", "dropout": "neither",
}

TRAINED = {"bias", "noise_inflation"}

# Which arm the taxonomy says should win each mode, written here rather than
# inferred from the results afterwards.
EXPECTED = {"first": "health-conditioned", "second": "adaptive R (Mehra)",
            "neither": "-- none --"}


def main():
    arms = []
    for name in ARMS:
        try:
            arms.append((LABELS[name], load_arm(name)))
        except Exception as problem:
            print("  %s unavailable: %s" % (name, str(problem)[:60]))
    if len(arms) < 2:
        print("Need at least two arms; train them first.")
        return 1

    seeds = list(SEEDS)
    print("MOMENT ORDER PREDICTS WHICH MECHANISM WINS\n")
    print("%d runs of 20 s, seeds %d-%d, fault on the left encoder."
          % (len(seeds), min(seeds), max(seeds)))
    print("Two of these six modes were in training. The other four were not,")
    print("and the claim is that moment order predicts them anyway.\n")

    records = []
    healthy = {}
    for label, arm in arms:
        means, _ = score(arm, seeds, None, 0.0)
        healthy[label] = means[0]
        records.append({"arm": label, "mode": "none", "moment": "none",
                        "trained": True, "severity": 0.0,
                        "speed_rmse": means[0], "nis": means[1],
                        "nees": means[2]})

    for mode, ladder in LADDERS.items():
        for severity in ladder:
            for label, arm in arms:
                means, _ = score(arm, seeds, mode, severity)
                records.append({"arm": label, "mode": mode,
                                "moment": MOMENT[mode],
                                "trained": mode in TRAINED,
                                "severity": severity,
                                "speed_rmse": means[0], "nis": means[1],
                                "nees": means[2]})
        print("  %-18s done" % mode)

    frame = pd.DataFrame(records)

    print("\n\nSPEED ERROR AT THE TOP OF EACH LADDER\n")
    print("  %-16s %8s %9s" % ("mode", "moment", "trained")
          + "".join("%22s" % label for label, _ in arms))
    print("  " + "-" * (35 + 22 * len(arms)))

    for mode, ladder in LADDERS.items():
        worst = frame[(frame["mode"] == mode)
                      & (frame["severity"] == ladder[-1])]
        row = "  %-16s %8s %9s" % (mode, MOMENT[mode],
                                   "yes" if mode in TRAINED else "no")
        for label, _ in arms:
            value = worst[worst["arm"] == label]["speed_rmse"].iloc[0]
            row += "%22.4f" % value
        print(row)
    print("  " + "-" * (35 + 22 * len(arms)))

    print("\n\nDID THE PREDICTION HOLD?\n")
    print("  %-16s %9s %24s %24s %8s"
          % ("mode", "moment", "predicted winner", "actual winner", "ok"))
    print("  " + "-" * 86)

    hits, testable = 0, 0
    for mode, ladder in LADDERS.items():
        worst = frame[(frame["mode"] == mode)
                      & (frame["severity"] == ladder[-1])]
        # Combined carries both mechanisms, so it cannot adjudicate between
        # them. The question is which single-mechanism arm wins.
        single = worst[worst["arm"].isin(["health-conditioned",
                                          "adaptive R (Mehra)"])]
        if single.empty:
            continue
        actual = single.loc[single["speed_rmse"].idxmin(), "arm"]
        predicted = EXPECTED[MOMENT[mode]]
        if MOMENT[mode] == "neither":
            mark = "n/a"
        else:
            testable += 1
            ok = actual == predicted
            hits += ok
            mark = "yes" if ok else "NO"
        print("  %-16s %9s %24s %24s %8s"
              % (mode, MOMENT[mode], predicted, actual, mark))
    print("  " + "-" * 86)
    print("\n  %d of %d testable modes went the way moment order says."
          % (hits, testable))
    print("  Two of them were in training and are not evidence of transfer.")
    print("  The ones to read are drift and scale_error, which the model has")
    print("  never seen in any form.")

    print("\n\nCOST OF EACH FAULT, RELATIVE TO THAT ARM'S OWN HEALTHY ERROR\n")
    print("  %-16s %9s" % ("mode", "moment")
          + "".join("%22s" % label for label, _ in arms))
    print("  " + "-" * (26 + 22 * len(arms)))
    for mode, ladder in LADDERS.items():
        worst = frame[(frame["mode"] == mode)
                      & (frame["severity"] == ladder[-1])]
        row = "  %-16s %9s" % (mode, MOMENT[mode])
        for label, _ in arms:
            value = worst[worst["arm"] == label]["speed_rmse"].iloc[0]
            row += "%21.1fx" % (value / healthy[label])
        print(row)
    print("  " + "-" * (26 + 22 * len(arms)))
    print("\n  Ratios rather than absolutes, because the arms do not start")
    print("  from the same healthy error -- health pays about half again for")
    print("  carrying six extra states, and a raw comparison charges it twice.")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "moments.csv"
    frame.to_csv(path, index=False, float_format="%.6g")
    print("\nWrote %d rows to %s" % (len(frame), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

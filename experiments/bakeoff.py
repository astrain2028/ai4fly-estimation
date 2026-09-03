"""
Every arm, one set of runs, one ladder of severities.

WHY THIS EXISTS

common.py was written so that no experiment could choose its own filter
settings, because before it existed one arm was using R = 0.15 while the rest
used 0.1805 and the numbers could not honestly go in the same table. It fixed
Q, R, P0 and the metrics.

It did not fix the seeds or the conditions, and every experiment went on
choosing those for itself. So the results now read:

    redundancy.py       seeds 1000-1007, bias at severity 2.0
    doubt/measurement   seeds 700-703, one severity per mode
    mmae/measurement    seeds 500-503, severities 0.5 / 1.5 / 3.0
    the README table    seeds 0-19, healthy only

Which means the ranking anyone would draw from those numbers is stitched
together from four different populations. That is exactly the failure common.py
was built to prevent, one level up, and stitched comparisons are how a win
gets quoted that was never measured.

This file is the single table. Same seeds, same severities, same channel,
every arm, one run each.

WHAT IS VARIED AND WHAT IS NOT

The fault is always on the left encoder. Which channel breaks is a separate
question and redundancy.py answers it; holding it fixed here keeps the
severity ladder readable and the runtime finite.

Bias and noise inflation only. They are the two modes the learned arms were
trained on, and between them they define the complementarity split: bias moves
the expected reading and noise inflation does not, so one is visible to a
model reading innovation direction and the other only to something reading
magnitude. Faults outside the training set are heldout.py's subject, not this
one's.

WHAT TO READ OFF IT

No arm should win everywhere. The analytic model must win when healthy,
because the simulator generates readings from the equations that model uses,
so it is not an approximation of the truth but the truth itself. If a learned
arm beats it there, something is wrong with the experiment rather than
impressive about the arm.

Down the severity ladder the ordering should invert, and where it inverts is
the result. An arm that only ever ties is not earning its compute.

COST

Eleven arms times seven conditions times the seeds. The Gaussian process and
the ensemble are minutes per condition, and the MMAE bank runs seven filters
per step. Expect the better part of an hour for everything, so arms can be
named on the command line to run a subset:

    python experiments/bakeoff.py                 everything
    python experiments/bakeoff.py fixed health    just those two
    python experiments/bakeoff.py --quick         fewer seeds, fewer severities
    python experiments/bakeoff.py --channel=gyro  break a different sensor

WHERE THE NUMBERS GO

Every run is written to results/bakeoff.csv, one row per arm per condition
per seed, before anything is averaged. Printed tables are a summary of that
file and not the record itself.

The reason is that the pooled means printed below cannot support a paired
comparison. Two arms differing by 5 per cent on average may differ that way
on every single run, or on one run out of eight, and those are different
claims. Keeping the per-seed rows leaves that question answerable later
without re-running anything -- which matters when the full sweep is the
better part of an hour.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from common import (ARMS, LABELS, NEES_STATES, NIS_DOF, P0, Q,
                    best_constant_R, load_arm, make_run)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

from faults import apply_fault
from trajectories import DT
from ukf import UKF, nis as nis_of, nees as nees_of

SEEDS = range(2000, 2008)
SEVERITIES = [0.5, 1.5, 3.0]
MODES = ["bias", "noise_inflation"]

# Which sensor breaks. The left encoder by default, but worth pointing at the
# gyro too: its healthy spread is 0.00799 against the encoders' 0.1805, and
# redundancy.py found that losing it alone costs the filter almost nothing.
# Whether the complementarity split survives on a channel the filter barely
# leans on is a different question from whether it holds on one it does.
CHANNEL = "left_encoder"

QUICK_SEEDS = range(2000, 2004)
QUICK_SEVERITIES = [1.5]

RESULTS = ROOT / "results"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conditions(severities):
    """Healthy first, then each mode down the ladder."""
    out = [("healthy", None, 0.0)]
    for mode in MODES:
        for severity in severities:
            out.append(("%s %.1f" % (mode.split("_")[0], severity),
                        mode, severity))
    return out


def one_run(arm, seed, mode, severity, channel=CHANNEL):
    """Filter a single run and return speed error, NIS and NEES.

    Two kinds of arm arrive here. Nine of them are measurement models, which
    are callable and get wrapped in a UKF. MMAE is a bank of filters rather
    than a measurement model -- it replaces the UKF instead of plugging into
    one -- so it is dispatched on whether the arm is callable at all.
    """
    run, meas, truth = make_run(seed)
    if mode is not None and severity > 0:
        meas = apply_fault(meas, channel, mode, severity, seed=seed, dt=DT)

    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    if hasattr(arm, "reset"):
        arm.reset()

    n_states = getattr(arm, "n_states", 7)
    if n_states > 7:
        Q_use, P_use = arm.filter_settings(Q, P0)
        start = np.zeros(n_states)
        start[:7] = truth[0, :7]
        truth = np.column_stack(
            [truth[:, :7], np.zeros((len(truth), n_states - 7))])
    else:
        Q_use, P_use, start = Q, P0, truth[0].copy()

    R = best_constant_R()
    if callable(arm):
        means, covs, innov, S = UKF(Q_use, R, measure=arm).run(
            readings, start, P_use, DT)
    else:
        means, covs, innov, S = arm.run(readings, start, P_use, DT)

    return (float(np.sqrt(np.mean((means[:, 3] - truth[:, 3]) ** 2))),
            float(nis_of(innov, S).mean()),
            float(nees_of(means, covs, truth, states=NEES_STATES).mean()))


def score(arm, seeds, mode, severity, channel=CHANNEL):
    """Every seed, and the average over them. Every arm sees the same ones."""
    rows = [one_run(arm, seed, mode, severity, channel) for seed in seeds]
    means = tuple(float(np.mean([r[i] for r in rows])) for i in range(3))
    return means, rows


def collect(names, seeds, severities, channel=CHANNEL):
    """Load what is available and score it, saying what is missing and why.

    Returns the pooled means for printing and every individual run for the
    csv, because a mean cannot be un-averaged later.
    """
    entries, missing = [], []

    for name in names:
        if name == "mmae":
            continue
        try:
            entries.append((LABELS.get(name, name), load_arm(name)))
        except Exception as problem:
            missing.append((name, "%s: %s" % (type(problem).__name__,
                                              str(problem)[:60])))

    if "mmae" in names:
        try:
            module = _load(ROOT / "models" / "mmae" / "measurement.py",
                           "mmae_for_bakeoff")
            entries.append(("MMAE bank of %d" % 7,
                            module.MMAE(Q, best_constant_R())))
        except Exception as problem:
            missing.append(("mmae", "%s: %s" % (type(problem).__name__,
                                                str(problem)[:60])))

    results, records = {}, []
    for label, arm in entries:
        print("  running %-26s" % label, end="", flush=True)
        pooled = []
        for name, mode, severity in conditions(severities):
            means, rows = score(arm, seeds, mode, severity, channel)
            pooled.append(means)
            for seed, (speed, nis, nees) in zip(seeds, rows):
                records.append({"arm": label, "condition": name,
                                "mode": mode or "none", "severity": severity,
                                "channel": channel, "seed": seed,
                                "speed_rmse": speed, "nis": nis, "nees": nees})
        results[label] = pooled
        print("done")
    return results, missing, records


def table(results, severities, index, title, note):
    """One metric, arms down the side and conditions across."""
    heads = [label for label, _, _ in conditions(severities)]
    print("\n\n%s\n" % title)
    print("  %-26s" % "" + "".join("%13s" % h for h in heads))
    print("  " + "-" * (26 + 13 * len(heads)))
    for label, rows in results.items():
        print("  %-26s" % label
              + "".join("%13.4f" % row[index] for row in rows))
    print("  " + "-" * (26 + 13 * len(heads)))
    print("\n  %s" % note)


def main():
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv

    channel = CHANNEL
    for arg in sys.argv[1:]:
        if arg.startswith("--channel="):
            channel = arg.split("=", 1)[1]

    seeds = QUICK_SEEDS if quick else SEEDS
    severities = QUICK_SEVERITIES if quick else SEVERITIES
    names = words or (ARMS + ["mmae"])

    print("EVERY ARM, THE SAME RUNS\n")
    print("%d runs of 20 s, seeds %d-%d, fault on the %s."
          % (len(list(seeds)), min(seeds), max(seeds), channel))
    print("Severities %s. Every arm gets the same best constant R; those"
          % ", ".join("%.1f" % s for s in severities))
    print("carrying their own covariance override it.\n")

    results, missing, records = collect(names, seeds, severities, channel)

    if missing:
        print("\n  not available:")
        for name, why in missing:
            print("    %-12s %s" % (name, why))
        print("  (learned arms need their weights; those are not in the")
        print("   repository, so train them or expect them here.)")

    if not results:
        print("\nNothing ran.")
        return 1

    table(results, severities, 0, "SPEED ERROR, m/s",
          "The analytic model should win the healthy column and lose down "
          "the\n  ladder. Where the ordering inverts is the whole result.")

    table(results, severities, 1, "NIS -- want %d" % NIS_DOF,
          "Whether an arm's innovations are as big as it claimed. An arm "
          "that\n  stays near target under a fault is one that noticed the "
          "fault.")

    table(results, severities, 2, "NEES on speed and turn rate -- want 2",
          "Whether the estimate is as close to the truth as it claims. "
          "Needs\n  the true state, so it exists in simulation and not on "
          "hardware.")

    print("\n\nBEST ARM PER CONDITION\n")
    heads = [label for label, _, _ in conditions(severities)]
    for i, head in enumerate(heads):
        best = min(results.items(), key=lambda kv: kv[1][i][0])
        print("  %-14s %-28s %.4f" % (head, best[0], best[1][i][0]))
    print("\n  If one arm wins every row, either it is genuinely better or")
    print("  the conditions are too narrow to separate them. Neither has")
    print("  been true here so far.")

    # One row per arm per condition per seed, written before averaging. A
    # partial sweep goes to its own file rather than overwriting the full one.
    RESULTS.mkdir(exist_ok=True)
    stem = "bakeoff" if channel == CHANNEL else "bakeoff_%s" % channel
    if quick or words:
        stem += "_partial"
    path = RESULTS / ("%s.csv" % stem)
    pd.DataFrame(records).to_csv(path, index=False, float_format="%.6g")
    print("\nWrote %d rows to %s" % (len(records), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
What the epistemic term buys layered, on the ground robot.

layered sums three covariances: the learned noise, the model's doubt about
its own prediction, and a covariance-matching residual for whatever neither
explains. Every published layered number so far ran without the second. This
runs the same arm with and without it, on the same seeds, across every fault
the robot has been tested on: the bakeoff's ladder for bias and noise, and
the top of the moment-order ladder for the four held-out modes.

What should happen
------------------

The doubt measures how unfamiliar the model's inputs are, and a sensor fault
does not change the vehicle state. What it can see is the health entries,
which the filter estimates and feeds back and which training bounds at 3.0.
So it should stay near its training level wherever the fault is one the
model was taught -- bias, noise -- and rise on the faults it was not, where
the filter moves health somewhere unfamiliar. The health and doubt arms drive
health to about 9 on scale_error; in layered the residual absorbs most of
that fault first, so health moves less, and whether the doubt still notices
is part of the question.

Two readings of the term are recorded. The share columns say how much it
contributed to R: the doubt as a fraction of the learned noise, on the
faulted channel, averaged over steps. The novelty columns say how unusual
the input was, as a multiple of its average over the training rows -- the
number the vehicle reports in flight, whether or not it moves R.

    python experiments/epistemic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import bakeoff
from common import load_arm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loader import load_module

layered_module = load_module(ROOT / "models" / "layered" / "measurement.py",
                             "layered_for_epistemic")

SEEDS = bakeoff.SEEDS
CONDITIONS = ([("healthy", "none", 0.0)]
              + [("%s %.1f" % (m.split("_")[0], s), m, s)
                 for m in ("bias", "noise_inflation") for s in (0.5, 1.5, 3.0)]
              + [("drift 3.0", "drift", 3.0), ("scale_error 3.0", "scale_error", 3.0),
                 ("stuck 0.5", "stuck", 0.5), ("dropout 0.5", "dropout", 0.5)])
FAULTED = 0          # the left encoder, bakeoff.CHANNEL


def arm(with_epistemic):
    """layered, carrying what load_arm attaches for the wider filter."""
    measure = layered_module.load_measurement_model(
        with_epistemic=with_epistemic)
    template = load_arm("layered")
    measure.n_states = template.n_states
    measure.filter_settings = template.filter_settings
    return measure


def shares(measure):
    """What each term contributed over one run, and how novel the input was.

    Returns the doubt and residual as fractions of the learned noise on the
    faulted channel, the doubt's share over all channels, and the novelty
    ratio -- mean and peak over the run, highest channel -- that the vehicle
    would report.
    """
    parts = measure.parts
    ale = np.array([p[0] for p in parts])
    epi = np.array([p[1] for p in parts])
    unm = np.array([p[2] for p in parts])
    novelty = np.array([p[3] for p in parts]).max(axis=1)
    return (float(np.mean(epi[:, FAULTED] / ale[:, FAULTED])),
            float(np.mean(epi / ale)),
            float(np.mean(unm[:, FAULTED] / ale[:, FAULTED])),
            float(novelty.mean()), float(novelty.max()))


def main():
    arms = [("layered", arm(False)), ("layered + epistemic", arm(True))]
    print("THE EPISTEMIC TERM IN LAYERED, GROUND ROBOT\n")
    print("%d seeds, left encoder faulted, the same runs for both arms.\n"
          % len(SEEDS))

    records = []
    for label, measure in arms:
        for name, mode, severity in CONDITIONS:
            for seed in SEEDS:
                speed, nis, nees = bakeoff.one_run(
                    measure, seed, None if mode == "none" else mode, severity)
                epi_f, epi_all, unm_f, nov_mean, nov_peak = shares(measure)
                records.append({"arm": label, "condition": name, "mode": mode,
                                "severity": severity, "seed": seed,
                                "speed_rmse": speed, "nis": nis, "nees": nees,
                                "epi_share": epi_f, "epi_share_all": epi_all,
                                "unm_share": unm_f, "novelty_mean": nov_mean,
                                "novelty_peak": nov_peak})
            print("  %-20s %-16s done" % (label, name))

    frame = pd.DataFrame(records)
    table = frame.groupby(["condition", "arm"], sort=False)[
        ["speed_rmse", "nis", "epi_share", "unm_share", "novelty_mean",
         "novelty_peak"]].mean().unstack("arm")

    print("\n\n  %-16s %9s %9s %6s %6s %8s %9s %8s %7s"
          % ("", "error", "+ epi", "NIS", "+ epi", "doubt", "residual",
             "novelty", "peak"))
    print("  " + "-" * 86)
    for name, _, _ in CONDITIONS:
        row = table.loc[name]
        on = "layered + epistemic"
        print("  %-16s %9.4f %9.4f %6.2f %6.2f %7.2f%% %8.0f%% %8.2f %7.2f"
              % (name, row[("speed_rmse", "layered")],
                 row[("speed_rmse", on)], row[("nis", "layered")],
                 row[("nis", on)], 100 * row[("epi_share", on)],
                 100 * row[("unm_share", on)], row[("novelty_mean", on)],
                 row[("novelty_peak", on)]))
    print("  " + "-" * 86)
    print("\n  doubt and residual: as a share of the learned noise on the")
    print("  faulted channel. novelty: the input's unfamiliarity to the model,")
    print("  1 on training-like inputs, mean and peak over the run, highest")
    print("  channel. The paired change per seed is in the csv.")

    path = ROOT / "results" / "epistemic.csv"
    frame.to_csv(path, index=False, float_format="%.6g")
    print("\nWrote %d rows to %s" % (len(frame), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

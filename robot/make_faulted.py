"""
Builds a dataset in which sensors are sometimes degraded, and says by how
much.

WHY A SEPARATE FILE

make_dataset.py produces a healthy robot. Everything in this project so far
has been trained on it, which is why no learned model responds to a fault:
the models take the state as their input, faults arrive in the measurement,
and a state the model has seen a thousand times looks the same whether the
sensor reporting it is fine or ruined.

Fixing that means putting sensor health into the state, and that means having
training data where health varies and is labelled. This file makes it.

WHAT A ROW LOOKS LIKE

Every row carries three new columns -- severity_left, severity_right,
severity_gyro -- each a continuous number, zero for a healthy channel and
rising with degradation. A severity of 1 is trouble about the size of that
channel's own healthy noise.

Continuous rather than a healthy/faulty flag, deliberately. A Gaussian filter
carries means and covariances, so a discrete mode is not something it can
represent without a mixture and the cost that implies. A continuous level is
something it can carry natively, and it admits partial degradation, which is
what most real faults look like before they become total.

WHICH MODES, AND WHY BOTH

Two kinds of fault appear here, and they are different in a way that matters
more than it first appears.

    bias              shifts what the sensor reads
    noise_inflation   leaves the reading centred correctly and adds spread

A filter estimates health through the covariance between its sample points'
health and their predicted readings. That covariance is only non-zero if the
predicted reading actually changes with health -- which is true for a bias
and false for pure noise inflation, where the expected reading is unchanged
by construction.

So the two modes are expected to behave differently, and both are included so
that the difference can be measured rather than assumed. If health turns out
to be estimable for one and not the other, that is a property of the
formulation worth reporting, not a bug to hide.

HOW RUNS ARE ASSIGNED

A third of runs are healthy throughout. The rest get one faulted channel, one
mode, and a severity drawn uniformly. Only one channel is faulted at a time:
with three sensors constraining two quantities there is exactly one spare, so
a single failure leaves the remaining two able to disagree with it. Two
simultaneous failures do not, and learning to separate them needs training
examples of every pair, which is a combinatorial problem for later.

Severity is constant within a run rather than growing during it. That keeps
the label unambiguous -- one number describes the whole run -- and detecting a
fault that appears mid-run is a different experiment, which degradation.py
already covers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import faults
import make_dataset
import sensors
from trajectories import DT, random_run

N_RUNS = 150
DURATION = 20.0

CHANNELS = ["left_encoder", "right_encoder", "gyro"]
SEVERITY_COLUMNS = ["severity_left", "severity_right", "severity_gyro"]

MODES = ["bias", "noise_inflation"]
HEALTHY_FRACTION = 1.0 / 3.0
MAX_SEVERITY = 3.0

COLUMNS = make_dataset.COLUMNS + SEVERITY_COLUMNS + ["fault_mode"]


def assign(run_id, rng):
    """Decide what is wrong with this run, if anything.

    Returns the channel index, the mode, and the severity. A healthy run is
    reported as channel None so that nothing downstream has to check a
    severity against zero to find out.
    """
    if rng.random() < HEALTHY_FRACTION:
        return None, "none", 0.0
    channel = int(rng.integers(len(CHANNELS)))
    mode = MODES[int(rng.integers(len(MODES)))]
    # Uniform on severity rather than on log severity: the interesting region
    # is the one where a fault is comparable to the noise, not the one where
    # it swamps everything.
    severity = float(rng.uniform(0.25, MAX_SEVERITY))
    return channel, mode, severity


def build(n_runs=N_RUNS, duration=DURATION, seed=0, **simulator):
    """Generate the faulted dataset as a DataFrame.

    Extra keyword arguments are passed to the healthy generator, so noise
    growth, encoder resolution, and vehicle calibration error can be varied
    here too.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for run_id in range(n_runs):
        channel, mode, severity = assign(run_id, rng)

        # One run of the healthy simulator, then the fault applied on top.
        # Doing it in that order means a faulted run and its healthy twin
        # differ by the fault alone, which is what makes them comparable.
        one = make_dataset.build(n_runs=1, duration=duration,
                                 first_run=run_id, **simulator)

        severities = [0.0, 0.0, 0.0]
        if channel is not None:
            name = CHANNELS[channel]
            readings = {c: one[c].values for c in CHANNELS}
            broken = faults.apply_fault(readings, name, mode, severity,
                                        seed=10_000 + run_id, dt=DT)
            one[name] = broken[name]
            severities[channel] = severity

        for column, value in zip(SEVERITY_COLUMNS, severities):
            one[column] = value
        one["fault_mode"] = mode
        frames.append(one)

    return pd.concat(frames, ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "robot_faulted.csv"

    print("Building a dataset with degraded sensors")
    df = build()
    print("  %d rows, %d runs" % (len(df), df["run"].nunique()))

    per_run = df.groupby("run").first()
    healthy = (per_run[SEVERITY_COLUMNS].sum(axis=1) == 0).sum()
    print("\n  %d healthy runs, %d faulted" % (healthy, len(per_run) - healthy))

    print("\n  %-18s %8s %14s" % ("", "runs", "mean severity"))
    for mode in ["none"] + MODES:
        rows = per_run[per_run["fault_mode"] == mode]
        worst = rows[SEVERITY_COLUMNS].max(axis=1)
        print("  %-18s %8d %14.2f"
              % (mode, len(rows), worst.mean() if len(rows) else 0.0))

    print("\n  %-18s %8s" % ("faulted channel", "runs"))
    for column in SEVERITY_COLUMNS:
        print("  %-18s %8d" % (column, (per_run[column] > 0).sum()))

    print("\nDoes the label describe what actually happened?\n")
    print("  %-18s %10s %12s %12s"
          % ("", "severity", "mean shift", "extra spread"))
    for mode in MODES:
        rows = per_run[per_run["fault_mode"] == mode]
        rows = rows[rows["severity_left"] > 0]
        if not len(rows):
            continue
        run_id = rows.index[0]
        one = df[df["run"] == run_id]
        error = one["left_encoder"].values - one["left_spin"].values
        clean = make_dataset.build(n_runs=1, duration=DURATION,
                                   first_run=run_id)
        base = clean["left_encoder"].values - clean["left_spin"].values
        print("  %-18s %10.2f %12.4f %12.4f"
              % (mode, rows.loc[run_id, "severity_left"],
                 error.mean() - base.mean(), error.std() - base.std()))

    print("\n  bias moves the first column and not the second.")
    print("  noise_inflation moves the second and not the first.")
    print("  A filter updates health from the first. That is the whole")
    print("  question this dataset exists to settle.")

    df.to_csv(out, index=False, float_format="%.6g")
    print("\nSaved %s" % out.name)

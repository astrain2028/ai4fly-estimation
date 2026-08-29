"""
Builds a dataset in which sensors are sometimes degraded, and says how.

WHY A SEPARATE FILE

make_dataset.py produces a healthy robot. Everything trained on it is blind to
faults: the models take the state as their input, faults arrive in the
measurement, and a state the model has seen a thousand times looks the same
whether the sensor reporting it is fine or ruined. Putting sensor health into
the state needs training data where health varies and is labelled, which is
what this makes.

TWO NUMBERS PER SENSOR, NOT ONE

An earlier version of this file gave each sensor a single degradation level
and a label saying which kind of fault it was. That does not work, and the
reason is worth stating because it is a property of the formulation rather
than of the code.

The two fault modes want opposite responses from a measurement model:

    bias              shifts the reading      -> the model should shift h
    noise_inflation   leaves the reading      -> the model should inflate R
                      centred and adds spread    and leave h alone

A model given one severity number cannot tell which response is called for,
so it learns the average of the two. Measured: it corrected 57 per cent of a
bias it should have corrected fully, and applied a mean shift of 0.134 to
noise-only faults where the correct shift is zero. The filter then relied on a
ruined sensor 16 per cent less when it should have been something nearer a
factor of three.

So each sensor now carries two levels, and the model can move h for one and R
for the other without either compromising the other.

WHY SOME RUNS HAVE BOTH AT ONCE

If a bias level were only ever non-zero when the noise level was zero, the two
inputs would be almost perfectly anti-correlated across the training set, and
a model could fit it without ever learning that they mean different things.
Roughly a third of faulted runs therefore carry both kinds on the same
channel, at independently drawn levels, purely so the two columns vary
independently.

HOW RUNS ARE ASSIGNED

A third of runs are healthy. The rest degrade one channel, with each mode
independently present or absent. Only one channel at a time: three sensors
constrain two quantities, so there is exactly one spare, and a single failure
leaves the other two able to disagree with it. Two failed channels do not, and
learning to separate them needs examples of every pair.

WHEN THE FAULT ARRIVES

Severity used to be constant within a run, which kept the label unambiguous
and left a hole. A model trained that way has seen sensors that were already
broken at the first sample and sensors that were fine at the last, and never
one that changed. Asked to estimate health on a sensor that fails at ten
seconds, it was 94 per cent worse than on the same fault present from the
start -- against 20 per cent for covariance matching, which has no training
set to be outside of.

That was read at the time as the health state's process noise being tuned too
tight, and it is partly that. But no setting of a process noise teaches a
model what a transition looks like. The data has to contain one.

So half of faulted runs now develop their fault partway through, either as a
step -- fine, then broken -- or as a ramp that worsens over the rest of the
run. The severity columns are per sample rather than per run, which the
training format already allowed: every row is one (state, health) pair mapped
to a reading, and nothing required the health part to be the same on every
row of a run.

Both modes on a channel share one onset time. A single physical failure
degrades a sensor in whatever ways it degrades it at once, and drawing two
independent onsets would be inventing a coincidence to learn from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import faults
import make_dataset
from trajectories import DT

N_RUNS = 150
DURATION = 20.0

CHANNELS = ["left_encoder", "right_encoder", "gyro"]
SHORT = ["left", "right", "gyro"]
MODES = ["bias", "noise_inflation"]

# severity_bias_left, ..., severity_noise_gyro
SEVERITY_COLUMNS = ["severity_%s_%s" % (m.split("_")[0], s)
                    for m in MODES for s in SHORT]

HEALTHY_FRACTION = 1.0 / 3.0
MODE_PRESENT = 0.6           # chance each mode appears on a faulted channel
MAX_SEVERITY = 3.0

# How often a faulted run starts healthy and degrades, rather than being
# broken throughout. Half, because both regimes have to stay well represented:
# a model trained only on transitions would be no better off than one trained
# only on constants, just wrong at the other end.
ONSET_FRACTION = 0.5

# When the fault arrives, as a fraction of the run. Bounded away from both
# ends so that every onset run carries some healthy samples and some faulted
# ones -- an onset at 0.98 is a healthy run with a mislabelled tail.
ONSET_WINDOW = (0.2, 0.8)

SHAPES = ["step", "ramp"]

COLUMNS = make_dataset.COLUMNS + SEVERITY_COLUMNS


def assign(rng):
    """Decide what is wrong with this run, and when it went wrong.

    Returns the channel index, a peak severity per mode, and either None for
    a fault present throughout or a (shape, when) pair for one that arrives.
    All zero and None for a healthy run.
    """
    if rng.random() < HEALTHY_FRACTION:
        return None, {mode: 0.0 for mode in MODES}, None

    channel = int(rng.integers(len(CHANNELS)))
    onset = None
    if rng.random() < ONSET_FRACTION:
        onset = (SHAPES[int(rng.integers(len(SHAPES)))],
                 float(rng.uniform(*ONSET_WINDOW)))

    while True:
        levels = {mode: (float(rng.uniform(0.25, MAX_SEVERITY))
                         if rng.random() < MODE_PRESENT else 0.0)
                  for mode in MODES}
        if any(levels.values()):      # a faulted run must have some fault
            return channel, levels, onset


def severity_profile(level, onset, n):
    """The severity of one mode at each sample of a run.

    A step is the harder case and the one the onset experiment measures: the
    sensor is fine and then it is not. A ramp is the commoner one physically,
    a sensor going gradually out of calibration, and it is also the only one
    of the two where the intermediate severities are ever visited -- which is
    what the continuous parameterisation is for.
    """
    if onset is None:
        return np.full(n, float(level))

    shape, when = onset
    profile = np.zeros(n)
    start = int(when * n)
    if shape == "step":
        profile[start:] = level
    else:
        profile[start:] = level * np.linspace(0.0, 1.0, n - start)
    return profile


def build(n_runs=N_RUNS, duration=DURATION, seed=0, **simulator):
    """Generate the faulted dataset as a DataFrame.

    Extra keyword arguments pass through to the healthy generator, so noise
    growth, encoder resolution and vehicle calibration error can be varied
    here too.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for run_id in range(n_runs):
        channel, levels, onset = assign(rng)

        # A healthy run first, then the faults on top, so a faulted run and
        # its healthy twin differ by the fault alone.
        one = make_dataset.build(n_runs=1, duration=duration,
                                 first_run=run_id, **simulator)

        severities = {column: np.zeros(len(one)) for column in SEVERITY_COLUMNS}
        if channel is not None:
            name = CHANNELS[channel]
            for offset, mode in enumerate(MODES):
                if levels[mode] <= 0:
                    continue
                # The severity is now a value per sample, which the fault
                # functions take directly: the reading at each moment is
                # degraded by however bad the sensor is at that moment.
                profile = severity_profile(levels[mode], onset, len(one))
                readings = {c: one[c].values for c in CHANNELS}
                broken = faults.apply_fault(
                    readings, name, mode, profile,
                    seed=10_000 + run_id + 1000 * offset, dt=DT)
                one[name] = broken[name]
                severities["severity_%s_%s"
                           % (mode.split("_")[0], SHORT[channel])] = profile

        for column, value in severities.items():
            one[column] = value
        frames.append(one)

    return pd.concat(frames, ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "robot_faulted.csv"

    # How many runs, from the command line. The health model carries six
    # health levels on eleven inputs, so it is markedly hungrier than the
    # arms that take five, and the default may not be enough for it.
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS

    print("Building a dataset with degraded sensors, %d runs" % n_runs)
    df = build(n_runs=n_runs)
    print("  %d rows, %d severity columns"
          % (len(df), len(SEVERITY_COLUMNS)))

    # Peak rather than first: a run whose fault arrives at ten seconds reads
    # as perfectly healthy on its first row, and summarising by that would
    # report a third of the faulted runs as healthy.
    per_run = df.groupby("run")[SEVERITY_COLUMNS].max()
    bias_cols = [c for c in SEVERITY_COLUMNS if "bias" in c]
    noise_cols = [c for c in SEVERITY_COLUMNS if "noise" in c]
    has_bias = per_run[bias_cols].sum(axis=1) > 0
    has_noise = per_run[noise_cols].sum(axis=1) > 0

    print("\n  %-22s %8s" % ("condition", "runs"))
    print("  %-22s %8d" % ("healthy", (~has_bias & ~has_noise).sum()))
    print("  %-22s %8d" % ("bias only", (has_bias & ~has_noise).sum()))
    print("  %-22s %8d" % ("noise only", (~has_bias & has_noise).sum()))
    print("  %-22s %8d" % ("both on one channel", (has_bias & has_noise).sum()))

    # Onset is not stored as a column -- it is visible in the data as a run
    # that starts clean and does not end that way.
    starts = df.groupby("run")[SEVERITY_COLUMNS].first().sum(axis=1)
    peaks = per_run.sum(axis=1)
    faulted = peaks > 0
    late = faulted & (starts == 0)
    print("\n  %-22s %8d" % ("faulted from sample 0", (faulted & ~late).sum()))
    print("  %-22s %8d" % ("faulted partway in", late.sum()))
    print("  %-22s %8.1f%%"
          % ("rows actually faulted",
             100.0 * (df[SEVERITY_COLUMNS].sum(axis=1) > 0).mean()))

    print("\nAre the two columns independent enough to be separable?\n")
    left = per_run[["severity_bias_left", "severity_noise_left"]]
    print("  correlation between bias and noise level on the left encoder:"
          " %+.3f" % left.corr().iloc[0, 1])
    print("  (near zero is what the 'both' runs are for; strongly negative")
    print("   would mean a model could fit one by reading the other)")

    print("\nDoes each label describe what it claims?\n")
    print("  %-22s %10s %12s %12s"
          % ("", "severity", "mean shift", "extra spread"))
    for mode, column in [("bias", "severity_bias_left"),
                         ("noise_inflation", "severity_noise_left")]:
        other = ("severity_noise_left" if mode == "bias"
                 else "severity_bias_left")
        pure = per_run[(per_run[column] > 0) & (per_run[other] == 0)]
        if not len(pure):
            continue
        run_id = pure.index[0]
        one = df[df["run"] == run_id]
        clean = make_dataset.build(n_runs=1, duration=DURATION,
                                   first_run=run_id)

        # Only the faulted samples. On an onset run the healthy first half
        # would otherwise halve every effect and make both labels look weak.
        active = one[column].values > 0
        error = (one["left_encoder"].values - one["left_spin"].values)[active]
        base = (clean["left_encoder"].values
                - clean["left_spin"].values)[active]
        print("  %-22s %10.2f %12.4f %12.4f"
              % (mode, one[column].values[active].mean(),
                 error.mean() - base.mean(), error.std() - base.std()))

    print("\n  bias moves the first column and not the second.")
    print("  noise_inflation moves the second and not the first.")
    print("  Two labels, two effects, and now a model can tell them apart.")

    df.to_csv(out, index=False, float_format="%.6g")
    print("\nSaved %s" % out.name)

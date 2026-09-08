"""
Training data: flights with labelled sensor health.

WHAT A ROW IS

One sample: the six true states, the nine readings, and six numbers saying how
degraded each device was at that moment. The model learns the map from the
first and third to the second.

    roll pitch yaw p q r | accel_x..mag_z | severity_bias_accel .. severity_noise_mag

Health is labelled per device and per mode, which is two numbers per device
rather than one. That split was forced on the ground robot by measurement: a
single severity per sensor made the model learn the average of two opposite
responses, correcting 57 per cent of a bias it should have corrected fully and
applying a spurious shift to noise-only faults. Two numbers let it move the
predicted reading for one and the predicted spread for the other.

SEVERITY VARIES WITHIN A RUN

Half of faulted runs develop their fault partway through rather than carrying
it from the first sample. Without those examples a model has only ever seen
devices that were already broken when the flight began, and asked to track one
that fails mid-flight it does badly -- on the ground robot, 94 per cent worse
than the same fault present from the start, which retraining with onsets
brought to 58.

Both modes on a device share one onset time. A single physical failure
degrades a part in whatever ways it degrades it at once.

ONE DEVICE AT A TIME

Nine measurements constrain six states, so three are spare and a single failed
device leaves the other two able to disagree with it. Two failed devices do
not, and learning to separate them would need examples of every pair. The
ground robot found the same cliff with less room: three sensors, two
quantities, exactly one spare.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import faults
from faults import DEVICES, REFERENCE
from sensors import CHANNELS, read_sensors
from trajectories import DT, DURATION, random_run, truth_matrix

STATES = ["roll", "pitch", "yaw", "p", "q", "r"]
MODES = ["bias", "noise_inflation"]
SHORT = list(DEVICES)

SEVERITY_COLUMNS = ["severity_%s_%s" % (m.split("_")[0], d)
                    for m in MODES for d in SHORT]

COLUMNS = ["run"] + STATES + CHANNELS + SEVERITY_COLUMNS

N_RUNS = 400
HEALTHY_FRACTION = 1.0 / 3.0
MODE_PRESENT = 0.6
MAX_SEVERITY = 3.0

ONSET_FRACTION = 0.5
ONSET_WINDOW = (0.2, 0.8)
SHAPES = ["step", "ramp"]


def assign(rng):
    """What is wrong with this run, and when it went wrong."""
    if rng.random() < HEALTHY_FRACTION:
        return None, {mode: 0.0 for mode in MODES}, None

    device = SHORT[int(rng.integers(len(SHORT)))]
    onset = None
    if rng.random() < ONSET_FRACTION:
        onset = (SHAPES[int(rng.integers(len(SHAPES)))],
                 float(rng.uniform(*ONSET_WINDOW)))

    while True:
        levels = {mode: (float(rng.uniform(0.25, MAX_SEVERITY))
                         if rng.random() < MODE_PRESENT else 0.0)
                  for mode in MODES}
        if any(levels.values()):
            return device, levels, onset


def severity_profile(level, onset, n):
    """The severity of one mode at each sample."""
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


def build(n_runs=N_RUNS, duration=DURATION, seed=0):
    """Generate the dataset as a DataFrame."""
    rng = np.random.default_rng(seed)
    frames = []

    for run_id in range(n_runs):
        device, levels, onset = assign(rng)

        run = random_run(run_id, duration=duration)
        meas = read_sensors(run, seed=run_id)
        n = len(run["roll"])

        severities = {column: np.zeros(n) for column in SEVERITY_COLUMNS}
        if device is not None:
            for offset, mode in enumerate(MODES):
                if levels[mode] <= 0:
                    continue
                profile = severity_profile(levels[mode], onset, n)
                meas = faults.apply_fault(meas, device, mode, profile,
                                          seed=50_000 + run_id + 1000 * offset,
                                          dt=DT)
                severities["severity_%s_%s"
                           % (mode.split("_")[0], device)] = profile

        frame = pd.DataFrame({"run": np.full(n, run_id)})
        for i, name in enumerate(STATES):
            frame[name] = truth_matrix(run)[:, i]
        for name in CHANNELS:
            frame[name] = meas[name]
        for column, value in severities.items():
            frame[column] = value
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)

    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else N_RUNS
    print("Building %d flights with degraded sensors\n" % n_runs)
    df = build(n_runs=n_runs)
    print("  %d rows, %d severity columns" % (len(df), len(SEVERITY_COLUMNS)))

    peak = df.groupby("run")[SEVERITY_COLUMNS].max()
    bias_cols = [c for c in SEVERITY_COLUMNS if "bias" in c]
    noise_cols = [c for c in SEVERITY_COLUMNS if "noise" in c]
    has_bias = peak[bias_cols].sum(axis=1) > 0
    has_noise = peak[noise_cols].sum(axis=1) > 0

    print("\n  %-24s %8s" % ("condition", "runs"))
    print("  %-24s %8d" % ("healthy", (~has_bias & ~has_noise).sum()))
    print("  %-24s %8d" % ("bias only", (has_bias & ~has_noise).sum()))
    print("  %-24s %8d" % ("noise only", (~has_bias & has_noise).sum()))
    print("  %-24s %8d" % ("both on one device", (has_bias & has_noise).sum()))

    starts = df.groupby("run")[SEVERITY_COLUMNS].first().sum(axis=1)
    faulted = peak.sum(axis=1) > 0
    late = faulted & (starts == 0)
    print("\n  %-24s %8d" % ("faulted from sample 0", (faulted & ~late).sum()))
    print("  %-24s %8d" % ("faulted partway in", late.sum()))
    print("  %-24s %7.1f%%" % ("rows actually faulted",
                               100 * (df[SEVERITY_COLUMNS].sum(axis=1) > 0).mean()))

    print("\n\nWhich device broke, and how often?\n")
    for device in SHORT:
        cols = [c for c in SEVERITY_COLUMNS if c.endswith(device)]
        print("  %-8s %d runs" % (device, (peak[cols].sum(axis=1) > 0).sum()))

    print("\n\nDoes each label describe what it claims?\n")
    print("  %-22s %10s %12s %14s"
          % ("", "severity", "mean shift", "extra spread"))
    for mode, device in [("bias", "accel"), ("noise_inflation", "accel")]:
        column = "severity_%s_%s" % (mode.split("_")[0], device)
        other = ("severity_noise_accel" if mode == "bias"
                 else "severity_bias_accel")
        pure = peak[(peak[column] > 0) & (peak[other] == 0)]
        if not len(pure):
            continue
        run_id = int(pure.index[0])
        one = df[df["run"] == run_id]
        active = one[column].values > 0

        clean_run = random_run(run_id)
        clean = read_sensors(clean_run, seed=run_id)
        cols = [CHANNELS[c] for c in DEVICES[device]]
        got = one[cols].values[active]
        was = np.column_stack([clean[c] for c in cols])[active]
        print("  %-22s %10.2f %12.4f %14.4f"
              % (mode, one[column].values[active].mean(),
                 (got - was).mean(), (got - was).std()))

    path = out / "quad_faulted.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print("\nSaved %s" % path.relative_to(Path(__file__).parent))

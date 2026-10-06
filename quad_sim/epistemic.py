"""
What the epistemic term buys layered, on the quadcopter.

The companion to experiments/epistemic.py, on the vehicle where it has the
most to see. The term's two readings are recorded: its share of R, and the
novelty ratio -- how unusual the input is to the model, 1 on training-like
inputs -- which the vehicle reports in flight.

Conditions
----------

The bakeoff's ladder for bias and noise, the two modes the model was trained
on, with the accelerometer and then the magnetometer degraded. Then the four
modes it was never shown, at the top of their ladders. Then aggressive
flight: healthy sensors, but body rates at twice the training envelope's
peak -- the kind of novelty a real vehicle brings that a fault does not, since
it moves the vehicle states rather than the health entries.

Same eight flights for every condition, layered with and without the term.

    python quad_sim/epistemic.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from loader import load_module

bakeoff = load_module(HERE / "bakeoff.py", "quad_bakeoff_for_epistemic")
quad = bakeoff.quad

LADDER = ([("healthy", None, 0.0)]
          + [("%s %.1f" % (m.split("_")[0], s), m, s)
             for m in ("bias", "noise_inflation") for s in (0.5, 1.5, 3.0)])
UNTRAINED = [("drift 3.0", "drift", 3.0), ("scale_error 3.0", "scale_error", 3.0),
             ("stuck 0.5", "stuck", 0.5), ("dropout 0.5", "dropout", 0.5)]
DEVICES = ["accel", "mag"]
DEVICE_CHANNELS = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}

# How much harder than the training flights the aggressive ones are flown.
AGGRESSION = 2.0
TRAINING_RATE_SCALE = bakeoff.random_run.__globals__["RATE_SCALE"]


def summary(measure):
    """Share of R, and novelty per device: mean and peak over the run."""
    ale = np.array([p[0] for p in measure.parts])
    epi = np.array([p[1] for p in measure.parts])
    unm = np.array([p[2] for p in measure.parts])
    novelty = np.array([p[3] for p in measure.parts])
    out = {"epi_share": float(np.mean(epi / ale)),
           "unm_share": float(np.mean(unm / ale))}
    for device, columns in DEVICE_CHANNELS.items():
        per_step = novelty[:, columns].max(axis=1)
        out["novelty_%s" % device] = float(per_step.mean())
        out["novelty_peak_%s" % device] = float(per_step.max())
    return out


def conditions():
    """(device, label, mode, severity, rate scale) for every run set."""
    out = []
    for device in DEVICES:
        for name, mode, severity in LADDER + UNTRAINED:
            out.append((device, name, mode, severity, 1.0))
    out.append(("none", "aggressive x%g" % AGGRESSION, None, 0.0, AGGRESSION))
    return out


def main():
    arms = [("layered", quad.load_measurement_model(with_epistemic=False)),
            ("layered + epistemic",
             quad.load_measurement_model(with_epistemic=True))]
    print("THE EPISTEMIC TERM IN LAYERED, QUADCOPTER\n")

    records = []
    globals_ = bakeoff.random_run.__globals__
    for device, name, mode, severity, aggression in conditions():
        bakeoff.DEVICE = device if device != "none" else "accel"
        globals_["RATE_SCALE"] = TRAINING_RATE_SCALE * aggression
        try:
            for label, measure in arms:
                for seed in bakeoff.SEEDS:
                    error, nis = bakeoff.one_run(measure, seed, mode, severity,
                                                 True)
                    records.append(dict(device=device, arm=label,
                                        condition=name, mode=mode or "none",
                                        severity=severity, seed=seed,
                                        attitude_rmse_deg=error, nis=nis,
                                        **summary(measure)))
        finally:
            globals_["RATE_SCALE"] = TRAINING_RATE_SCALE
        print("  %-6s %-16s done" % (device, name))

    frame = pd.DataFrame(records)
    on = frame[frame["arm"] == "layered + epistemic"]
    off = frame[frame["arm"] == "layered"]
    keys = ["device", "condition", "seed"]
    paired = on.merge(off, on=keys, suffixes=("", "_off"))
    table = paired.groupby(["device", "condition"], sort=False).mean(
        numeric_only=True)

    print("\n\n  %-6s %-16s %8s %8s %6s %6s %7s %8s %8s %8s"
          % ("", "", "error", "+ epi", "NIS", "+ epi", "doubt", "accel",
             "gyro", "mag"))
    print("  " + "-" * 92)
    for (device, name), row in table.iterrows():
        print("  %-6s %-16s %8.3f %8.3f %6.2f %6.2f %6.2f%% %8.2f %8.2f %8.2f"
              % (device, name, row["attitude_rmse_deg_off"],
                 row["attitude_rmse_deg"], row["nis_off"], row["nis"],
                 100 * row["epi_share"], row["novelty_accel"],
                 row["novelty_gyro"], row["novelty_mag"]))
    print("  " + "-" * 92)
    print("\n  error: roll and pitch RMS, degrees. doubt: the term's share of")
    print("  the learned noise. accel, gyro, mag: mean novelty per device,")
    print("  1 on training-like inputs; peaks are in the csv.")

    path = ROOT / "results" / "quad_epistemic.csv"
    frame.to_csv(path, index=False, float_format="%.6g")
    print("\nWrote %d rows to %s" % (len(frame), path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

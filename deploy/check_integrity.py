"""
Does the integrity monitor declare the estimate invalid when every sensor
fails -- and only then?

deploy/integrity.py decides whether the estimate can be used at all, and
with --allow-rtl that decision asks the flight controller to bring the
vehicle home. A monitor that misses a total failure leaves the pilot flying
on a number that means nothing; one that fires on anything less sends the
vehicle home for no reason. Both are measured here, on the vehicle-side code
itself: deploy/runtime.py's estimator, the exported weights, and the monitor,
stepped over simulated quadcopter flights with the failure arriving halfway.

Flights
-------

Must go invalid, soon after the failure:

    every device frozen     stuck from 10 s, as a dead driver resending its
                            last sample looks
    every device dead       reading zero from 10 s
    every device garbage    NaN from 10 s
    stream lost             no readings at all from 10 s

Must never go invalid -- every alarm here is false:

    healthy                 the bakeoff's held-out seeds
    aggressive x1.5, x2     healthy sensors, body rates past the training peak
    one device failed       each device, frozen, dead or returning NaN, and
                            each of bias, noise and scale error at severity 3
    two devices failed      each pair, frozen or dead: the third still flies
    every device noisy      all three at ten times their noise, from 10 s:
                            still informative, so the filter's job

The suspect tier is reported alongside: how often the model-based signals
mark a device before anything has gone wrong, which is why they are kept
from the decision to go home. Its residual-ratio levels are set here first,
per device, at integrity.COLLAPSE_MARGIN times the highest any of twenty
healthy calibration flights reached, and written to
results/integrity_threshold.csv, which deploy/pi_main.py reads.

Written to results/integrity.csv.

    python deploy/check_integrity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

calib = load_module(HERE / "calibrate_novelty.py", "integrity_calib")
integrity = load_module(HERE / "integrity.py", "integrity_check")
pi_main = load_module(HERE / "pi_main.py", "integrity_pi_main")

ONSET = 10.0                     # seconds; flights are 20 s
HELD_OUT = range(400, 408)
CALIBRATION = range(500, 520)
DEVICES = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}
OUT = ROOT / "results" / "integrity.csv"
THRESHOLDS = ROOT / "results" / "integrity_threshold.csv"


def readings_for(seed, failures=(), aggression=1.0, lost=False):
    """One flight's readings, with each (device, mode, severity) applied from
    ONSET, and its true starting state. Returns a list with None for a gap.
    """
    trajectories, sensors, faults = calib.tools("quad")[3:]
    base = trajectories.RATE_SCALE
    trajectories.RATE_SCALE = base * aggression
    try:
        run = trajectories.random_run(seed)
    finally:
        trajectories.RATE_SCALE = base
    meas = sensors.read_sensors(run, seed=seed)
    n = len(meas[sensors.CHANNELS[0]])
    onset = int(ONSET / trajectories.DT)

    for device, mode, severity in failures:
        if mode == "stuck":
            meas = faults.apply_fault(meas, device, "stuck", 1 - onset / n,
                                      seed=seed)
        elif mode in faults.MODES:
            ramp = np.where(np.arange(n) >= onset, severity, 0.0)
            meas = faults.apply_fault(meas, device, mode, ramp, seed=seed)
        else:
            meas = dict(meas)
            for c in faults.DEVICES[device]:
                column = meas[faults.CHANNELS[c]].copy()
                column[onset:] = 0.0 if mode == "dead" else np.nan
                meas[faults.CHANNELS[c]] = column

    rows = list(sensors.stack(meas))
    if lost:
        rows[onset:] = [None] * (n - onset)
    return rows, trajectories.truth_matrix(run)[0], trajectories.DT


def fly(rows, start_state, dt, novelty_alert, window, collapse=None):
    """Step the vehicle's estimator and monitor; return per-step statuses."""
    runtime, _, parts = calib.tools("quad")[:3]
    (move, measure, Q, R, start, P0, _, health_states, _, _) = parts
    measure.reset()
    start = start.copy()
    start[:len(start_state)] = start_state
    estimator = runtime.Estimator(move, measure, Q, R, start, P0, dt,
                                  health_states=health_states)
    monitor = integrity.Integrity(DEVICES, dt=dt, health=integrity.QUAD_HEALTH,
                                  novelty_alert=novelty_alert,
                                  novelty_time_constant=window,
                                  collapse=collapse)
    out = []
    for k, reading in enumerate(rows):
        record = estimator.step(k * dt, reading)
        status = monitor.update(record, reading)
        out.append((status["state"], set(status["suspect"]),
                    dict(monitor.ratio)))
    return out


def summarise(label, group, flights, onset_step, dt):
    """Count the outcomes over one scenario's flights."""
    invalid_before = invalid_after = degraded_before = degraded_after = 0
    suspect_before = suspect_after = 0
    latencies = []
    for steps in flights:
        before, after = steps[:onset_step], steps[onset_step:]
        invalid_before += any(s == "invalid" for s, _, _ in before)
        degraded_before += any(s == "degraded" for s, _, _ in before)
        suspect_before += any(sus for _, sus, _ in before)
        suspect_after += any(sus for _, sus, _ in after)
        hits = [k for k, (s, _, _) in enumerate(after) if s == "invalid"]
        if hits:
            invalid_after += 1
            latencies.append(hits[0] * dt)
        degraded_after += any(s == "degraded" for s, _, _ in after)
    n = len(flights)
    return {"group": group, "scenario": label, "flights": n,
            "invalid_after_onset": invalid_after,
            "invalid_before_onset": invalid_before,
            "latency_median_s": np.median(latencies) if latencies else np.nan,
            "latency_max_s": max(latencies) if latencies else np.nan,
            "degraded_after_onset": degraded_after,
            "degraded_before_onset": degraded_before,
            "suspect_before_onset": suspect_before,
            "suspect_after_onset": suspect_after}


def main():
    novelty_alert, window = pi_main.novelty_alert("quadcopter")
    trajectories = calib.tools("quad")[3]
    dt = trajectories.DT
    onset_step = int(ONSET / dt)

    # The suspect tier's residual ratio, on healthy flights the check is not
    # scored on: how high it goes when nothing is wrong.
    print("Residual over predicted noise on %d healthy calibration flights"
          % len(CALIBRATION), flush=True)
    peaks = {name: 0.0 for name in DEVICES}
    for seed in CALIBRATION:
        rows, start, dt = readings_for(seed)
        steps = fly(rows, start, dt, novelty_alert, window)
        for _, _, ratio in steps[100:]:
            for name in DEVICES:
                peaks[name] = max(peaks[name], ratio[name])
    collapse = {name: integrity.COLLAPSE_MARGIN * peaks[name]
                for name in DEVICES}
    for name in DEVICES:
        print("  %-6s peak %7.1f   suspect above %.1f"
              % (name, peaks[name], collapse[name]))
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame([{"vehicle": "quad", "device": name,
                   "healthy_peak": peaks[name], "threshold": collapse[name]}
                  for name in DEVICES]).to_csv(THRESHOLDS, index=False)

    pairs = [("accel", "gyro"), ("accel", "mag"), ("gyro", "mag")]
    every = list(DEVICES)
    scenarios = [
        ("must fire", "every device frozen",
         dict(failures=[(d, "stuck", 0) for d in every])),
        ("must fire", "every device dead",
         dict(failures=[(d, "dead", 0) for d in every])),
        ("must fire", "every device garbage",
         dict(failures=[(d, "nan", 0) for d in every])),
        ("must fire", "stream lost", dict(lost=True)),
        ("must not fire", "healthy", {}),
        ("must not fire", "aggressive x1.5", dict(aggression=1.5)),
        ("must not fire", "aggressive x2", dict(aggression=2.0)),
        ("must not fire", "every device noisy x10",
         dict(failures=[(d, "noise_inflation", 10.0) for d in every])),
    ]
    for device in every:
        for mode, severity in [("stuck", 0), ("dead", 0), ("nan", 0),
                               ("bias", 3.0),
                               ("noise_inflation", 3.0),
                               ("scale_error", 3.0)]:
            scenarios.append(("must not fire", "%s %s" % (device, mode),
                              dict(failures=[(device, mode, severity)])))
    for pair in pairs:
        for mode in ("stuck", "dead"):
            scenarios.append(("must not fire", "%s+%s %s" % (pair + (mode,)),
                              dict(failures=[(d, mode, 0) for d in pair])))

    rows_out = []
    for group, label, kwargs in scenarios:
        flights = []
        for seed in HELD_OUT:
            rows, start, dt = readings_for(seed, **kwargs)
            flights.append(fly(rows, start, dt, novelty_alert, window,
                               collapse))
        row = summarise(label, group, flights, onset_step, dt)
        rows_out.append(row)
        print("  %-14s %-28s invalid %d/%d after, %d before; latency %s; "
              "degraded %d; suspect before %d"
              % (group, label, row["invalid_after_onset"], row["flights"],
                 row["invalid_before_onset"],
                 "-" if np.isnan(row["latency_median_s"])
                 else "%.2f-%.2f s" % (row["latency_median_s"],
                                       row["latency_max_s"]),
                 row["degraded_after_onset"], row["suspect_before_onset"]),
              flush=True)

    frame = pd.DataFrame(rows_out)
    for name in DEVICES:
        frame["healthy_ratio_peak_%s" % name] = peaks[name]
    OUT.parent.mkdir(exist_ok=True)
    frame.to_csv(OUT, index=False)

    fire = frame[frame.group == "must fire"]
    quiet = frame[frame.group == "must not fire"]
    caught = (fire.invalid_after_onset == fire.flights).all()
    false = int(quiet.invalid_after_onset.sum() + frame.invalid_before_onset.sum())
    print("\n  every total failure declared invalid: %s" % ("yes" if caught else "NO"))
    print("  false invalid on everything else: %d" % false)
    print("  slowest detection: %.2f s" % fire.latency_max_s.max())
    print("\n  Wrote %s and %s" % (OUT.relative_to(ROOT),
                                   THRESHOLDS.relative_to(ROOT)))
    return 0 if caught and false == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

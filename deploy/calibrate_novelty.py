"""
Where the in-flight novelty alert should sit, and what it would catch.

The epistemic term reports how unfamiliar the model's inputs are -- a ratio,
1 on training-like inputs -- and deploy/mavlink_sink.py smooths it per device
and raises an alert above a level. That level has to come from flights, not
from a guess: healthy simulated quadcopter flights already reach a smoothed
novelty of several, because the health estimates wander while the filter
settles and on some flights well after.

Method
------

The vehicle-side code itself: deploy/runtime.py's estimator in numpy, with
the exported weights and posterior, and the sink's own smoothing
(mavlink_sink.Novelty). Each flight is simulated once and its per-step
novelty recorded; each candidate smoothing window is then replayed over the
same recordings. For a window, twenty healthy calibration flights set each
device's level at MARGIN times the highest smoothed novelty any of them
reached after the first two seconds. Then, on flights the level was not set
on:

    healthy         the bakeoff's own seeds: every alert here is false
    aggressive      healthy sensors, body rates at 1.5 and 2 times the
                    training peak (quadcopter only): leaving the envelope
    faults          each fault mode at the top of its ladder

Why several windows: healthy flights pass briefly through states the
training data covers thinly, and there the model is honestly unsure -- one
healthy quadcopter flight reaches a one-second novelty of 35 while the median
is under 2. A level that clears such a flight is too high to catch much. A
longer window lets a brief excursion fade while a sustained one, like flying
outside the envelope, builds. The window flown is the one that catches the
most with no false alarm on the held-out healthy flights.

Written to results/novelty_threshold.csv, which deploy/pi_main.py reads, and
results/novelty_detection.csv, both with every window.

    python deploy/calibrate_novelty.py          both vehicles
    python deploy/calibrate_novelty.py quad     one
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

MARGIN = 1.5
ARM_AFTER = 100            # steps: the sink's own settling allowance
TIME_CONSTANTS = [1.0, 2.0, 5.0, 10.0]    # seconds of smoothing, compared
AGGRESSIONS = [1.5, 2.0]
CALIBRATION = {"quad": range(500, 520), "robot": range(3000, 3020)}
HELD_OUT = {"quad": range(400, 408), "robot": range(2000, 2008)}
MODES = [("bias", 3.0), ("noise_inflation", 3.0), ("drift", 3.0),
         ("scale_error", 3.0), ("stuck", 0.5), ("dropout", 0.5)]
TRAINED = {"bias", "noise_inflation"}

ROBOT_DEVICES = {"left": [0], "right": [1], "gyro": [2]}
THRESHOLDS = ROOT / "results" / "novelty_threshold.csv"
DETECTION = ROOT / "results" / "novelty_detection.csv"


_TOOLS = {}


def tools(vehicle):
    """The runtime, the sink, the estimator's parts, and the simulator.

    Loaded once per process and in this order. The runtime loads the robot's
    filter code, which imports the robot's dynamics by bare name; the
    quadcopter's simulator has modules of the same names, so they are
    imported only after, with the names cleared, as runtime.py's own
    self-test does.
    """
    if vehicle not in _TOOLS:
        runtime = load_module(HERE / "runtime.py", "calib_runtime")
        sink = load_module(HERE / "mavlink_sink.py", "calib_sink")
        build = runtime.quadcopter if vehicle == "quad" else runtime.ground_robot
        parts = build(with_epistemic=True)
        for name in ("trajectories", "sensors", "dynamics", "faults"):
            sys.modules.pop(name, None)
        folder = ROOT / ("quad_sim" if vehicle == "quad" else "robot")
        sys.path.insert(0, str(folder))
        import trajectories
        import sensors
        import faults
        _TOOLS[vehicle] = (runtime, sink, parts, trajectories, sensors, faults)
    return _TOOLS[vehicle]


def flight(vehicle, seed, mode=None, severity=0.0, device=None, aggression=1.0):
    """Readings and the starting state of one simulated flight."""
    trajectories, sensors, faults = tools(vehicle)[3:]
    if vehicle == "quad":
        base = trajectories.RATE_SCALE
        trajectories.RATE_SCALE = base * aggression
        try:
            run = trajectories.random_run(seed)
        finally:
            trajectories.RATE_SCALE = base
        meas = sensors.read_sensors(run, seed=seed)
        if mode is not None:
            meas = faults.apply_fault(meas, device, mode, severity, seed=seed,
                                      dt=trajectories.DT)
        return sensors.stack(meas), trajectories.truth_matrix(run)[0]
    run = trajectories.random_run(seed, duration=20.0)
    meas = sensors.read_sensors(run, seed=seed, dt=trajectories.DT)
    if mode is not None:
        meas = faults.apply_fault(meas, device, mode, severity, seed=seed,
                                  dt=trajectories.DT)
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    start = np.array([run["x"][0], run["y"][0], run["heading"][0],
                      run["speed"][0], run["turn_rate"][0]])
    return readings, start


def doubt_series(vehicle, readings, start_state):
    """The estimator's per-channel novelty at every step of one flight."""
    runtime, sink, parts = tools(vehicle)[:3]
    (move, measure, Q, R, start, P0, dt, health_states, names,
     channels) = parts
    measure.reset()
    start = start.copy()
    start[:len(start_state)] = start_state
    estimator = runtime.Estimator(move, measure, Q, R, start, P0, dt,
                                  health_states=health_states)
    return np.array([estimator.step(k * dt, readings[k])["doubt"]
                     for k in range(len(readings))]), dt


def smoothed(vehicle, series, dt, time_constant):
    """The sink's own smoothing replayed over a recorded flight.

    Per device, the level at every step after the settling allowance. Using
    mavlink_sink.Novelty itself means the level is set on exactly the number
    the vehicle will compare against it.
    """
    sink = tools(vehicle)[1]
    devices = sink.QUAD_DEVICES if vehicle == "quad" else ROBOT_DEVICES
    monitor = sink.Novelty(devices, time_constant=time_constant, dt=dt)
    levels = [monitor.update(row) for row in series][ARM_AFTER:]
    return {name: np.array([level[name] for level in levels])
            for name in devices}


def calibrate(vehicle):
    """Record every flight once, then score each smoothing window on them."""
    print("\n%s: %d healthy calibration flights, windows %s s"
          % (vehicle.upper(), len(CALIBRATION[vehicle]),
             ", ".join("%g" % t for t in TIME_CONSTANTS)), flush=True)
    calibration = [doubt_series(vehicle, *flight(vehicle, s))
                   for s in CALIBRATION[vehicle]]

    sets = [("healthy", "false", [flight(vehicle, s)
                                  for s in HELD_OUT[vehicle]])]
    if vehicle == "quad":
        for aggression in AGGRESSIONS:
            sets.append(("aggressive x%g" % aggression, "envelope",
                         [flight(vehicle, s, aggression=aggression)
                          for s in HELD_OUT[vehicle]]))
    devices = ["accel", "mag"] if vehicle == "quad" else ["left_encoder"]
    for device in devices:
        for mode, severity in MODES:
            kind = "trained" if mode in TRAINED else "untrained"
            sets.append(("%s %s %.1f" % (device, mode, severity), kind,
                         [flight(vehicle, s, mode, severity, device)
                          for s in HELD_OUT[vehicle]]))
    recorded = [(label, kind, [doubt_series(vehicle, *run) for run in runs])
                for label, kind, runs in sets]

    thresholds, rows = [], []
    for tau in TIME_CONSTANTS:
        peaks = pd.DataFrame([{name: float(v.max()) for name, v in
                               smoothed(vehicle, series, dt, tau).items()}
                              for series, dt in calibration])
        level = MARGIN * peaks.max()
        for name in peaks.columns:
            thresholds.append({"vehicle": vehicle, "device": name,
                               "time_constant": tau,
                               "healthy_peak": peaks[name].max(),
                               "healthy_median_peak": peaks[name].median(),
                               "threshold": level[name]})
        for label, kind, flights in recorded:
            alarmed, first = 0, []
            for series, dt in flights:
                hits = [np.flatnonzero(v > level[name])
                        for name, v in smoothed(vehicle, series, dt,
                                                tau).items()]
                hits = [h[0] for h in hits if len(h)]
                if hits:
                    alarmed += 1
                    first.append((ARM_AFTER + min(hits)) * dt)
            rows.append({"vehicle": vehicle, "time_constant": tau,
                         "condition": label, "kind": kind,
                         "flights": len(flights), "alarmed": alarmed,
                         "first_alarm_s": (float(np.median(first)) if first
                                           else np.nan)})
    detection = pd.DataFrame(rows)
    thresholds = pd.DataFrame(thresholds)

    # The window to fly: the most flights caught outside the healthy set,
    # with no false alarm on the healthy flights the levels were not set on;
    # the shorter window on a tie, since it alarms sooner.
    caught = detection[detection["kind"] != "false"].groupby(
        "time_constant")["alarmed"].sum()
    false = detection[detection["kind"] == "false"].set_index(
        "time_constant")["alarmed"]
    usable = [t for t in TIME_CONSTANTS if false[t] == 0]
    chosen = max(usable, key=lambda t: (caught[t], -t)) if usable else None
    thresholds["chosen"] = thresholds["time_constant"] == chosen
    detection["chosen"] = detection["time_constant"] == chosen

    print("\n  %-30s %-9s" % ("", "")
          + "".join("%9s" % ("%gs" % t) for t in TIME_CONSTANTS))
    for (label, kind), part in detection.groupby(["condition", "kind"],
                                                 sort=False):
        print("  %-30s %-9s" % (label, kind) + "".join(
            "%9s" % ("%d/%d" % (r.alarmed, r.flights))
            for r in part.itertuples()))
    print("\n  flights caught, excluding healthy: " + ", ".join(
        "%gs %d" % (t, caught[t]) for t in TIME_CONSTANTS))
    print("  chosen window: %s" % ("%g s" % chosen if chosen is not None
                                  else "none without false alarms"))
    for r in thresholds[thresholds["chosen"]].itertuples():
        print("  %-6s alert level %6.2f  (healthy peak %.2f, median %.2f)"
              % (r.device, r.threshold, r.healthy_peak, r.healthy_median_peak))
    return thresholds, detection


def merge(path, frame, vehicle):
    """Replace one vehicle's rows in a results csv, keeping the other's."""
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old[old["vehicle"] != vehicle], frame])
    frame.to_csv(path, index=False, float_format="%.4g")


def main():
    vehicles = sys.argv[1:] or ["quad", "robot"]
    if len(vehicles) > 1:
        # Each vehicle in its own process: their simulators share module
        # names (trajectories, sensors, faults), and a process holds one.
        for vehicle in vehicles:
            done = subprocess.run([sys.executable, __file__, vehicle])
            if done.returncode:
                return done.returncode
        print("\nWrote %s and %s" % (THRESHOLDS.relative_to(ROOT),
                                     DETECTION.relative_to(ROOT)))
        return 0
    vehicle = vehicles[0]
    thresholds, detection = calibrate(vehicle)
    merge(THRESHOLDS, thresholds, vehicle)
    merge(DETECTION, detection, vehicle)
    return 0


if __name__ == "__main__":
    sys.exit(main())

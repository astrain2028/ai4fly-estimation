"""
Do the quadcopter's health estimates read as fault severities?

On the ground robot they do: experiments/health_value.py shows the health
entry for a degraded sensor rising with the severity injected, which is what
makes it a diagnostic and not just an extra state. That result was never
checked on the quadcopter. quad_sim/bakeoff.py scores attitude accuracy and
NIS, and the health arm wins accuracy there, but a state can improve the
estimate by absorbing whatever the measurement map gets wrong without ever
meaning what its name says.

The question matters for deployment. deploy/mavlink_sink.py raises an alert
when a health entry crosses a level, and an alert that fires on healthy
flights is worse than none. So this runs the deployed configuration --
deploy/runtime.py's quadcopter, with the exported weights -- on healthy
flights and on flights with a fault arriving halfway, and asks how long each
entry stays above a level.

Result
------

Healthy flights: on three of eight, one entry sits above 2.0 for nearly the
whole flight (the magnetometer bias entry on one, the gyro bias entry on
another, the magnetometer noise entry on a third). Faulted flights: an
accelerometer or gyro bias at severity 3 moves the matching entry to under
1.0, and noise inflation on either device to about 0.1. The magnetometer
entries do respond to faults, to about 4.6 and 2.6, and also to nothing.

So the quadcopter's health entries are not severities. They earn their place
in the state by improving accuracy, as the bakeoff shows, but what they
absorb on this vehicle is measurement-map error as much as sensor fault, and
the two are not separable from the entry's value. The level alert is
therefore off for the quadcopter in deploy/pi_main.py, and the robot's
health_value result should not be read as transferring.

    python quad_sim/health_readout.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

# The runtime pulls in robot/ukf.py, whose own imports want the robot's
# module names. Load it first, then clear those names so the quad's take
# their place.
runtime = load_module(ROOT / "deploy" / "runtime.py", "readout_runtime")
for name in ("trajectories", "sensors", "dynamics", "faults"):
    sys.modules.pop(name, None)
sys.path.insert(0, str(HERE))
import trajectories
import sensors
import faults

DEVICES = ["accel", "gyro", "mag"]
ENTRIES = ["bias_accel", "bias_gyro", "bias_mag",
           "noise_accel", "noise_gyro", "noise_mag"]
LEVELS = [1.0, 2.0, 3.0]
HEALTHY_SEEDS = range(1, 9)
SEVERITIES = [2.0, 3.0]


def health_track(readings):
    """The six health entries at every step of the deployed filter."""
    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = runtime.quadcopter()
    estimator = runtime.Estimator(move, measure, Q, R, start, P0, dt,
                                  health_states=health_states)
    track = []
    for k in range(len(readings)):
        record = estimator.step(k * dt, readings[k])
        track.append(record["mean"][health_states])
    return np.array(track)


def longest_run(above):
    """Longest stretch of consecutive True values."""
    best = current = 0
    for flag in above:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def main():
    print("QUADCOPTER HEALTH ENTRIES AS A READOUT, DEPLOYED CONFIGURATION\n")
    rows = []

    print("Healthy flights: longest consecutive steps any entry spends above")
    print("each level, at 50 Hz, and which entry it was.\n")
    print("  %-6s" % "seed" + "".join("  %6s" % ("> %.0f" % L) for L in LEVELS)
          + "   entry        peak")
    for seed in HEALTHY_SEEDS:
        flight = trajectories.random_run(seed)
        track = health_track(sensors.stack(sensors.read_sensors(flight,
                                                               seed=seed)))
        runs = [[longest_run(track[:, j] > L) for j in range(6)] for L in LEVELS]
        worst = int(np.argmax(runs[0]))
        print("  %-6d" % seed + "".join("  %6d" % max(r) for r in runs)
              + "   %-12s %.2f" % (ENTRIES[worst], track[:, worst].max()))
        rows.append(["healthy seed %d" % seed, ENTRIES[worst], len(track)]
                    + [max(r) for r in runs] + [track[-1, worst]])

    print("\nFaulted flights, fault arriving halfway: the matching entry's")
    print("longest stretch above each level after onset, and its final value.")
    print("Severity 3 means the entry should read about 3.\n")
    print("  %-34s" % "fault" + "".join("  %6s" % ("> %.0f" % L) for L in LEVELS)
          + "   steps   final")
    for mode, offset in (("bias", 0), ("noise_inflation", 3)):
        for d, device in enumerate(DEVICES):
            for severity in SEVERITIES:
                flight = trajectories.random_run(10 + d)
                raw = sensors.read_sensors(flight, seed=10 + d)
                n = len(raw["accel_x"])
                profile = np.zeros(n)
                profile[n // 2:] = severity
                broken = faults.apply_fault(raw, device, mode, profile, seed=d)
                track = health_track(sensors.stack(broken))[n // 2:]
                j = offset + d
                runs = [longest_run(track[:, j] > L) for L in LEVELS]
                label = "%s %s severity %.0f" % (device, mode, severity)
                print("  %-34s" % label + "".join("  %6d" % r for r in runs)
                      + "   %5d   %.2f" % (len(track), track[-1, j]))
                rows.append([label, ENTRIES[j], len(track)] + runs
                            + [track[-1, j]])

    out = ROOT / "results" / "quad_health_readout.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as handle:
        handle.write("condition,entry,steps," + ",".join(
            "above_%.0f" % L for L in LEVELS) + ",final\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print("\nWrote %s" % out.relative_to(ROOT))

    print("\nThe entries improve accuracy (quad_sim/bakeoff.py) without")
    print("reading as severities. A level alert on them would fire on")
    print("healthy flights and miss accelerometer and gyro faults, so")
    print("deploy/pi_main.py leaves it off for this vehicle.")


if __name__ == "__main__":
    main()

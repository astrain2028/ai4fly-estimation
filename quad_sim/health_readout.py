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

Healthy flights, started from the true first state as every other experiment
is: no entry passes 2 on any of eight flights; the largest is 1.51.

That corrects an earlier version of this file, which started the filter from
zeros -- heading 0 -- and reported entries sitting above 2 for nearly the
whole of three healthy flights. Those three were the three flying near 180
degrees: the filter began half a turn wrong, and the health entries absorbed
the mismatch while it turned and never gave it back. The zero-start runs are
kept in the output, labelled, so the effect stays reproducible. On the
vehicle the filter now starts from the flight controller's own attitude
(deploy/mavlink_source.py), because this is what happens otherwise.

Faulted flights: ten seconds after a severity-3 bias arrives mid-flight, the
matching bias entry reads 0.62 to 0.97 -- rising, but nowhere near 3. The
ground robot's trace shows a mid-run bias entry still climbing after 32
seconds (experiments/trace.py), so this may be slowness rather than a wrong
answer; ten seconds does not distinguish the two. Noise entries stay near
zero under noise faults, as they must: a fault that does not move the mean
gives the first-moment update nothing to read (dh/dm = 0).

So on the quadcopter the health entries are quiet when nothing is wrong and
slow to report a fault. A level alert on them would not false-alarm but
would not fire within ten seconds either, and it stays off in
deploy/pi_main.py; the entries are reported, not alarmed.

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


def health_track(readings, initial=None):
    """The six health entries at every step of the deployed filter.

    ``initial`` is the attitude and rates the filter starts from. Every other
    experiment starts from the true first state, and so does this one; None
    starts from all zeros, which is how an earlier version of this file ran
    and is kept only to show what that did (see the module docstring).
    """
    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = runtime.quadcopter()
    if initial is not None:
        start[:len(initial)] = initial
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
    for start in ("truth", "zero"):
        print("  started from %s" % ("the true first state" if start == "truth"
                                     else "zeros (heading 0)"))
        for seed in HEALTHY_SEEDS:
            flight = trajectories.random_run(seed)
            initial = (trajectories.truth_matrix(flight)[0]
                       if start == "truth" else None)
            track = health_track(sensors.stack(sensors.read_sensors(
                flight, seed=seed)), initial)
            runs = [[longest_run(track[:, j] > L) for j in range(6)]
                    for L in LEVELS]
            worst = int(np.argmax(track.max(axis=0)))
            heading = np.degrees(trajectories.truth_matrix(flight)[0, 2])
            print("  %-6d" % seed + "".join("  %6d" % max(r) for r in runs)
                  + "   %-12s %.2f   heading %+.0f deg"
                  % (ENTRIES[worst], track[:, worst].max(), heading))
            rows.append(["healthy seed %d" % seed, start, ENTRIES[worst],
                         len(track)] + [max(r) for r in runs]
                        + [track[-1, worst], track[:, worst].max()])

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
                track = health_track(sensors.stack(broken),
                                     trajectories.truth_matrix(flight)[0])
                track = track[n // 2:]
                j = offset + d
                runs = [longest_run(track[:, j] > L) for L in LEVELS]
                label = "%s %s severity %.0f" % (device, mode, severity)
                print("  %-34s" % label + "".join("  %6d" % r for r in runs)
                      + "   %5d   %.2f" % (len(track), track[-1, j]))
                rows.append([label, "truth", ENTRIES[j], len(track)] + runs
                            + [track[-1, j], track[:, j].max()])

    out = ROOT / "results" / "quad_health_readout.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as handle:
        handle.write("condition,start,entry,steps," + ",".join(
            "above_%.0f" % L for L in LEVELS) + ",final,peak\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")
    print("\nWrote %s" % out.relative_to(ROOT))

    print("\nStarted from the true first state, the entries stay quiet on")
    print("healthy flights and are slow to report a fault; started from")
    print("heading 0, flights near 180 degrees carry the heading error in")
    print("their health entries for the rest of the flight.")


if __name__ == "__main__":
    main()

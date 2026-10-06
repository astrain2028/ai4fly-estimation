"""
The process that runs on the companion computer.

What it is
----------

One Python process, one loop: read sensor messages from the flight
controller, step the estimator, send the health estimates back, write a log.
It is the composition of three files that each carry their own test --

    deploy/mavlink_source.py    flight controller  ->  (timestamp, reading)
    deploy/runtime.py           (timestamp, reading)  ->  estimate
    deploy/mavlink_sink.py      estimate  ->  flight controller

-- and adds only what a process needs to run unattended: arguments, a
reconnect loop, a clean stop on SIGTERM, and a log that rotates by start
time so a crash never overwrites the record of what preceded it.

Why not ROS, and why not C++
----------------------------

The loop costs about 1.7 ms per step on the quadcopter against a 20 ms
budget at 50 Hz, measured in deploy/runtime.py. That is ten times under, so
a compiled port would be premature -- and it would be a second estimator to
keep in step with the first, which this repository has just spent effort
removing elsewhere. If the margin ever closes, the cost is in the unscented
transform's Python loops, which vectorise; it is not in the language.

A single companion computer on a single serial link to a single flight
controller does not need middleware. What ROS would add is a dependency
stack on the Pi, and a callback around the filter that makes its per-step
cost -- the claim this project exists to make -- unmeasurable behind
scheduler jitter.

The estimator is advisory. It sends named floats and alerts; it does not
feed the flight controller's EKF, and the vehicle flies on its own estimate
whether this process is running or not. That is deliberate, and it is the
first configuration to fly.

Running it
----------

    python deploy/pi_main.py --link /dev/serial0:921600
    python deploy/pi_main.py --link udpin:0.0.0.0:14550 --vehicle quadcopter
    python deploy/pi_main.py --link flight.tlog --once     # offline replay

The return path defaults to the same connection. With a .tlog as the link
there is nowhere to send, so --no-sink.

Status
------

The three parts have each been tested against a loopback or a replay. This
file has been run against a simulated vehicle on a UDP loopback, which is
in its self-test. It has not been run against a flight controller.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

runtime = load_module(HERE / "runtime.py", "pi_runtime")
source_module = load_module(HERE / "mavlink_source.py", "pi_source")
sink_module = load_module(HERE / "mavlink_sink.py", "pi_sink")

VEHICLES = {"quadcopter": runtime.quadcopter, "ground_robot": runtime.ground_robot}

# Filter NIS target is one per channel. Used for the sink's alert threshold.
NIS_TARGET = {"quadcopter": 9.0, "ground_robot": 3.0}

# Level at which a health entry raises an alert, or None for no such alert.
# On the robot the entries track injected severity (experiments/health_value.py).
# On the quadcopter, started from the true attitude, they stay under 2 on
# healthy flights but reach only 0.6 to 1.0 ten seconds into a severity-3
# bias (quad_sim/health_readout.py): an alert would not false-alarm, and
# would not fire in time either. They are reported as named values, not
# alarmed. The novelty alert is the quadcopter's warning channel.
HEALTH_ALERT = {"quadcopter": None, "ground_robot": 1.0}

# Devices the epistemic term's novelty is reported for, by channel, and the
# levels that raise an alert. The levels are set on healthy simulated flights
# by deploy/calibrate_novelty.py and read from its output; a vehicle with no
# calibration reports novelty and never alarms on it.
DEVICES = {"quadcopter": sink_module.QUAD_DEVICES,
           "ground_robot": {"left": [0], "right": [1], "gyro": [2]}}
CALIBRATION_NAME = {"quadcopter": "quad", "ground_robot": "robot"}


def novelty_alert(vehicle):
    """Alert levels per device and the smoothing window they were set for.

    From results/novelty_threshold.csv, the rows the calibration chose. With
    no calibration, ({}, 1.0): novelty is reported and never alarmed.
    """
    path = ROOT / "results" / "novelty_threshold.csv"
    if not path.exists():
        return {}, 1.0
    import csv
    with open(path) as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["vehicle"] == CALIBRATION_NAME[vehicle]
                and row.get("chosen", "True") == "True"]
    if not rows:
        return {}, 1.0
    return ({row["device"]: float(row["threshold"]) for row in rows},
            float(rows[0]["time_constant"]))


class Stop(Exception):
    pass


def _install_signal_handlers():
    def handler(signum, frame):
        raise Stop()
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run_once(args, log_dir, started_at):
    """One connection's worth of estimation. Returns steps completed."""
    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = VEHICLES[args.vehicle]()

    estimator = runtime.Estimator(move, measure, Q, R, start, P0, dt,
                                  health_states=health_states)
    # The flight controller's attitude, filled by the source before its first
    # reading. The quadcopter's filter starts from it; see mavlink_source.
    initial = {} if args.vehicle == "quadcopter" else None
    source = source_module.mavlink_source(args.link, dt_nominal=dt,
                                          initial=initial)
    sink = None
    if not args.no_sink:
        levels, window = novelty_alert(args.vehicle)
        sink = sink_module.Sink(args.sink or args.link,
                                period=args.report_period,
                                nis_target=NIS_TARGET[args.vehicle],
                                health_alert=HEALTH_ALERT[args.vehicle],
                                devices=DEVICES[args.vehicle],
                                novelty_alert=levels,
                                novelty_time_constant=window, dt=dt)

    log_path = log_dir / ("%s_%s.csv" % (args.vehicle, started_at))
    steps = 0

    # The log is written here rather than by runtime.run, because that
    # function steps the estimator itself and each record also has to reach
    # the sink. One step, one log row, one push; the columns are
    # runtime.run's so deploy/logs are interchangeable.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(log_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(runtime.log_header(names, channels))
        started = False
        for timestamp, reading in source:
            if not started and initial is not None:
                started = True
                if initial:
                    estimator.mean[:6] = [initial[k] for k in
                                          ("roll", "pitch", "yaw", "p", "q",
                                           "r")]
                    print("starting from the flight controller's attitude: "
                          "roll %.1f, pitch %.1f, yaw %.1f deg"
                          % tuple(np.degrees([initial["roll"], initial["pitch"],
                                              initial["yaw"]])), flush=True)
                else:
                    print("warning: no ATTITUDE from the flight controller; "
                          "starting level at heading 0. If the vehicle is not "
                          "facing that way the health estimates will absorb "
                          "the error for the whole flight.", file=sys.stderr,
                          flush=True)
            record = estimator.step(timestamp, reading)
            if sink is not None:
                sink.push(record, health_states)
            writer.writerow(runtime.log_row(record, channels))
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
    return steps, log_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--link", required=True,
                        help="pymavlink connection: serial device:baud, "
                             "udpin:host:port, or a .tlog path")
    parser.add_argument("--vehicle", default="quadcopter",
                        choices=sorted(VEHICLES))
    parser.add_argument("--sink", default=None,
                        help="where to send estimates; defaults to --link")
    parser.add_argument("--no-sink", action="store_true",
                        help="log only; required for a .tlog link")
    parser.add_argument("--report-period", type=float, default=1.0,
                        help="seconds between NAMED_VALUE_FLOAT reports")
    parser.add_argument("--log-dir", default=str(HERE / "logs"))
    parser.add_argument("--once", action="store_true",
                        help="exit when the source ends instead of reconnecting")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after this many steps (0 = unlimited)")
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.link.endswith(".tlog") and not args.no_sink:
        parser.error("a .tlog link has nowhere to send to; pass --no-sink")

    _install_signal_handlers()
    log_dir = Path(args.log_dir)
    started_at = time.strftime("%Y%m%d_%H%M%S")

    print("estimator: %s, link %s, sink %s"
          % (args.vehicle, args.link,
             "off" if args.no_sink else (args.sink or args.link)),
          flush=True)

    while True:
        try:
            steps, log_path = run_once(args, log_dir, started_at)
            print("source ended after %d steps; log %s" % (steps, log_path),
                  flush=True)
            if args.once or (args.max_steps and steps >= args.max_steps):
                return 0
        except Stop:
            print("stopping", flush=True)
            return 0
        except Exception as problem:            # a dropped link, most likely
            print("link error: %s" % problem, flush=True)
            if args.once:
                return 1
        time.sleep(args.reconnect_delay)
        started_at = time.strftime("%Y%m%d_%H%M%S")


def selftest():
    """The whole process against a simulated vehicle over a UDP loopback.

    A thread plays the flight controller: it sends heartbeats and one
    SCALED_IMU per sample from a quad_sim flight, converted to MAVLink's
    integer units, at the filter rate. pi_main reads them on one port and
    sends its estimates to another, where a receiver counts them. Nothing
    about a serial link is tested; everything from parsing a message to
    writing a log row and sending a report is.
    """
    import threading

    import numpy as np
    try:
        from pymavlink import mavutil
    except ImportError:
        print("pymavlink is not installed; nothing to test. The estimator")
        print("does not need it -- only the MAVLink source, sink and this")
        print("process do.")
        return 0

    # The simulated flight comes from quad_sim, whose modules share names with
    # the robot's. Import them with the robot's lifted out of the cache, then
    # put the robot's back, so main() below starts in exactly the state a
    # real launch does. An earlier version left quad_sim's in place, and that
    # hid a startup crash the vehicle would have hit on its first run.
    names = ("trajectories", "sensors", "dynamics")
    robot_modules = {m: sys.modules.pop(m) for m in names if m in sys.modules}
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "quad_sim"))
    import trajectories as qtraj
    import sensors as qsens
    sys.path[:] = saved
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(robot_modules)

    flight = qtraj.random_run(3)
    readings = qsens.stack(qsens.read_sensors(flight, seed=3))
    n_steps = 300
    G = 9.80665

    up_port, down_port = 14570, 14571
    vehicle = mavutil.mavlink_connection("udpout:127.0.0.1:%d" % up_port,
                                         source_system=1, source_component=1)
    ground = mavutil.mavlink_connection("udpin:127.0.0.1:%d" % down_port)
    received = []

    truth0 = qtraj.truth_matrix(flight)[0]

    def fly():
        # Heartbeats first so the source's wait_heartbeat returns, and the
        # flight's true starting attitude as ATTITUDE, which a real flight
        # controller streams from its own EKF; then the readings at the
        # nominal rate.
        for _ in range(3):
            vehicle.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_QUADROTOR,
                                       mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                       0, 0, 0)
            vehicle.mav.attitude_send(0, *[float(v) for v in truth0])
            time.sleep(0.1)
        for k in range(n_steps):
            r = readings[k]
            mag = r[6:9] / np.linalg.norm(r[6:9]) * 500.0    # milligauss-ish
            vehicle.mav.scaled_imu_send(
                int(k * qtraj.DT * 1000),
                int(r[0] * 1000 / G), int(r[1] * 1000 / G), int(r[2] * 1000 / G),
                int(r[3] * 1000), int(r[4] * 1000), int(r[5] * 1000),
                int(mag[0]), int(mag[1]), int(mag[2]))
            if k % 25 == 0:
                vehicle.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_QUADROTOR,
                                           mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                           0, 0, 0)
            if k % 5 == 0:
                # ATTITUDE at 10 Hz, as a flight controller streams it.
                vehicle.mav.attitude_send(
                    int(k * qtraj.DT * 1000),
                    *[float(v) for v in qtraj.truth_matrix(flight)[k]])
            time.sleep(qtraj.DT)

    def listen():
        deadline = time.time() + n_steps * qtraj.DT + 8.0
        while time.time() < deadline:
            msg = ground.recv_match(blocking=True, timeout=0.5)
            if msg is not None and msg.get_type() != "BAD_DATA":
                received.append(msg.get_type())

    print("PI PROCESS AGAINST A SIMULATED VEHICLE OVER UDP LOOPBACK\n")
    print("%d SCALED_IMU messages at %d Hz from a quad_sim flight, estimates"
          % (n_steps, round(1 / qtraj.DT)))
    print("returned as NAMED_VALUE_FLOAT to a receiver on this machine.\n")

    threading.Thread(target=listen, daemon=True).start()
    threading.Thread(target=fly, daemon=True).start()

    log_dir = HERE / "logs" / "selftest"
    code = main(["--link", "udpin:127.0.0.1:%d" % up_port,
                 "--sink", "udpout:127.0.0.1:%d" % down_port,
                 "--vehicle", "quadcopter", "--once",
                 "--max-steps", str(n_steps), "--report-period", "0.5",
                 "--log-dir", str(log_dir)])
    time.sleep(1.0)

    logs = sorted(log_dir.glob("quadcopter_*.csv"))
    rows = sum(1 for _ in open(logs[-1])) - 1 if logs else 0
    kinds = {k: received.count(k) for k in set(received)}

    import pandas as pd
    frame = pd.read_csv(logs[-1]) if logs else None
    if frame is not None:
        # Rows are matched to the truth by timestamp, not position: readings
        # that arrive while the source waits for ATTITUDE are not logged.
        index = np.round(frame["time"].values / qtraj.DT).astype(int)
        keep = (frame["updated"].values == 1) & (index < len(readings))
        truth = qtraj.truth_matrix(flight)[index[keep]]
        att = np.degrees(frame[["roll", "pitch"]].values[keep] - truth[:, :2])
        health = frame[["bias_accel", "bias_gyro", "bias_mag", "noise_accel",
                        "noise_gyro", "noise_mag"]].values.max()
    else:
        att, health = np.array([[np.nan]]), np.nan

    print("  exit code               %d" % code)
    print("  log rows                %d of %d" % (rows, n_steps))
    print("  attitude error          %.3f deg" % np.sqrt(np.mean(att ** 2)))
    print("  largest health estimate %.2f on a healthy flight" % health)
    print("  messages returned       %s" % ", ".join(
        "%s x%d" % kv for kv in sorted(kinds.items())) or "none")

    # A healthy flight started from the flight controller's attitude should
    # keep every health estimate low; started half a turn off it would not.
    ok = (code == 0 and rows == n_steps and health < 2.0
          and kinds.get("NAMED_VALUE_FLOAT", 0) > 0)
    print("\n  process ran, every reading logged, reports returned, health")
    print("  quiet on a healthy flight: %s" % ("yes" if ok else "NO"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if len(sys.argv) == 1 else main())

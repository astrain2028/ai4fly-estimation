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

The one exception is opt-in. deploy/integrity.py judges every step whether
the estimate can be used at all, and with --allow-rtl an estimate that has
become invalid -- every sensor failed, or the readings stopped -- makes this
process ask the flight controller for RTL, under the conditions
deploy/mavlink_sink.py's ReturnHome sets out. Off by default, quadcopter
only, and to be tried in SITL first (deploy/pi/README.md).

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
file has been run against a simulated vehicle over a TCP and a udpin
loopback, which is in its self-test. It has not been run against a flight
controller.
"""

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

# MAVLink 2 from the first byte. ArduPilot speaks it on a link set to
# protocol 2, and only MAVLink 2's COMMAND_ACK says whom it answers: read as
# MAVLink 1, every acknowledgement looks addressed to this process. pymavlink
# reads this when it is first imported, so it is set before anything is.
os.environ.setdefault("MAVLINK20", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

runtime = load_module(HERE / "runtime.py", "pi_runtime")
source_module = load_module(HERE / "mavlink_source.py", "pi_source")
sink_module = load_module(HERE / "mavlink_sink.py", "pi_sink")
integrity_module = load_module(HERE / "integrity.py", "pi_integrity")

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

# Health-state indices per device, for the integrity monitor's suspect tier.
HEALTH_BY_DEVICE = {"quadcopter": integrity_module.QUAD_HEALTH,
                    "ground_robot": None}


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


SERIAL = re.compile(r"^(/dev/[^:,]+|COM\d+)[:,](\d+)$", re.IGNORECASE)
SENDS_UNPROMPTED = ("udpout:", "udpbcast:", "tcp:", "mcast:")


def open_link(text, system):
    """A pymavlink connection, speaking as this vehicle's companion computer.

    A serial device with a baud rate is accepted as ``device:baud`` or
    ``device,baud``. pymavlink itself reads anything with a colon in it as a
    network address, so ``/dev/serial0:921600`` handed to it unparsed is a
    UDP socket on a host called /dev/serial0, and nothing is ever heard.
    """
    from pymavlink import mavutil
    serial = SERIAL.match(text)
    options = dict(source_system=system,
                   source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
    if serial:
        return mavutil.mavlink_connection(serial.group(1),
                                          baud=int(serial.group(2)), **options)
    return mavutil.mavlink_connection(text, **options)


def collapse_levels(vehicle):
    """The integrity monitor's residual-ratio levels per device.

    From results/integrity_threshold.csv, set on healthy flights by
    deploy/check_integrity.py. Without it the monitor's suspect tier simply
    does not use the residual; its failed tier, which decides whether the
    estimate is usable, never depended on it.
    """
    path = ROOT / "results" / "integrity_threshold.csv"
    if not path.exists():
        return {}
    import csv
    with open(path) as handle:
        return {row["device"]: float(row["threshold"])
                for row in csv.DictReader(handle)
                if row["vehicle"] == CALIBRATION_NAME[vehicle]}


class Stop(Exception):
    pass


def _install_signal_handlers():
    def handler(signum, frame):
        raise Stop()
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def run_once(args, log_dir, started_at, vehicle=None, home=None):
    """One connection's worth of estimation. Returns steps completed.

    ``vehicle`` and ``home`` belong to main() and outlive a reconnect: what
    the flight controller last said about itself, and ReturnHome's memory of
    this flight -- a pilot's decision to leave RTL must survive the link
    dropping and coming back.
    """
    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = VEHICLES[args.vehicle]()

    estimator = runtime.Estimator(move, measure, Q, R, start, P0, dt,
                                  health_states=health_states)
    # The flight controller's attitude, filled by the source before its first
    # reading. The quadcopter's filter starts from it; see mavlink_source.
    initial = {} if args.vehicle == "quadcopter" else None
    vehicle = {} if vehicle is None else vehicle

    # One link, read by the source and written by the sink. Two connections
    # to one udpin port is two sockets, and pymavlink sends from a udpin
    # socket only to peers that socket has heard -- the sink's would hear
    # no one and every report and request would vanish. This process speaks
    # as the vehicle's companion computer: the vehicle's system id, the
    # onboard-computer component.
    link = None if args.replay else open_link(args.link, args.system)
    source = source_module.mavlink_source(link if link is not None else args.link,
                                          dt_nominal=dt, initial=initial,
                                          vehicle=vehicle, system=args.system)
    levels, window = novelty_alert(args.vehicle)
    monitor = integrity_module.Integrity(
        DEVICES[args.vehicle], dt=dt,
        health=HEALTH_BY_DEVICE[args.vehicle] if health_states else None,
        novelty_alert=levels, novelty_time_constant=window,
        collapse=collapse_levels(args.vehicle))
    sink = None
    if not args.no_sink:
        sink = sink_module.Sink(open_link(args.sink, args.system)
                                if args.sink else link,
                                source_system=args.system,
                                period=args.report_period,
                                nis_target=NIS_TARGET[args.vehicle],
                                health_alert=HEALTH_ALERT[args.vehicle],
                                devices=DEVICES[args.vehicle],
                                novelty_alert=levels,
                                novelty_time_constant=window, dt=dt)
    if home is not None:
        home.sink, home.vehicle = sink, vehicle

    log_path = log_dir / ("%s_%s.csv" % (args.vehicle, started_at))
    steps = 0

    # The log is written here rather than by runtime.run, because that
    # function steps the estimator itself and each record also has to reach
    # the sink. One step, one log row, one push; the columns are
    # runtime.run's so deploy/logs are interchangeable. A log that cannot be
    # written -- a full or read-only card -- costs the log, not the loop: the
    # monitor and the return-home request keep running.
    import csv
    handle = None
    try:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "w", newline="")
            writer = csv.writer(handle)
            writer.writerow(runtime.log_header(names, channels))
        except OSError as problem:
            print("warning: cannot write the log (%s); running without it"
                  % problem, file=sys.stderr, flush=True)
            writer = None
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
            status = monitor.update(record, reading)
            record["integrity"] = status
            if status["changed"]:
                print("integrity: %s%s" % (status["state"],
                                           " (%s)" % status["reason"]
                                           if status["reason"] else ""),
                      flush=True)
            if sink is not None:
                sink.push(record, health_states, integrity=status)
            if home is not None and sink is not None:
                for event in home.update(status):
                    print("return home: %s" % event, flush=True)
            if writer is not None:
                try:
                    writer.writerow(runtime.log_row(record, channels))
                except OSError as problem:
                    print("warning: the log stopped (%s); running without it"
                          % problem, file=sys.stderr, flush=True)
                    writer = None
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
    finally:
        for closing in (handle, getattr(sink, "link", None), link):
            try:
                if closing is not None:
                    closing.close()
            except Exception:
                pass
    return steps, log_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--link", required=True,
                        help="the flight controller: a serial device and "
                             "baud (/dev/serial0:921600), udpin:host:port, "
                             "tcp:host:port, or a .tlog path to replay")
    parser.add_argument("--vehicle", default="quadcopter",
                        choices=sorted(VEHICLES))
    parser.add_argument("--sink", default=None,
                        help="where to send estimates; defaults to --link, "
                             "over the same connection")
    parser.add_argument("--system", type=int, default=1,
                        help="the flight controller's MAVLink system id "
                             "(ArduPilot's SYSID_THISMAV); messages from any "
                             "other system are ignored")
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
    parser.add_argument("--allow-rtl", action="store_true",
                        help="when every sensor has failed, ask the flight "
                             "controller to switch to RTL. Off by default; "
                             "quadcopter only; try it in SITL first "
                             "(deploy/pi/README.md)")
    args = parser.parse_args(argv)

    args.replay = args.link.lower().endswith(".tlog")
    if args.replay and not args.no_sink:
        parser.error("a .tlog link has nowhere to send to; pass --no-sink")
    if not 1 <= args.system <= 255:
        parser.error("--system is a MAVLink system id, 1 to 255")
    if args.allow_rtl and args.vehicle != "quadcopter":
        parser.error("--allow-rtl knows ArduCopter's mode numbers only")
    if args.allow_rtl and args.no_sink:
        parser.error("--allow-rtl needs the sink to send its request")
    if args.allow_rtl and args.sink:
        parser.error("--allow-rtl answers over --link, where the flight "
                     "controller's acknowledgement arrives; leave --sink out")
    if args.sink and not (args.sink.startswith(SENDS_UNPROMPTED)
                          or SERIAL.match(args.sink)
                          or args.sink.startswith(("/dev/", "COM"))):
        # udpin, udp, tcpin and a bare host:port all listen: pymavlink sends
        # from them only to peers they have heard, and a sink hears no one.
        parser.error("a separate --sink must send unprompted (udpout:, "
                     "udpbcast:, tcp:, mcast: or a serial device); leave it "
                     "out to answer over --link")

    _install_signal_handlers()
    log_dir = Path(args.log_dir)
    started_at = time.strftime("%Y%m%d_%H%M%S")

    print("estimator: %s, link %s, sink %s"
          % (args.vehicle, args.link,
             "off" if args.no_sink else (args.sink or "back over the link")),
          flush=True)
    if args.allow_rtl:
        print("RTL requests ENABLED: if every sensor fails while armed and "
              "flying, this process will ask the flight controller for RTL",
              flush=True)

    # Kept across reconnects: what the flight controller last said about
    # itself, and ReturnHome's memory of this flight.
    vehicle = {}
    home = sink_module.ReturnHome(None, vehicle) if args.allow_rtl else None

    while True:
        try:
            steps, log_path = run_once(args, log_dir, started_at,
                                       vehicle=vehicle, home=home)
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
    """The whole process against a simulated vehicle over a loopback.

    A thread plays the flight controller: it sends heartbeats -- armed and
    flying -- and, once asked, one SCALED_IMU per sample from a quad_sim
    flight, converted to MAVLink's integer units, on a real-time schedule.
    It acts on a mode request only if addressed to it, acknowledges it to
    whoever sent it, and switches mode a while later, so that only the
    acknowledgement can end the asking. Nothing about a serial link is
    tested; everything from parsing a message to writing a log row, sending
    a report and asking for RTL is.

    Three flights:

        healthy         over TCP, with --allow-rtl: the estimate must stay
                        usable and nothing be asked
        every sensor    over a udpin link, the one the SITL procedure uses,
        frozen          with --allow-rtl: every device freezes at 3 s; the
                        estimate must go invalid, RTL be asked for exactly
                        once, and the acknowledgement end the asking
        separate sink   over TCP, estimates to a separate --sink receiver
    """
    import contextlib
    import io
    import socket
    import threading

    import numpy as np
    try:
        from pymavlink import mavutil
    except ImportError:
        print("pymavlink is not installed; nothing to test. The estimator")
        print("does not need it -- only the MAVLink source, sink and this")
        print("process do.")
        return 0
    import pandas as pd

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
    truth = qtraj.truth_matrix(flight)
    G = 9.80665
    link = mavutil.mavlink
    ARMED = link.MAV_MODE_FLAG_SAFETY_ARMED | link.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
    LOITER = 5
    SWITCH_DELAY = sink_module.ACK_TIMEOUT + 1.0   # ack first, mode much later

    class FakeUdp:
        """The flight controller's end of a udpin link, as a plain socket.

        pymavlink's udpout binds, on Windows, to its destination's port, so
        a pymavlink flight controller and this process on one machine would
        share a port and hear each other's replies as their own. A plain
        socket on a port of its own is what a real peer is.
        """

        def __init__(self, port):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", 0))
            self.sock.setblocking(False)
            self.dest = ("127.0.0.1", port)
            self.mav = link.MAVLink(self, srcSystem=1, srcComponent=1)
            self.parser = link.MAVLink(None)
            self.inbox = []

        def write(self, buf):
            try:
                self.sock.sendto(buf, self.dest)
            except OSError:
                pass                        # nobody listening yet

        def recv_match(self, blocking=False, timeout=None):
            while not self.inbox:
                try:
                    data, _ = self.sock.recvfrom(65535)
                except OSError:             # nothing waiting, or a refusal
                    return None
                self.inbox += self.parser.parse_buffer(data) or []
            return self.inbox.pop(0)

        def close(self):
            self.sock.close()

    def loopback(label, n_steps, transport, port, down_port=None,
                 freeze_from=None, allow_rtl=True):
        """One flight against a fake flight controller.

        ``transport`` is "tcp" -- the fake listens, this process connects --
        or "udp", this process listening as with udpin. With ``down_port``
        the estimates go to a separate receiver there, as with --sink;
        without it they come back over the link itself, the default.
        """
        if transport == "tcp":
            vehicle = mavutil.mavlink_connection("tcpin:127.0.0.1:%d" % port,
                                                 source_system=1,
                                                 source_component=1)
            link_arg = "tcp:127.0.0.1:%d" % port
        else:
            vehicle = FakeUdp(port)
            link_arg = "udpin:127.0.0.1:%d" % port
        ground = (mavutil.mavlink_connection("udpin:127.0.0.1:%d" % down_port)
                  if down_port else None)
        received, texts, requests = [], [], []
        state = {"mode": LOITER, "ack": None, "switch_at": None,
                 "streaming": False}
        finished = threading.Event()

        def heartbeat():
            vehicle.mav.heartbeat_send(link.MAV_TYPE_QUADROTOR,
                                       link.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                       ARMED, state["mode"],
                                       link.MAV_STATE_ACTIVE)

        def handle(msg):
            if msg is None or msg.get_type() == "BAD_DATA":
                return
            kind = msg.get_type()
            if kind == "COMMAND_LONG":
                if (msg.target_system, msg.target_component) != (1, 1):
                    return                  # not addressed to this autopilot
                if (msg.command == link.MAV_CMD_SET_MESSAGE_INTERVAL
                        and int(msg.param1) == link.MAVLINK_MSG_ID_SCALED_IMU):
                    state["streaming"] = True
                elif msg.command == link.MAV_CMD_DO_SET_MODE:
                    requests.append(int(msg.param2))
                    state["ack"] = (msg.command, msg.get_srcSystem(),
                                    msg.get_srcComponent(), int(msg.param2))
                return
            received.append(kind)
            if kind == "STATUSTEXT":
                texts.append(msg.text)

        def drain(connection):
            while True:
                msg = connection.recv_match(blocking=False)
                if msg is None:
                    return
                handle(msg)

        def fly():
            # Heartbeats and the flight's true starting attitude, as a flight
            # controller streams them from its own EKF, until the process
            # asks for SCALED_IMU -- a real one sends nothing before -- then
            # the readings on a real-time schedule: reading k at k * DT after
            # the first, stamped so, as a flight controller's clock keeps
            # time, for as long as the process runs.
            deadline = time.time() + 60.0
            while not state["streaming"] and time.time() < deadline:
                heartbeat()
                vehicle.mav.attitude_send(0, *[float(v) for v in truth[0]])
                time.sleep(0.1)
                drain(vehicle)
            k, start = 0, time.time()
            while not finished.is_set() and k < len(readings):
                wait = start + k * qtraj.DT - time.time()
                if wait > 0:
                    time.sleep(wait)
                frozen = freeze_from is not None and k >= freeze_from
                r = readings[freeze_from - 1 if frozen else k]
                mag = r[6:9] / np.linalg.norm(r[6:9]) * 500.0  # milligauss-ish
                vehicle.mav.scaled_imu_send(
                    int(k * qtraj.DT * 1000),
                    int(r[0] * 1000 / G), int(r[1] * 1000 / G),
                    int(r[2] * 1000 / G),
                    int(r[3] * 1000), int(r[4] * 1000), int(r[5] * 1000),
                    int(mag[0]), int(mag[1]), int(mag[2]))
                drain(vehicle)
                if state["ack"] is not None:
                    # The answer, addressed to whoever asked; the mode itself
                    # changes only well after, as if the vehicle were slow.
                    command, system, component, mode = state["ack"]
                    vehicle.mav.command_ack_send(command,
                                                 link.MAV_RESULT_ACCEPTED,
                                                 0, 0, system, component)
                    state["ack"] = None
                    state["switch_at"] = (time.time() + SWITCH_DELAY, mode)
                if (state["switch_at"] is not None
                        and time.time() >= state["switch_at"][0]):
                    state["mode"], state["switch_at"] = state["switch_at"][1], None
                    heartbeat()
                if k % 25 == 0:
                    heartbeat()
                    vehicle.mav.extended_sys_state_send(
                        0, link.MAV_LANDED_STATE_IN_AIR)
                if k % 5 == 0:
                    # ATTITUDE at 10 Hz, as a flight controller streams it.
                    vehicle.mav.attitude_send(
                        int(k * qtraj.DT * 1000),
                        *[float(v) for v in truth[k]])
                k += 1

        def listen():
            while not finished.is_set():
                handle(ground.recv_match(blocking=True, timeout=0.5))

        if ground is not None:
            threading.Thread(target=listen, daemon=True).start()
        flying = threading.Thread(target=fly, daemon=True)
        flying.start()

        log_dir = HERE / "logs" / "selftest" / label
        argv = ["--link", link_arg, "--vehicle", "quadcopter", "--once",
                "--max-steps", str(n_steps), "--report-period", "0.5",
                "--log-dir", str(log_dir)]
        if allow_rtl:
            argv += ["--allow-rtl"]
        if down_port:
            argv += ["--sink", "udpout:127.0.0.1:%d" % down_port]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main(argv)
        print(printed.getvalue(), end="")
        time.sleep(0.5)
        finished.set()
        flying.join(timeout=2.0)
        logs = sorted(log_dir.glob("quadcopter_*.csv"))
        frame = pd.read_csv(logs[-1]) if logs else None
        return code, frame, received, texts, requests, printed.getvalue()

    print("PI PROCESS AGAINST A SIMULATED VEHICLE OVER LOOPBACK\n")
    print("SCALED_IMU at %d Hz from a quad_sim flight. The healthy and frozen"
          % round(1 / qtraj.DT))
    print("flights run with --allow-rtl and answer over the link itself, the")
    print("default; a third sends its estimates to a separate receiver.\n")

    # -- a healthy flight, over TCP -----------------------------------------
    n_steps = 300
    code, frame, received, texts, requests, out = loopback(
        "healthy", n_steps, "tcp", 14570)
    rows = 0 if frame is None else len(frame)
    kinds = {k: received.count(k) for k in set(received)}
    if frame is not None:
        # Rows are matched to the truth by timestamp, not position.
        index = np.round(frame["time"].values / qtraj.DT).astype(int)
        keep = (frame["updated"].values == 1) & (index < len(readings))
        updated = int(keep.sum())
        att = np.degrees(frame[["roll", "pitch"]].values[keep]
                         - truth[index[keep], :2])
        health = frame[["bias_accel", "bias_gyro", "bias_mag", "noise_accel",
                        "noise_gyro", "noise_mag"]].values.max()
        worst = int(frame["integrity"].max())
        silent = int(frame["integrity_reason"].fillna("")
                     .str.startswith("no sensor data").sum())
    else:
        updated, att, health, worst, silent = 0, np.array([[np.nan]]), np.nan, -1, -1

    print("HEALTHY FLIGHT, TCP")
    print("  exit code               %d" % code)
    print("  log rows                %d of %d, %d with a reading"
          % (rows, n_steps, updated))
    print("  attitude error          %.3f deg" % np.sqrt(np.mean(att ** 2)))
    print("  largest health estimate %.2f" % health)
    print("  worst integrity         %d  (0 ok, 1 degraded, 2 invalid)" % worst)
    print("  RTL requests            %d" % len(requests))
    print("  messages returned       %s\n" % ", ".join(
        "%s x%d" % kv for kv in sorted(kinds.items())) or "none")

    # A healthy flight started from the flight controller's attitude should
    # keep every health estimate low; started half a turn off it would not.
    # Nearly every row should carry a reading: the fake flight controller
    # streams only once asked, as a real one does, and keeps streaming until
    # the process stops.
    healthy_ok = (code == 0 and rows == n_steps and updated >= n_steps - 5
                  and silent == 0 and health < 2.0 and worst == 0
                  and not requests and kinds.get("NAMED_VALUE_FLOAT", 0) > 0)

    # -- every sensor frozen at 3 s, over udpin -------------------------------
    n_steps, freeze = 500, 150
    code, frame, received, texts, requests, out = loopback(
        "frozen", n_steps, "udp", 14572, freeze_from=freeze)
    first = np.nan
    before = True
    if frame is not None and (frame["integrity"] == 2).any():
        first = float(frame.loc[frame["integrity"] == 2, "time"].iloc[0])
        before = bool((frame.loc[frame["time"] < freeze * qtraj.DT,
                                 "integrity"] > 0).any())
    invalid_text = [t for t in texts if t.startswith("ESTIMATE INVALID")]
    asked = [t for t in texts if "requesting RTL" in t]
    accepted = "RTL request accepted" in out

    print("\nEVERY SENSOR FROZEN AT %.1f s, udpin" % (freeze * qtraj.DT))
    print("  exit code               %d" % code)
    print("  declared invalid at     %.2f s  (%.2f s after the failure)"
          % (first, first - freeze * qtraj.DT))
    print("  RTL requests            %d  %s" % (len(requests), requests))
    print("  acknowledgement heard   %s" % ("yes" if accepted else "NO"))
    print("  alerts                  %s" % texts)

    # The fake flight controller changes mode SWITCH_DELAY after its
    # acknowledgement, longer than ReturnHome waits before asking again: a
    # second request here would mean the acknowledgement went unheard.
    frozen_ok = (code == 0 and np.isfinite(first) and not before
                 and 0 < first - freeze * qtraj.DT < 2.5
                 and requests == [sink_module.COPTER_RTL] and accepted
                 and invalid_text and asked)

    # -- a separate --sink: reports reach a receiver that is not the link -----
    code, frame, received, texts, requests, out = loopback(
        "sink", 150, "tcp", 14574, down_port=14575, allow_rtl=False)
    named = received.count("NAMED_VALUE_FLOAT")
    print("\nSEPARATE --sink, TCP")
    print("  exit code               %d" % code)
    print("  named values received   %d" % named)
    sink_ok = code == 0 and frame is not None and len(frame) == 150 and named > 0

    print("\n  healthy: usable throughout, nothing asked: %s"
          % ("yes" if healthy_ok else "NO"))
    print("  frozen: invalid within 2.5 s of the failure and not before, RTL")
    print("  asked for exactly once over udpin, the acknowledgement heard: %s"
          % ("yes" if frozen_ok else "NO"))
    print("  separate sink: reports arrived: %s" % ("yes" if sink_ok else "NO"))
    return 0 if healthy_ok and frozen_ok and sink_ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if len(sys.argv) == 1 else main())

"""
A MAVLink stream as a source for deploy/runtime.py.

Status
------

This file has not been run against a flight controller, a SITL instance, or
a recorded log. It is written against the pymavlink API and the MAVLink
message definitions, and its self-test exercises only the unit conversion and
the missing-reading policy with synthetic messages. Everything else is a
claim about how a link behaves that a bench test has to confirm. That is
stated here rather than discovered later, because a file that reads as
tested and is not is worse than one that says so.

What it produces
----------------

A generator of ``(timestamp, reading)`` pairs in the form runtime.run
expects, from a MAVLink connection -- serial, UDP, or a recorded ``.tlog``
file, which pymavlink replays through the same interface. The last of those
is how this file gets tested before any hardware exists: record one flight,
replay it here, and the estimator sees what it would have seen live.

Which message, and why the quadcopter comes first
-------------------------------------------------

``SCALED_IMU`` carries a three-axis accelerometer, a three-axis gyro and a
three-axis magnetometer with one timestamp. That is the quad_sim sensor suite
exactly, nine channels in the same order, so the adapter for that vehicle is a
unit conversion and nothing more.

The ground robot is less direct. ArduPilot reports wheel encoders as
``WHEEL_DISTANCE`` -- cumulative distance per wheel -- so the rates the
filter wants have to be recovered by differencing consecutive messages
against their timestamps, and the gyro comes from a separate ``SCALED_IMU``
message that has to be paired with it. Both are solvable and neither is done
here; the quad path is the one to prove the loop on.

Units
-----

MAVLink integers with fixed scaling, converted to what quad_sim trained on:

    accelerometer   milli-g          ->  m/s^2      multiply by 9.80665 / 1000
    gyro            milli-rad/s      ->  rad/s      divide by 1000
    magnetometer    milli-gauss      ->  unit vector normalise the three axes

The magnetometer is normalised because quad_sim's model reads the field's
direction and not its strength -- the strength depends on the calibration
and the site, and neither is something the estimator should have opinions
about.

Frame
-----

MAVLink's body frame is forward-right-down, and so is quad_sim's: the level
accelerometer reads +g on z in both. No axis remapping is applied. That
agreement is asserted in the self-test on a synthetic level reading rather
than assumed, since a sign error here would look like a permanent 180 degree
roll to the filter.

Missing readings
----------------

When no message arrives within the expected interval the generator yields
``(time, None)``, and the estimator does a predict-only step. Repeating the
previous reading instead would look identical to a frozen sensor to every
mechanism in this project. The time of a gap step advances by the wall-clock
time that actually passed, so a filter predicting through a two-second
silence integrates two seconds, and deploy/integrity.py can tell how long
the stream has been silent.

At the end of a recorded ``.tlog`` the generator stops rather than
predicting through silence forever.

Whose messages
--------------

ArduPilot forwards what it hears on one link to the others, so this link
carries a ground station's heartbeats, a gimbal's, and on a shared telemetry
network another vehicle's. The first heartbeat from an ArduPilot autopilot
component of the expected MAVLink system -- ArduPilot's SYSID_THISMAV, 1
unless it has been changed, and deploy/pi_main.py's --system -- fixes which
system and component is the flight controller, and
every message from anyone else -- readings, attitude, heartbeats,
acknowledgements -- is ignored. A peripheral's "disarmed" heartbeat must not
reset what deploy/mavlink_sink.py's ReturnHome remembers, and its request
must not go to another vehicle.

The vehicle's state
-------------------

Given a ``vehicle`` dict, the generator also keeps the flight controller's
``HEARTBEAT`` -- armed or not, which flight mode, which kind of vehicle --
its ``EXTENDED_SYS_STATE`` landed state, asked for at 2 Hz, and the last
``COMMAND_ACK`` per command. ReturnHome reads them before it asks for
anything.

A reboot
--------

The flight controller's ``time_boot_ms`` restarts at zero when it reboots,
and it forgets the stream it was asked for: ArduPilot puts ``SCALED_IMU`` in
no stream-rate group, so after a reboot it is simply never sent. Two things
cover that. Whenever the flight controller is still heard but no reading has
come for RE_REQUEST seconds, the stream is asked for again, at most once a
second -- which also covers a request lost at startup. And a timestamp that
runs backwards by more than REBOOT_JUMP is taken as a reboot, accepted, and
counted in ``vehicle["reboots"]``. Without either, the readings after a
reboot never came or were dropped as out of order, and the estimator went
quiet without a word.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

REBOOT_JUMP = 1.0          # seconds a timestamp may run backwards before it is
                           # taken as a flight-controller reboot

G = 9.80665

# Per-axis scale from the raw SCALED_IMU integers, in channel order:
# accel x/y/z, gyro x/y/z, mag x/y/z. The magnetometer scale is a placeholder;
# those three channels are normalised to a unit vector after scaling.
CHANNELS = ["accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z"]
SCALE = np.array([G / 1000.0] * 3 + [1e-3] * 3 + [1.0] * 3)

FIELDS = ["xacc", "yacc", "zacc", "xgyro", "ygyro", "zgyro",
          "xmag", "ymag", "zmag"]


def reading_from_scaled_imu(message):
    """Nine channels in quad_sim's units, from one SCALED_IMU message.

    ``message`` needs only the nine integer fields by attribute, which is
    what pymavlink's message objects provide and what the self-test fakes.
    """
    raw = np.array([float(getattr(message, f)) for f in FIELDS])
    reading = raw * SCALE
    field = reading[6:9]
    norm = np.linalg.norm(field)
    if norm > 0:
        reading[6:9] = field / norm
    return reading


ATTITUDE_WAIT = 2.0        # seconds to wait for the flight controller's attitude
RE_REQUEST = 1.0           # seconds without a reading, the flight controller
                           # still heard, before the stream is asked for again
HEARD = 3.0                # seconds: a heartbeat older than this is not hearing it
SILENT_WARNING = 5.0       # seconds after connecting with no reading at all

# What counts as the flight controller: an ArduPilot autopilot component that
# is a vehicle. Everything else on the link -- a ground station, a gimbal,
# another vehicle on a shared telemetry network -- ArduPilot forwards to this
# link too, and none of it may speak for this vehicle.
NOT_A_VEHICLE = ("MAV_TYPE_GCS", "MAV_TYPE_GIMBAL", "MAV_TYPE_ADSB",
                 "MAV_TYPE_ONBOARD_CONTROLLER")


def is_flight_controller(msg, mavlink):
    """Whether a HEARTBEAT comes from an ArduPilot flight controller."""
    return (msg.get_type() == "HEARTBEAT"
            and msg.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
            and msg.get_srcComponent() == mavlink.MAV_COMP_ID_AUTOPILOT1
            and msg.type not in [getattr(mavlink, t) for t in NOT_A_VEHICLE])


def _note(vehicle, msg, clock):
    """Keep what the flight controller says about itself.

    Called only with messages already known to come from it.
    """
    kind = msg.get_type()
    if kind == "HEARTBEAT":
        from pymavlink import mavutil
        vehicle.update(
            armed=bool(msg.base_mode
                       & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
            status=msg.system_status, mode=msg.custom_mode,
            autopilot=msg.autopilot, type=msg.type, heartbeat=clock())
    elif kind == "EXTENDED_SYS_STATE":
        vehicle["landed"] = msg.landed_state
    elif kind == "COMMAND_ACK":
        vehicle.setdefault("acks", {})[msg.command] = {
            "result": msg.result,
            "target_system": getattr(msg, "target_system", 0),
            "target_component": getattr(msg, "target_component", 0),
            "time": clock()}


def mavlink_source(connection, dt_nominal=0.02, message="SCALED_IMU",
                   timeout_factor=2.0, initial=None, vehicle=None,
                   system=None, clock=time.monotonic):
    """Yield ``(seconds, reading)`` from a MAVLink connection, or None on a gap.

    ``connection`` is a connection already open -- which is how
    deploy/pi_main.py shares one link between this and the sink, and how a
    serial device gets its baud rate (pi_main's open_link) -- or anything
    pymavlink's ``mavlink_connection`` accepts as it stands: a UDP or TCP
    endpoint (``"udpin:0.0.0.0:14550"``), a serial device at 115200 baud, or
    a ``.tlog`` path for offline replay. Not ``"/dev/serial0:921600"``:
    pymavlink reads anything with a colon as a network address.

    Only messages from the flight controller are used: the first heartbeat
    from an ArduPilot autopilot component -- of MAVLink system ``system``,
    if given -- fixes its system and component, and anything from anyone
    else is ignored. Give ``system``: on a shared telemetry network the first
    autopilot heard may be another vehicle's.

    Timestamps are the flight controller's ``time_boot_ms``, in seconds, so
    the estimator's measured interval reflects when the reading was taken
    and not when it was received.

    ``initial``, if given, is a dict filled before the first reading with the
    flight controller's own attitude and rates -- roll, pitch, yaw, p, q, r --
    from one ATTITUDE message, for the estimator to start from. It is left
    empty if none arrives in ATTITUDE_WAIT seconds. Starting the filter from
    zeros is not a harmless default: quad_sim/health_readout.py found that a
    filter started half a turn wrong in heading spends the rest of the flight
    with its health estimates soaking up the mismatch.

    ``vehicle``, if given, is a dict kept up to date from the flight
    controller: ``armed``, ``status`` (MAV_STATE), ``mode`` (the custom mode
    number), ``autopilot`` and ``type``, ``landed`` (EXTENDED_SYS_STATE's
    landed state), ``heartbeat`` (``clock()`` when the last arrived),
    ``system`` and ``component``, ``acks`` (the last COMMAND_ACK per command)
    and ``reboots``. It may outlive one connection: values from an earlier
    one stay until the flight controller says otherwise.
    """
    from pymavlink import mavutil          # imported here: not a runtime dep
    mavlink = mavutil.mavlink

    replay = isinstance(connection, str) and connection.lower().endswith(".tlog")
    if replay:
        # The sequential reader: it ends at the end of the file. The default
        # for a .tlog maps it into memory and waits at the end forever.
        link = mavutil.mavlogfile(connection)
    elif isinstance(connection, str):
        link = mavutil.mavlink_connection(connection)
    else:
        link = connection

    def receive(types, timeout):
        """The next message of these types from anyone, or None.

        Reads what is already buffered before it decides nothing came: a
        zero timeout must not mean "do not look". In a replay None is the
        end of the file.
        """
        if replay:
            return link.recv_match(type=types, blocking=False)
        if timeout <= 0:
            return link.recv_match(type=types, blocking=False)
        return link.recv_match(type=types, blocking=True, timeout=timeout)

    # The flight controller, by the first heartbeat that is one. Waiting for
    # it says why, now and then, rather than hang in silence: a wrong
    # --system, a wrong baud rate and a different autopilot all look alike
    # from outside.
    started = clock()
    next_note = started + SILENT_WARNING
    others, heard_any = {}, False
    while True:
        msg = receive(None, 1.0)
        if msg is not None and msg.get_type() != "BAD_DATA":
            heard_any = True
            if msg.get_type() == "HEARTBEAT":
                if is_flight_controller(msg, mavlink) and (
                        system is None or msg.get_srcSystem() == system):
                    break
                others[(msg.get_srcSystem(), msg.get_srcComponent())] = (
                    msg.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
                    and msg.get_srcComponent() == mavlink.MAV_COMP_ID_AUTOPILOT1)
        if (msg is None and replay) or (not replay and clock() >= next_note):
            fcs = [s for (s, c), is_fc in others.items() if is_fc]
            if fcs:
                why = ("heard an ArduPilot flight controller as system %s, "
                       "but --system is %s" % (", ".join(map(str, sorted(fcs))),
                                               system))
            elif others:
                why = ("heard heartbeats from %s, none from an ArduPilot "
                       "flight controller" % sorted(others))
            elif heard_any:
                why = "MAVLink traffic, but no heartbeat"
            else:
                why = "no MAVLink at all: check the device, baud rate and wiring"
            print("warning: no flight controller after %.0f s: %s"
                  % (clock() - started, why), file=sys.stderr, flush=True)
            next_note = clock() + 30.0
            if replay:
                return
    fc = (msg.get_srcSystem(), msg.get_srcComponent())
    heard = clock()
    if vehicle is not None:
        vehicle.update(system=fc[0], component=fc[1])
        vehicle.setdefault("reboots", 0)
        vehicle.setdefault("acks", {})
        _note(vehicle, msg, clock)

    def ours(msg):
        return (msg.get_srcSystem(), msg.get_srcComponent()) == fc

    def ask(message_id, interval_us):
        # ArduPilot streams SCALED_IMU in no stream-rate group, so the
        # request is the only thing that makes it arrive -- and it lasts only
        # until the flight controller reboots. A replay cannot be asked.
        if not replay:
            link.mav.command_long_send(
                fc[0], fc[1], mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                message_id, interval_us, 0, 0, 0, 0, 0)

    pending = None              # a reading met while waiting for ATTITUDE
    if initial is not None:
        # ArduPilot streams ATTITUDE on most links already; ask for it anyway
        # at 10 Hz rather than rely on the stream-rate parameters. A replay
        # waits until the first ATTITUDE or the first reading, whichever the
        # recording has first, and starts without an attitude in the second
        # case, as a live link does when none comes.
        ask(mavlink.MAVLINK_MSG_ID_ATTITUDE, 100000)
        deadline = clock() + ATTITUDE_WAIT
        while replay or clock() < deadline:
            got = receive(["ATTITUDE", message], deadline - clock())
            if got is None:
                if replay:
                    return
                continue
            if not ours(got):
                continue
            if got.get_type() == "ATTITUDE":
                initial.update(roll=got.roll, pitch=got.pitch, yaw=got.yaw,
                               p=got.rollspeed, q=got.pitchspeed,
                               r=got.yawspeed)
                break
            if replay:
                pending = got
                break

    # Ask for the readings at the filter's rate, and for the landed state at
    # 2 Hz, which is how deploy/mavlink_sink.py's ReturnHome knows the
    # vehicle is in the air.
    message_id = getattr(mavlink, "MAVLINK_MSG_ID_%s" % message)

    def ask_for_stream():
        ask(message_id, int(1e6 * dt_nominal))
        if vehicle is not None:
            ask(mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 500000)

    ask_for_stream()
    asked = clock()

    types = [message, "HEARTBEAT", "COMMAND_ACK", "EXTENDED_SYS_STATE"]
    timeout = timeout_factor * dt_nominal
    last_time = None        # the last time yielded, a reading's or a gap's
    last_real = None        # the last flight-controller timestamp accepted
    last_reading = None     # clock() when the last reading arrived
    stale_since = None      # clock() when readings began arriving stale
    last_wall = clock()
    deadline = last_wall + timeout
    warned = False
    while True:
        if pending is not None:
            msg, pending = pending, None
        else:
            msg = receive(types, deadline - clock())
        if msg is not None:
            if not ours(msg):
                continue
            if msg.get_type() != message:
                if msg.get_type() == "HEARTBEAT":
                    heard = clock()
                if vehicle is not None:
                    _note(vehicle, msg, clock)
                continue
        if msg is None:
            if replay:
                return                   # the end of the recording
            if clock() < deadline:
                continue
            now = clock()
            deadline = now + timeout
            # No readings while the flight controller is still talking: it
            # has rebooted and forgotten the request, or never got it. Ask
            # again, once a second at most.
            silent = now - (started if last_reading is None else last_reading)
            if (silent > RE_REQUEST and now - heard < HEARD
                    and now - asked > RE_REQUEST):
                ask_for_stream()
                asked = now
            if last_reading is None and silent > SILENT_WARNING and not warned:
                print("warning: no %s from the flight controller after %.0f s"
                      % (message, silent), file=sys.stderr, flush=True)
                warned = True
            # Nothing arrived inside the window. Advance by the time that
            # actually passed, so the estimator predicts through the gap, and
            # report no reading rather than a stale one.
            if last_time is None:
                last_wall = now
                continue
            last_time = last_time + (now - last_wall)
            last_wall = now
            yield last_time, None
            continue

        stamp = msg.time_boot_ms / 1000.0
        if last_real is not None and stamp < last_real - REBOOT_JUMP:
            # The clock restarted: a reboot. Accept it, and ask for the
            # stream again, since the request did not survive it.
            if vehicle is not None:
                vehicle["reboots"] = vehicle.get("reboots", 0) + 1
            ask_for_stream()
            asked = clock()
        elif last_time is not None and stamp <= last_time:
            # A duplicate, out of order, or a reading the estimator has
            # already predicted past during a gap: using it would integrate
            # the same stretch of time twice. Unless they keep coming: a
            # second of readings, every one behind the time already
            # predicted through, means the two clocks have parted -- a
            # reboot at short uptime, a latency that grew, a simulator
            # running slow -- and waiting for the flight controller's clock
            # to catch up could take for ever. Then its clock is taken back.
            if replay:
                continue
            now = clock()
            if stale_since is None:
                stale_since = now
            if now - stale_since < REBOOT_JUMP:
                continue
            if stamp < last_real and vehicle is not None:
                vehicle["reboots"] = vehicle.get("reboots", 0) + 1
            ask_for_stream()
            asked = now
        elif replay and last_time is not None:
            # A recorded hole is replayed as the gap steps it was live, on
            # the recording's own clock.
            gap = last_time + timeout
            while gap < stamp - 1e-9:
                yield gap, None
                last_time, gap = gap, gap + timeout
        stale_since = None
        last_real = last_time = stamp
        last_reading = last_wall = clock()
        deadline = last_wall + timeout
        yield stamp, reading_from_scaled_imu(msg)


if __name__ == "__main__":
    print("UNIT CONVERSION AND FRAME, ON SYNTHETIC MESSAGES\n")
    print("pymavlink is not required for this check and is not imported.\n")

    class Fake:
        """The nine SCALED_IMU integer fields, and nothing else."""

        def __init__(self, **fields):
            for name in FIELDS:
                setattr(self, name, fields.get(name, 0))

    # A level, stationary vehicle: accelerometer reads +1 g on z in MAVLink's
    # forward-right-down frame, gyro zero, magnetometer pointing north and
    # down at roughly Boulder's inclination.
    level = Fake(zacc=1000, xmag=407, zmag=-914)
    got = reading_from_scaled_imu(level)

    print("  %-10s %12s %12s" % ("channel", "converted", "expected"))
    expected = [0.0, 0.0, G, 0.0, 0.0, 0.0, 0.407, 0.0, -0.914]
    ok = True
    for name, value, want in zip(CHANNELS, got, expected):
        close = abs(value - want) < 2e-3
        ok = ok and close
        print("  %-10s %12.4f %12.4f %s" % (name, value, want,
                                             "" if close else "  <-- off"))
    print("\n  frame agrees with quad_sim (level reads +g on z): %s"
          % ("yes" if ok else "NO"))

    print("\n  magnetometer normalised to unit length: |m| = %.6f"
          % np.linalg.norm(got[6:9]))

    print("\n  Gyro at 1000 mrad/s per axis -> rad/s:",
          reading_from_scaled_imu(Fake(xgyro=1000, ygyro=1000,
                                       zgyro=1000))[3:6])

    # A recorded flight, replayed: a .tlog written here with the traffic a
    # companion link really carries -- a ground station's heartbeat, a
    # gimbal's, another vehicle's, and that vehicle's own readings and
    # attitude -- around the flight controller's. Only the flight
    # controller's may come through, all of them, and the replay must end.
    replay_ok = True
    try:
        from pymavlink import mavutil
    except ImportError:
        print("\n  pymavlink is not installed; the replay check is skipped.")
    else:
        import struct
        import tempfile
        mav = mavutil.mavlink
        armed = mav.MAV_MODE_FLAG_SAFETY_ARMED | mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

        def encoder(system, component):
            return mav.MAVLink(None, srcSystem=system, srcComponent=component)

        fc, other = encoder(1, 1), encoder(2, 1)
        gcs, gimbal = encoder(255, 190), encoder(1, 154)
        records = []

        def put(enc, msg):
            # A .tlog record: the time in microseconds, then the packet.
            stamp_us = len(records) * 1000
            records.append(struct.pack(">Q", stamp_us) + msg.pack(enc))

        put(gcs, gcs.heartbeat_encode(mav.MAV_TYPE_GCS, mav.MAV_AUTOPILOT_INVALID,
                                      0, 0, 0))
        put(other, other.heartbeat_encode(mav.MAV_TYPE_QUADROTOR,
                                          mav.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                          armed, 5, mav.MAV_STATE_ACTIVE))
        put(fc, fc.heartbeat_encode(mav.MAV_TYPE_QUADROTOR,
                                    mav.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                    0, 0, mav.MAV_STATE_STANDBY))
        put(other, other.attitude_encode(0, 1.0, 1.0, 1.0, 0, 0, 0))
        put(fc, fc.attitude_encode(0, 0.1, 0.2, 0.3, 0, 0, 0))
        n_readings = 50
        for k in range(n_readings):
            put(fc, fc.scaled_imu_encode(20 * (k + 1), 0, 0, 1000, k, 0, 0,
                                         407, 0, -914))
            put(other, other.scaled_imu_encode(20 * (k + 1) + 5, 9, 9, 9, 9, 9,
                                               9, 9, 9, 9))
            if k % 10 == 0:
                put(gimbal, gimbal.heartbeat_encode(mav.MAV_TYPE_GIMBAL,
                                                    mav.MAV_AUTOPILOT_GENERIC,
                                                    0, 0, 0))
        path = Path(tempfile.mkdtemp()) / "replay_check.tlog"
        path.write_bytes(b"".join(records))

        initial, vehicle = {}, {}
        got = list(mavlink_source(str(path), initial=initial, vehicle=vehicle,
                                  system=1))
        readings = [r for _, r in got if r is not None]
        gyro_x = [round(r[3] * 1000) for r in readings]
        replay_ok = (len(readings) == n_readings
                     and gyro_x == list(range(n_readings))
                     and abs(initial.get("roll", 0) - 0.1) < 1e-6
                     and vehicle.get("system") == 1
                     and vehicle.get("armed") is False)
        print("\n  replayed %d of %d of the flight controller's readings, none of"
              % (len(readings), n_readings))
        print("  the other vehicle's; started from its attitude (roll %.1f);"
              % initial.get("roll", float("nan")))
        print("  vehicle system %s, armed %s; the replay ended: %s"
              % (vehicle.get("system"), vehicle.get("armed"),
                 "yes" if replay_ok else "NO"))

    # The link's timing, on a virtual clock: a flight controller that streams
    # only once asked, forgets on a reboot, and can run its clock at a
    # different rate from this one, behind a link that can go silent; and a
    # consumer that can stall. Each case is what has broken before.
    timing_ok = True
    try:
        from pymavlink import mavutil
    except ImportError:
        print("\n  pymavlink is not installed; the timing check is skipped.")
    else:
        mav = mavutil.mavlink

        class Msg:
            def __init__(self, kind, **fields):
                self.kind = kind
                self.__dict__.update(fields)

            def get_type(self):
                return self.kind

            def get_srcSystem(self):
                return 1

            def get_srcComponent(self):
                return 1

        class SimLink:
            """A flight controller and the link to it, in virtual time."""

            def __init__(self, rate=1.0, reboot_at=None, boot=2.0,
                         outage=None, lose_asks=0, stall_at=None, stall=0.0):
                self.now, self.rate = 0.0, rate
                self.reboot_at, self.boot = reboot_at, boot
                self.outage, self.lose_asks = outage, lose_asks
                self.stall_at, self.stall = stall_at, stall
                # time_boot_ms counts from power-on, the boot included.
                self.powered_at = self.booted_at = 0.0
                self.streaming = False
                self.next_imu = self.next_beat = 0.0
                self.queue = []
                self.mav = self

            # -- what the source sends -----------------------------------
            def command_long_send(self, system, component, command, conf,
                                  p1, *rest):
                if p1 != mav.MAVLINK_MSG_ID_SCALED_IMU:
                    return
                if self.lose_asks > 0:
                    self.lose_asks -= 1
                    return
                if self.now >= self.booted_at:
                    self.streaming = True

            # -- what it receives ------------------------------------------
            def _emit_until(self, t):
                while min(self.next_imu, self.next_beat) <= t:
                    at = min(self.next_imu, self.next_beat)
                    if self.reboot_at is not None and at >= self.reboot_at:
                        self.powered_at = self.reboot_at
                        self.booted_at = self.reboot_at + self.boot
                        self.streaming, self.reboot_at = False, None
                    up = at >= self.booted_at
                    lost = (self.outage is not None
                            and self.outage[0] <= at < self.outage[1])
                    if at == self.next_beat:
                        if up and not lost:
                            self.queue.append(Msg(
                                "HEARTBEAT", type=mav.MAV_TYPE_QUADROTOR,
                                autopilot=mav.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                base_mode=0, custom_mode=0, system_status=3))
                        self.next_beat += 1.0
                    else:
                        if up and self.streaming and not lost:
                            ms = int(1000 * self.rate * (at - self.powered_at))
                            self.queue.append(Msg(
                                "SCALED_IMU", time_boot_ms=ms,
                                **{f: 0 for f in FIELDS}))
                        self.next_imu += 0.02

            def recv_match(self, type=None, blocking=False, timeout=None):
                end = self.now + (timeout or 0.0) if blocking else self.now
                while True:
                    self._emit_until(self.now)
                    while self.queue:
                        msg = self.queue.pop(0)
                        if type is None or msg.get_type() in type:
                            return msg
                    if self.now >= end:
                        return None
                    self.now = min(end, min(self.next_imu, self.next_beat))

            def clock(self):
                return self.now

        def run(label, seconds, check, **kw):
            sim = SimLink(**kw)
            got = []
            for stamp, reading in mavlink_source(sim, vehicle={}, system=1,
                                                 clock=sim.clock):
                got.append((sim.now, stamp, reading is not None))
                sim.now += 0.002                    # the consumer's step
                if sim.stall_at is not None and sim.now >= sim.stall_at:
                    sim.now += sim.stall
                    sim.stall_at = None
                if sim.now > seconds:
                    break
            good, note = check(got)
            print("  %-46s %s  %s" % (label, note, "" if good else "<-- wrong"))
            return good

        def readings_after(t):
            return lambda got: (sum(1 for w, _, r in got if r and w > t),)

        def resumed(after, within):
            def check(got):
                back = [w for w, _, r in got if r and w > after]
                late = back[0] - after if back else float("nan")
                return (bool(back) and late <= within,
                        "readings again %.2f s after" % late)
            return check

        def stall_check(got):
            gaps = sum(1 for _, _, r in got if not r)
            stamps = [s for _, s, r in got if r]
            rising = all(b > a for a, b in zip(stamps, stamps[1:]))
            return (gaps == 0 and rising and len(stamps) > 400,
                    "%d readings, %d gap steps, in order: %s"
                    % (len(stamps), gaps, rising))

        print("\n  Link timing, on a virtual clock:")
        timing_ok &= run("consumer stalls 2.1 s at 5 s", 10.0, stall_check,
                         stall_at=5.0, stall=2.1)
        timing_ok &= run("first two stream requests lost", 10.0,
                         resumed(0.0, 3.5), lose_asks=2)
        timing_ok &= run("reboot at 3 s of uptime", 15.0,
                         resumed(5.0, 3.0), reboot_at=3.0)
        timing_ok &= run("reboot at 20 s of uptime", 30.0,
                         resumed(22.0, 3.0), reboot_at=20.0)
        timing_ok &= run("FC clock 1% slow, 5 s outage", 20.0,
                         resumed(10.0, 2.0), rate=0.99, outage=(5.0, 10.0))

    print("\nWhat this did not test: a live link. Whether SCALED_IMU arrives at")
    print("the assumed rate, and what happens across a flight-controller")
    print("reboot, are bench questions.")
    sys.exit(0 if ok and replay_ok and timing_ok else 1)

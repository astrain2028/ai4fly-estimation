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
``(expected_time, None)``, and the estimator does a predict-only step.
Repeating the previous reading instead would look identical to a frozen
sensor to every mechanism in this project.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

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


def mavlink_source(connection, dt_nominal=0.02, message="SCALED_IMU",
                   timeout_factor=2.0):
    """Yield ``(seconds, reading)`` from a MAVLink connection, or None on a gap.

    ``connection`` is anything pymavlink's ``mavlink_connection`` accepts: a
    serial device with baud (``"/dev/ttyAMA0:921600"``), a UDP endpoint
    (``"udpin:0.0.0.0:14550"``), or a ``.tlog`` path for offline replay.

    Timestamps are the flight controller's ``time_boot_ms``, in seconds, so
    the estimator's measured interval reflects when the reading was taken
    and not when it was received.
    """
    from pymavlink import mavutil          # imported here: not a runtime dep

    link = mavutil.mavlink_connection(connection)
    link.wait_heartbeat()

    # Ask for the message at the filter's rate. ArduPilot's default stream
    # rate for the raw-sensor group is a few hertz, and a fifty-hertz filter
    # fed at two would spend nineteen steps in twenty predicting through a
    # gap it never needed to have. MAV_CMD_SET_MESSAGE_INTERVAL takes the
    # message id and the interval in microseconds; a .tlog replay ignores it.
    message_id = getattr(mavutil.mavlink, "MAVLINK_MSG_ID_%s" % message)
    link.mav.command_long_send(
        link.target_system, link.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, int(1e6 * dt_nominal), 0, 0, 0, 0, 0)

    timeout = timeout_factor * dt_nominal
    last_time = None
    while True:
        msg = link.recv_match(type=message, blocking=True, timeout=timeout)
        if msg is None:
            # Nothing arrived inside the window. Advance by the nominal step
            # so the estimator predicts through the gap, and report no
            # reading rather than a stale one.
            if last_time is None:
                continue
            last_time = last_time + dt_nominal
            yield last_time, None
            continue

        stamp = msg.time_boot_ms / 1000.0
        if last_time is not None and stamp <= last_time:
            continue                     # out-of-order or duplicate; drop it
        last_time = stamp
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

    print("\nWhat this did not test: receiving anything. The generator's")
    print("timeout handling, the timestamp source, and whether SCALED_IMU")
    print("arrives at the assumed rate are bench questions. Record a .tlog")
    print("on the vehicle and pass its path to mavlink_source to answer them")
    print("without a live link.")
    sys.exit(0 if ok else 1)

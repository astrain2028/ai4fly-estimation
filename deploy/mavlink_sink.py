"""
The estimator's output, sent back to the flight controller.

What goes back, and why these messages
--------------------------------------

The estimator on the companion computer is an advisory monitor. It does not
fly the vehicle, and nothing here feeds the flight controller's own EKF. What
it has to offer is a diagnosis -- how degraded each sensor is, and whether
the filter believes its own uncertainty -- and the useful place for that is
where a pilot and the flight log can see it.

``NAMED_VALUE_FLOAT`` is the message for that. ArduPilot displays named
floats in the ground station and writes them to the dataflash log alongside
everything else, so the health estimates end up time-aligned with the
vehicle's own record without any logging code on this side. Names are limited
to ten characters by the message definition, which is why the health entries
are abbreviated below.

``STATUSTEXT`` carries a short alert when something crosses a threshold: a
device's health estimate passes a level, or NIS runs high for long enough to
mean the filter has stopped trusting itself. Ground stations show these as
messages and can speak them, which is the only channel a pilot is likely to
notice in flight.

Rate
----

Named floats are sent at a fraction of the filter rate. The estimate is
updated fifty times a second and nobody needs six health values fifty times a
second on a telemetry link that also carries everything else. Once a second
for the values and only on change for the alerts is what the defaults do;
both are arguments.

Status
------

Untested against a flight controller, like the source. The self-test drives a
sink and a receiver over a UDP loopback on one machine, which exercises the
message construction and the pymavlink send path end to end and nothing about
a serial link or ArduPilot's handling of the messages.
"""

import time

import numpy as np

# Ten characters at most. The names are the health entries in the order the
# quadcopter's state carries them, then the filter's own consistency figure.
HEALTH_NAMES = ["hb_accel", "hb_gyro", "hb_mag", "hn_accel", "hn_gyro", "hn_mag"]

DEFAULT_PERIOD = 1.0            # seconds between NAMED_VALUE_FLOAT sends
HEALTH_ALERT = 1.0              # default level for a health alert; None disables
NIS_ALERT_FACTOR = 3.0          # NIS above this many times its target, sustained
NIS_ALERT_STEPS = 50            # for this many consecutive updates, raises one


class Sink:
    """Send estimates to a MAVLink endpoint as named floats and alerts."""

    def __init__(self, connection, source_system=1, period=DEFAULT_PERIOD,
                 nis_target=9.0, health_alert=HEALTH_ALERT,
                 health_names=HEALTH_NAMES, clock=time.time):
        from pymavlink import mavutil        # imported here: not a runtime dep

        self.link = mavutil.mavlink_connection(
            connection, source_system=source_system,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
        self.mav = self.link.mav
        self.severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        self.period = period
        self.nis_target = nis_target
        self.health_alert = health_alert     # None: report levels, never alarm
        self.health_names = list(health_names)
        self.clock = clock

        self.last_sent = -np.inf
        self.alerted = set()
        self.nis_high_run = 0
        self.sent = 0

    def _boot_ms(self, timestamp):
        return int(timestamp * 1000) & 0xFFFFFFFF

    def push(self, record, health_states):
        """Consider one estimator record. Sends only when something is due.

        ``record`` is what runtime.Estimator.step returns. Returns the number
        of messages sent this call.
        """
        sent = 0
        now = self.clock()
        stamp = self._boot_ms(record["time"])

        if now - self.last_sent >= self.period:
            mean = record["mean"]
            for name, index in zip(self.health_names, health_states):
                self.mav.named_value_float_send(stamp, name.encode("ascii"),
                                                float(mean[index]))
                sent += 1
            if record["nis"] is not None:
                self.mav.named_value_float_send(stamp, b"nis",
                                                float(record["nis"]))
                sent += 1
            self.mav.named_value_float_send(stamp, b"step_ms",
                                            float(record["ms"]))
            sent += 1
            self.last_sent = now

        # Alerts: once per crossing, not once per step. Whether a health
        # entry's level means anything is the vehicle's business -- see
        # HEALTH_ALERT in deploy/pi_main.py.
        for name, index in zip(self.health_names, health_states):
            if self.health_alert is None:
                break
            level = float(record["mean"][index])
            if level > self.health_alert and name not in self.alerted:
                self._alert("%s degraded, level %.2f" % (name, level))
                self.alerted.add(name)
                sent += 1
            elif level <= self.health_alert * 0.5:
                self.alerted.discard(name)           # re-arm after recovery

        if record["nis"] is not None:
            if record["nis"] > NIS_ALERT_FACTOR * self.nis_target:
                self.nis_high_run += 1
            else:
                self.nis_high_run = 0
            if self.nis_high_run == NIS_ALERT_STEPS:
                self._alert("estimator inconsistent, NIS %.0f for %d steps"
                            % (record["nis"], NIS_ALERT_STEPS))
                sent += 1

        self.sent += sent
        return sent

    def _alert(self, text):
        # STATUSTEXT carries at most 50 characters.
        self.mav.statustext_send(self.severity, text[:50].encode("ascii"))


if __name__ == "__main__":
    import sys
    import threading

    try:
        from pymavlink import mavutil
    except ImportError:
        print("pymavlink is not installed; nothing to test. The estimator")
        print("does not need it -- only this file and pi_main.py do.")
        sys.exit(0)

    print("SINK AND RECEIVER OVER A UDP LOOPBACK\n")
    print("No flight controller: a receiver on this machine plays its part,")
    print("which tests message construction and the send path and nothing")
    print("about a serial link.\n")

    port = 14560
    receiver = mavutil.mavlink_connection("udpin:127.0.0.1:%d" % port)
    got = []

    def listen():
        deadline = time.time() + 5.0
        while time.time() < deadline:
            msg = receiver.recv_match(blocking=True, timeout=0.5)
            if msg is not None:
                got.append(msg)

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    time.sleep(0.2)

    sink = Sink("udpout:127.0.0.1:%d" % port, period=0.0)
    health_states = list(range(6, 12))

    # Three records: healthy, then the gyro's bias estimate climbing past the
    # alert level, then NIS running high.
    base = np.zeros(12)
    records = [
        {"time": 1.00, "mean": base.copy(), "nis": 8.5, "ms": 1.7},
        {"time": 1.02, "mean": base.copy(), "nis": 8.7, "ms": 1.7},
    ]
    records[1]["mean"][7] = 1.4                       # hb_gyro over threshold
    for k in range(NIS_ALERT_STEPS):
        r = {"time": 1.04 + 0.02 * k, "mean": records[1]["mean"].copy(),
             "nis": 40.0, "ms": 1.7}
        records.append(r)

    total = sum(sink.push(r, health_states) for r in records)
    time.sleep(0.5)
    thread.join(timeout=6.0)

    kinds = {}
    for m in got:
        kinds[m.get_type()] = kinds.get(m.get_type(), 0) + 1
    print("  sent %d messages, received %d" % (total, len(got)))
    for kind, n in sorted(kinds.items()):
        print("    %-18s %d" % (kind, n))

    names = sorted({m.name for m in got if m.get_type() == "NAMED_VALUE_FLOAT"})
    texts = [m.text for m in got if m.get_type() == "STATUSTEXT"]
    print("\n  named values seen: %s" % ", ".join(names))
    print("  alerts seen:")
    for t in texts:
        print("    %r" % t)

    ok = (len(got) == total
          and set(HEALTH_NAMES + ["nis", "step_ms"]) <= set(names)
          and len(texts) == 2)
    print("\n  every message arrived, all names present, both alerts fired: %s"
          % ("yes" if ok else "NO"))
    sys.exit(0 if ok else 1)

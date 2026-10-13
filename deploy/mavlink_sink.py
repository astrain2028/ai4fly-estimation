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

When the estimate cannot be used
--------------------------------

deploy/integrity.py says when the estimate is degraded (a device has failed)
or invalid (every device has failed, or the readings have stopped). The sink
reports that as the named value ``integrity`` -- 0, 1, 2 -- and announces
every change: a warning for degraded, a critical alert for invalid, a notice
on recovery.

ReturnHome goes one step further, and only when deploy/pi_main.py is run
with --allow-rtl: on invalid it asks the flight controller to switch to RTL.
It is the one place this project reaches past advising, so it asks rather
than insists, and stands down at the first sign it should:

    only of an ArduPilot multicopter or helicopter, where mode 6 is RTL
    only when it is armed and in the air -- EXTENDED_SYS_STATE's landed
    state, or MAV_STATE_ACTIVE if that has not arrived -- and a heartbeat
    from it is recent. The landed state matters: ArduCopter reports
    MAV_STATE_CRITICAL in flight once any failsafe has latched, a
    warn-only low battery included
    never when it is already in RTL, SmartRTL, Auto RTL or Land
    at most ATTEMPTS requests, each given ACK_TIMEOUT to be acknowledged
    once per flight: if RTL engages and the pilot then switches out of it,
    it is never asked for again until the vehicle has disarmed -- and that
    memory lives across reconnects, since deploy/pi_main.py keeps one
    ReturnHome for the life of the process
    a refusal is announced, not retried -- ArduPilot refuses RTL without a
    position estimate, which is the case in which RTL would not help
    when one of these holds it back while the estimate is invalid, it says
    which, once per flight, so the log shows why nothing was asked

Whether ArduPilot does what is asked is ArduPilot's decision, made on its
own sensors, and it has three IMUs where this process sees one. That is
deliberate: the request is advice with a mode number on it.

Status
------

Untested against a flight controller, like the source. ReturnHome in
particular must be tried in ArduPilot SITL before --allow-rtl goes near a
vehicle; deploy/pi/README.md has the procedure. The self-test drives a
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

# The quadcopter's three devices by channel, in SCALED_IMU order.
QUAD_DEVICES = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}

# ArduCopter's flight-mode numbers, as HEARTBEAT.custom_mode carries them.
COPTER_RTL, COPTER_LAND, COPTER_SMART_RTL, COPTER_AUTO_RTL = 6, 9, 21, 27
ALREADY_SAFE = {COPTER_RTL, COPTER_LAND, COPTER_SMART_RTL, COPTER_AUTO_RTL}

# The vehicle kinds ArduCopter flies, by HEARTBEAT.type. Custom mode 6 is RTL
# only on Copter -- on Plane it is FBWB, on Rover FOLLOW -- so nothing is
# asked of anything else.
COPTER_TYPES = ("MAV_TYPE_QUADROTOR", "MAV_TYPE_HEXAROTOR",
                "MAV_TYPE_OCTOROTOR", "MAV_TYPE_TRICOPTER", "MAV_TYPE_COAXIAL",
                "MAV_TYPE_HELICOPTER", "MAV_TYPE_DODECAROTOR",
                "MAV_TYPE_DECAROTOR")

ACK_TIMEOUT = 2.0          # seconds to wait for COMMAND_ACK before asking again
ATTEMPTS = 3               # requests per flight, at most
HEARTBEAT_TIMEOUT = 3.0    # seconds: an older heartbeat means no one to ask


class Novelty:
    """The epistemic term's novelty ratio, smoothed per device.

    The ratio is 1 where the model's inputs look like its training data and
    grows where they do not. Per step it is spiky -- healthy simulated flights
    touch 30 for a step or two while the health estimates settle -- so what is
    reported and alarmed on is a geometric running mean over about a second:
    an exponential average of the log, which a lone spike cannot move far.
    The highest channel of each device is taken, since one unfamiliar channel
    is enough.

    Numpy only, and shared with deploy/calibrate_novelty.py, so the threshold
    is set on exactly the quantity the vehicle computes.
    """

    def __init__(self, devices, time_constant=1.0, dt=0.02):
        self.devices = dict(devices)
        self.alpha = dt / time_constant
        self.level = {name: 0.0 for name in self.devices}     # log units
        self.steps = 0

    def update(self, doubt):
        """Take one step's per-channel ratio; return the smoothed per device."""
        if doubt is not None:
            doubt = np.maximum(np.asarray(doubt, dtype=float), 1e-12)
            for name, columns in self.devices.items():
                target = float(np.log(doubt[columns].max()))
                self.level[name] += self.alpha * (target - self.level[name])
            self.steps += 1
        return self.current()

    def current(self):
        return {name: float(np.exp(v)) for name, v in self.level.items()}


class Sink:
    """Send estimates to a MAVLink endpoint as named floats and alerts."""

    def __init__(self, connection, source_system=1, period=DEFAULT_PERIOD,
                 nis_target=9.0, health_alert=HEALTH_ALERT,
                 health_names=HEALTH_NAMES, devices=None, novelty_alert=None,
                 novelty_time_constant=1.0, arm_after=100, dt=0.02,
                 clock=time.time):
        from pymavlink import mavutil        # imported here: not a runtime dep

        # A connection string opens a link of the sink's own. An open link is
        # shared -- deploy/pi_main.py passes the one the source reads, which
        # is the only way a udpin link can answer: pymavlink sends from a
        # udpin socket only to peers it has received from.
        if isinstance(connection, str):
            self.link = mavutil.mavlink_connection(
                connection, source_system=source_system,
                source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER)
        else:
            self.link = connection
        self.mavlink = mavutil.mavlink
        self.severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        self.integrity_state = "ok"
        self.period = period
        self.nis_target = nis_target
        self.health_alert = health_alert     # None: report levels, never alarm
        self.health_names = list(health_names)
        self.clock = clock

        # Novelty per device, if the estimator reports it. ``novelty_alert``
        # maps a device to the smoothed level that raises an alert, calibrated
        # on healthy flights by deploy/calibrate_novelty.py; a device missing
        # from it is reported and never alarmed. Nothing alarms in the first
        # ``arm_after`` steps, while the filter settles.
        self.novelty = (Novelty(devices, time_constant=novelty_time_constant,
                                dt=dt) if devices else None)
        self.novelty_alert = dict(novelty_alert or {})
        self.arm_after = arm_after

        self.last_sent = -np.inf
        self.alerted = set()
        self.nis_high_run = 0
        self.sent = 0

    @property
    def mav(self):
        # The link's encoder as it is now: pymavlink replaces it when it
        # finds the other end speaking MAVLink 2, and a copy kept from before
        # would go on sending MAVLink 1.
        return self.link.mav

    def _boot_ms(self, timestamp):
        return int(timestamp * 1000) & 0xFFFFFFFF

    def push(self, record, health_states, integrity=None):
        """Consider one estimator record. Sends only when something is due.

        ``record`` is what runtime.Estimator.step returns, and ``integrity``
        what deploy/integrity.py made of it, if it ran. Returns the number of
        messages sent this call.
        """
        sent = 0
        now = self.clock()
        stamp = self._boot_ms(record["time"])
        novelty = (self.novelty.update(record.get("doubt"))
                   if self.novelty is not None else {})

        # Every change of the estimate's usability is announced at once, at a
        # severity a ground station will speak.
        if integrity is not None and integrity["state"] != self.integrity_state:
            state = integrity["state"]
            if state == "invalid":
                self._alert("ESTIMATE INVALID: %s" % integrity["reason"],
                            self.mavlink.MAV_SEVERITY_CRITICAL)
            elif state == "degraded":
                self._alert("sensor failed: %s" % integrity["reason"])
            else:
                self._alert("sensors recovered",
                            self.mavlink.MAV_SEVERITY_NOTICE)
            self.integrity_state = state
            sent += 1

        if now - self.last_sent >= self.period:
            if integrity is not None:
                self.mav.named_value_float_send(stamp, b"integrity",
                                                float(integrity["level"]))
                sent += 1
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
            for device, level in novelty.items():
                name = ("nov_%s" % device)[:10]
                self.mav.named_value_float_send(stamp, name.encode("ascii"),
                                                level)
                sent += 1
            self.last_sent = now

        # Novelty alerts: the model is being asked about inputs unlike its
        # training data, sustained over about a second. Once per crossing.
        if self.novelty is not None and self.novelty.steps > self.arm_after:
            for device, level in novelty.items():
                limit = self.novelty_alert.get(device)
                if limit is None:
                    continue
                key = "nov_" + device
                if level > limit and key not in self.alerted:
                    self._alert("model outside training: %s x%.0f"
                                % (device, level))
                    self.alerted.add(key)
                    sent += 1
                elif level <= 0.5 * limit:
                    self.alerted.discard(key)

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

    def _alert(self, text, severity=None):
        # STATUSTEXT carries at most 50 characters.
        self.mav.statustext_send(self.severity if severity is None else severity,
                                 text[:50].encode("ascii"))

    def request_mode(self, system, component, mode):
        """Ask the flight controller to switch to ArduPilot custom ``mode``."""
        self.mav.command_long_send(
            system, component, self.mavlink.MAV_CMD_DO_SET_MODE, 0,
            self.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode,
            0, 0, 0, 0, 0)


class ReturnHome:
    """Ask the flight controller to return home when the estimate is invalid.

    Used only with --allow-rtl. ``vehicle`` is the dict deploy/mavlink_source.py
    keeps current from the flight controller alone. ``update`` is called once
    per step with deploy/integrity.py's status and returns what it did, as
    lines for the log. ``sink`` may be replaced between connections; the
    memory of this flight is kept. See the module docstring for every
    condition under which it does nothing.
    """

    def __init__(self, sink, vehicle, mode=COPTER_RTL, ack_timeout=ACK_TIMEOUT,
                 attempts=ATTEMPTS, heartbeat_timeout=HEARTBEAT_TIMEOUT,
                 clock=time.monotonic):
        self.sink = sink
        self.vehicle = vehicle
        self.mode = mode
        self.ack_timeout = ack_timeout
        self.attempts = attempts
        self.heartbeat_timeout = heartbeat_timeout
        self.clock = clock
        self.requests = 0               # every request ever sent, for the log
        self.reboots = None             # the flight controller's, last seen
        self._new_flight()

    def _new_flight(self):
        self.sent = 0                   # requests sent this flight
        self.last_request = None
        self.accepted = False           # the flight controller said yes
        self.engaged = False            # and was then seen in RTL
        self.done = False               # nothing more to ask this flight
        self.blocked = set()            # reasons already reported
        self.saw_safe = False           # in RTL or Land while invalid, unasked

    def _ours(self, ack):
        """Whether an acknowledgement answers this process's request."""
        target = (ack.get("target_system") or 0, ack.get("target_component") or 0)
        if target == (0, 0):
            return True                 # MAVLink 1 carries no target
        return target == (self.sink.mav.srcSystem, self.sink.mav.srcComponent)

    def _hold(self, events, reason):
        """Record once per flight why nothing is being asked."""
        if reason not in self.blocked:
            self.blocked.add(reason)
            events.append("not requesting RTL: %s" % reason)
        return events

    def update(self, status):
        events = []
        v = self.vehicle
        now = self.clock()
        mavlink = self.sink.mavlink

        # A new flight starts at a disarm, or at a reboot -- a disarm this
        # process may not have heard, if the link was down at the time.
        reboots = v.get("reboots")
        if self.reboots is not None and reboots != self.reboots:
            self._new_flight()
        self.reboots = reboots
        armed = v.get("armed")
        if armed is False:
            if (self.sent or self.engaged or self.done or self.blocked
                    or self.saw_safe):
                self._new_flight()      # disarmed: the next flight starts clean
            return events

        mode = v.get("mode")
        if mode == self.mode and (self.sent or self.accepted):
            self.engaged = True         # RTL is on, and this asked for it
        elif (self.engaged or self.saw_safe) and not self.done and (
                mode not in ALREADY_SAFE):
            # RTL or a landing was under way -- this process's or not -- and
            # something switched it off: almost certainly the pilot. Their
            # call stands for the rest of the flight.
            self.done = True
            events.append("%s left for mode %s; not asking again this flight"
                          % ("RTL" if self.engaged else "a return or landing",
                             mode))

        # Only an acknowledgement of this request counts: the right command,
        # after it was sent, addressed to this process.
        ack = v.get("acks", {}).get(mavlink.MAV_CMD_DO_SET_MODE)
        if (ack is not None and self.last_request is not None
                and ack["time"] >= self.last_request and self._ours(ack)
                and not (self.accepted or self.done)):
            if ack["result"] == mavlink.MAV_RESULT_ACCEPTED:
                self.accepted = True
                events.append("RTL request accepted")
            else:
                self.sink._alert("RTL request refused (result %d)"
                                 % ack["result"],
                                 mavlink.MAV_SEVERITY_CRITICAL)
                events.append("RTL request refused, result %d" % ack["result"])
                self.done = True

        if (self.done or self.engaged or self.accepted
                or status["state"] != "invalid" or not armed):
            return events

        copters = [getattr(mavlink, t) for t in COPTER_TYPES]
        if (v.get("autopilot") != mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
                or v.get("type") not in copters):
            return self._hold(events, "not an ArduCopter vehicle (type %s)"
                              % v.get("type"))
        if mode in ALREADY_SAFE:
            self.saw_safe = True
            return self._hold(events, "already in mode %s" % mode)
        landed = v.get("landed")
        flying = (landed in (mavlink.MAV_LANDED_STATE_IN_AIR,
                             mavlink.MAV_LANDED_STATE_TAKEOFF)
                  if landed is not None
                  else v.get("status") == mavlink.MAV_STATE_ACTIVE)
        if not flying:
            return self._hold(events, "not in the air")
        beat = v.get("heartbeat")
        if beat is None or now - beat > self.heartbeat_timeout:
            return self._hold(events, "no heartbeat from the flight "
                              "controller")
        if (self.last_request is not None
                and now - self.last_request < self.ack_timeout):
            return events               # still waiting on the last request
        if self.sent >= self.attempts:
            self.sink._alert("RTL request unanswered",
                             mavlink.MAV_SEVERITY_CRITICAL)
            events.append("RTL request unanswered after %d tries" % self.sent)
            self.done = True
            return events

        self.sink.request_mode(v.get("system", 1), v.get("component", 1),
                               self.mode)
        if self.sent == 0:
            self.sink._alert("estimate invalid: requesting RTL",
                             mavlink.MAV_SEVERITY_CRITICAL)
        self.sent += 1
        self.requests += 1
        self.last_request = now
        events.append("RTL requested (%d of %d): %s"
                      % (self.sent, self.attempts, status["reason"]))
        return events


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

    # Novelty, on the same sink (two udpout links to one port collide on
    # Windows): reporting per device switched on, named values sent once so
    # the stream stays small. Ordinary inputs for two seconds, then the
    # accelerometer's channels at 50 times their training level; the smoothed
    # level has to climb past the alert level of 5 and fire exactly once.
    sink.novelty = Novelty(QUAD_DEVICES)
    sink.novelty_alert = {"accel": 5.0}
    sink.arm_after = 50
    sink.period, sink.last_sent = 1e9, -np.inf
    for k in range(250):
        doubt = np.ones(9)
        if k >= 100:
            doubt[0:3] = 50.0
        total += sink.push({"time": 3.0 + 0.02 * k, "mean": base.copy(),
                            "nis": 8.5, "ms": 1.7, "doubt": doubt},
                           health_states)
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

    novelty_alerts = [t for t in texts if t.startswith("model outside")]
    ok = (len(got) == total
          and set(HEALTH_NAMES + ["nis", "step_ms", "nov_accel", "nov_gyro",
                                  "nov_mag"]) <= set(names)
          and len(texts) == 3 and len(novelty_alerts) == 1
          and "accel" in novelty_alerts[0])
    print("\n  every message arrived, all names present, health, NIS and")
    print("  novelty alerts each fired once: %s" % ("yes" if ok else "NO"))

    print("\nRETURN HOME, CONDITION BY CONDITION\n")
    link = mavutil.mavlink
    US = (1, link.MAV_COMP_ID_ONBOARD_COMPUTER)     # this process, on the link

    class Recorder:
        """A sink that records instead of sending."""
        mavlink = link

        class mav:
            srcSystem, srcComponent = US

        def __init__(self):
            self.requests, self.alerts = [], []

        def request_mode(self, system, component, mode):
            self.requests.append((system, component, mode))

        def _alert(self, text, severity=None):
            self.alerts.append(text)

    def case(label, steps, expect_requests, expect_event=None, armed=True,
             status=link.MAV_STATE_ACTIVE, landed=link.MAV_LANDED_STATE_IN_AIR,
             mode=5, kind=link.MAV_TYPE_QUADROTOR, beat=True, reply=None,
             reply_to=US, switches=None, changes=None, home=None,
             vehicle=None):
        """Step ReturnHome through a flight that goes invalid at step 10.

        ``reply`` is the flight controller's acknowledgement of the first
        request, addressed to ``reply_to``; ``switches`` is whether it then
        switches to RTL (by default, when it accepted a request addressed
        to this process). ``changes`` maps a step to vehicle-state changes
        -- the pilot's, say. Returns whether the requests, their target and
        the expected event came out right, and the objects for a follow-on.
        """
        clock = {"now": 100.0}
        if vehicle is None:
            vehicle = {}
        vehicle.update(armed=armed, status=status, landed=landed, mode=mode,
                       autopilot=link.MAV_AUTOPILOT_ARDUPILOTMEGA, type=kind,
                       heartbeat=100.0 if beat else None, system=1,
                       component=1, acks={})
        vehicle.setdefault("reboots", 0)
        if switches is None:
            switches = reply == link.MAV_RESULT_ACCEPTED and reply_to == US
        recorder = Recorder()
        if home is None:
            home = ReturnHome(recorder, vehicle, clock=lambda: clock["now"])
        else:
            home.sink, home.clock = recorder, (lambda: clock["now"])
        events, answered = [], False
        for k in range(steps):
            clock["now"] += 0.02
            if beat:
                vehicle["heartbeat"] = clock["now"]
            vehicle.update((changes or {}).get(k, {}))
            state = "invalid" if k >= 10 else "ok"
            events += home.update({"state": state,
                                   "reason": "every sensor failed"})
            if recorder.requests and not answered:
                answered = True
                if reply is not None:
                    vehicle["acks"][link.MAV_CMD_DO_SET_MODE] = {
                        "result": reply, "target_system": reply_to[0],
                        "target_component": reply_to[1], "time": clock["now"]}
                if switches:
                    vehicle["mode"] = COPTER_RTL
        good = (len(recorder.requests) == expect_requests
                and all(r == (1, 1, COPTER_RTL) for r in recorder.requests))
        if expect_event is not None:
            good = good and any(expect_event in e for e in events)
        print("  %-48s requests %d  %s" % (label, len(recorder.requests),
                                           "" if good else
                                           "<-- expected %d%s" % (
                                               expect_requests,
                                               ", event '%s'" % expect_event
                                               if expect_event else "")))
        return good, home, vehicle

    OK, DENIED = link.MAV_RESULT_ACCEPTED, link.MAV_RESULT_DENIED
    pilot = {200: {"mode": 5}}                  # the pilot takes it back
    checks = [
        case("armed, flying, invalid, accepted", 300, 1, "accepted",
             reply=OK)[0],
        case("accepted, then the pilot leaves RTL", 600, 1, "RTL left",
             reply=OK, changes=pilot)[0],
        case("no acknowledgement, RTL on, the pilot leaves", 600, 1,
             "RTL left", switches=True, changes=pilot)[0],
        case("accepted as a MAVLink 1 ack (no target)", 300, 1, "accepted",
             reply=OK, reply_to=(0, 0))[0],
        case("refused", 300, 1, "refused", reply=DENIED)[0],
        case("never acknowledged", 1000, ATTEMPTS, "unanswered")[0],
        case("acknowledged to a ground station", 1000, ATTEMPTS, "unanswered",
             reply=OK, reply_to=(255, 190))[0],
        case("acknowledged to another component", 1000, ATTEMPTS,
             "unanswered", reply=OK, reply_to=(1, 0))[0],
        case("in the air with a failsafe latched (CRITICAL)", 300, 1,
             reply=OK, status=link.MAV_STATE_CRITICAL)[0],
        case("climbing out in a takeoff", 300, 1, reply=OK,
             landed=link.MAV_LANDED_STATE_TAKEOFF)[0],
        case("no landed state yet, MAV_STATE_ACTIVE", 300, 1, reply=OK,
             landed=None)[0],
        case("disarmed", 300, 0, armed=False)[0],
        case("armed, not yet heard from", 300, 0, armed=None)[0],
        case("armed on the ground", 300, 0, "not in the air",
             landed=link.MAV_LANDED_STATE_ON_GROUND)[0],
        case("already landing", 300, 0, "already in mode",
             mode=COPTER_LAND)[0],
        case("already landing, then the pilot takes over", 600, 0,
             "landing left", mode=COPTER_LAND, changes=pilot)[0],
        case("already in Auto RTL", 300, 0, "already in mode",
             mode=COPTER_AUTO_RTL)[0],
        case("already in RTL", 300, 0, mode=COPTER_RTL)[0],
        case("a fixed-wing vehicle", 300, 0, "not an ArduCopter",
             kind=link.MAV_TYPE_FIXED_WING)[0],
        case("no heartbeat from the flight controller", 300, 0,
             "no heartbeat", beat=False)[0],
    ]
    # The pilot's override has to outlive a reconnect: the same ReturnHome,
    # a new sink, the same flight. A disarm or a reboot starts afresh.
    good, home, vehicle = case("pilot leaves RTL, then a reconnect", 600, 1,
                               "RTL left", reply=OK, changes=pilot)
    checks.append(good)
    checks.append(case("  the same flight after the reconnect", 300, 0,
                       home=home, vehicle=vehicle)[0])
    vehicle["armed"] = False
    home.update({"state": "ok", "reason": ""})        # the disarm, seen
    checks.append(case("  the next flight, after a disarm", 300, 1,
                       reply=OK, home=home, vehicle=vehicle)[0])
    good, home, vehicle = case("pilot leaves RTL, link down until", 600, 1,
                               "RTL left", reply=OK, changes=pilot)
    checks.append(good)
    vehicle["reboots"] += 1                           # the disarm, unseen
    checks.append(case("  the next flight, after a reboot", 300, 1,
                       reply=OK, home=home, vehicle=vehicle)[0])
    rtl_ok = all(checks)
    print("\n  each condition asked exactly as often as it should, of the")
    print("  flight controller, and said why when it held back: %s"
          % ("yes" if rtl_ok else "NO"))
    sys.exit(0 if ok and rtl_ok else 1)

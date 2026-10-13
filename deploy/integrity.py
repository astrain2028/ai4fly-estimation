"""
Whether the estimate can be used at all.

Everything else in this project assumes some sensors are working. The health
states find a broken sensor by letting the others outvote it; the residual
term widens the noise on a channel whose misses grow; the novelty ratio says
when the model's inputs are unfamiliar. Each of those needs a healthy
majority to measure against. When every sensor has failed there is no
majority, and the filter does not stop: it keeps producing an attitude, with
a covariance, from readings that carry no information. Something outside the
filter has to say so.

Two tiers
---------

**Failed** is decided without the learned model. A device has failed when its
readings carry no information, and that can be checked from the readings
alone:

    frozen          every axis has repeated its value exactly for
                    FROZEN_WINDOW readings -- a stuck or dead part, or a
                    driver resending its last sample
    not finite      NaN or infinity
    out of range    a value beyond anything the part can report
    no field        the magnetometer's vector has collapsed to nothing,
                    which is what a dead magnetometer reports once the
                    source normalises it

No threshold here depends on the model or on the simulator. That is the
point of the tier: on the first real flights the model-based signals are
expected to be high everywhere, because the real vehicle is not the
simulated one, and a check that could fire on that gap must not be the one
that sends the vehicle home.

**Suspect** is decided with the model, and is reported but never acted on:

    novelty         the epistemic term's smoothed novelty is above its
                    calibrated alert level (deploy/calibrate_novelty.py)
    untrusted       the filter's residual term has had to add far more noise
                    to the device than the model predicted -- it has stopped
                    listening to it. "Far more" is per device, set on healthy
                    flights by deploy/check_integrity.py: the gyro's residual
                    runs to hundreds of times its predicted noise on healthy
                    flights, because the model cannot know a gyro's per-run
                    bias, and the accelerometer's to a few
    health          a health entry for the device is past the highest level
                    in the training data

A sensor that is merely noisy, biased or miscalibrated still carries
information, and handling it is the filter's job; at most it is suspect.

One known limit. deploy/runtime.py refuses a whole reading if any channel in
it is not finite, so a single device returning NaN stops every update, not
only its own, and the estimate coasts on the motion model while this
reports the device failed and the state degraded. Over MAVLink it cannot
happen -- SCALED_IMU carries integers -- but a direct sensor driver, like
the I2C bench source, must hand over None for a bad sample rather than NaN,
or the filter must learn to update on the finite channels alone.

The system's state
------------------

    invalid     no reading for STREAM_TIMEOUT seconds, or the estimate itself
                has gone non-finite, or every device has failed for HOLD
                seconds. The estimate should not be used.
    degraded    at least one device has failed for HOLD seconds. The others
                carry on, and the health states are not built for a device
                that has stopped reporting, so it is worth a warning.
    ok          otherwise.

deploy/pi_main.py reports the state, alerts on every change, and -- only with
--allow-rtl -- asks the flight controller to return home when it becomes
invalid. deploy/check_integrity.py runs it over simulated flights in which
every sensor fails, and over the flights in which it must not fire.

Numpy only, and shared by the simulation check and the vehicle, so what is
tested is what flies.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from loader import load_module

FROZEN_WINDOW = 25         # identical readings in a row: half a second at 50 Hz
BAD_STEPS = 5              # consecutive non-finite or impossible readings
HOLD = 1.0                 # seconds a failure must last to change the state
STREAM_TIMEOUT = 2.0       # seconds with no reading at all

# Beyond anything the part can report. SCALED_IMU carries 16-bit milli-units,
# so over MAVLink these cannot be exceeded at all; they are for a direct
# sensor driver handing over garbage.
LIMITS = {"accel": 32 * 9.80665, "gyro": 35.0}
FIELD_FLOOR = 0.5          # a unit field vector shorter than this is no field

# Suspect tier.
CEILING = 3.0              # the highest health level in either training set
COLLAPSE_MARGIN = 1.5      # times the healthy peak, per device
SOFT_TIME_CONSTANT = 1.0   # seconds of smoothing on the residual ratio

STATES = ("ok", "degraded", "invalid")

QUAD_HEALTH = {"accel": [6, 9], "gyro": [7, 10], "mag": [8, 11]}


def _novelty_class():
    sink = load_module(HERE / "mavlink_sink.py", "integrity_sink")
    return sink.Novelty


class Integrity:
    """Per-step verdict on whether the sensors, and so the estimate, are usable.

    ``devices`` maps a device name to its channel indices in the reading.
    ``health`` maps a device to its health-state indices, for the suspect
    tier; ``novelty_alert`` to the smoothed novelty level that makes it
    suspect; ``collapse`` to the residual ratio that does. Any may be
    omitted, and the tier simply has less to go on.
    """

    def __init__(self, devices, dt=0.02, health=None, novelty_alert=None,
                 novelty_time_constant=1.0, frozen_window=FROZEN_WINDOW,
                 hold=HOLD, stream_timeout=STREAM_TIMEOUT, collapse=None,
                 arm_after=100):
        self.devices = {name: list(cols) for name, cols in devices.items()}
        self.dt = float(dt)
        self.health = dict(health or {})
        self.novelty_alert = dict(novelty_alert or {})
        self.frozen_window = int(frozen_window)
        self.hold = float(hold)
        self.stream_timeout = float(stream_timeout)
        self.collapse = dict(collapse or {})
        self.arm_after = int(arm_after)

        self.novelty = None
        if self.novelty_alert:
            self.novelty = _novelty_class()(self.devices,
                                            time_constant=novelty_time_constant,
                                            dt=self.dt)
        self.reset()

    def reset(self):
        self.previous = {name: None for name in self.devices}
        self.same = {name: 0 for name in self.devices}
        self.bad = {name: 0 for name in self.devices}
        self.failed_since = {name: None for name in self.devices}
        self.ratio = {name: 1.0 for name in self.devices}
        self.last_update = None         # time of the last reading
        self.last_step = None           # time of the last step, gap or not
        self.all_failed_since = None
        self.steps = 0
        self.state = "ok"

    # -- the failed tier: readings alone ------------------------------------

    def _reading_bad(self, name, values):
        """Why this device's reading is impossible, or None."""
        if not np.all(np.isfinite(values)):
            return "not finite"
        limit = LIMITS.get(name)
        if limit is not None and np.any(np.abs(values) > limit):
            return "out of range"
        if name == "mag" and np.linalg.norm(values) < FIELD_FLOOR:
            return "no field"
        return None

    def _device_failure(self, name, values):
        """Update one device's counters from a reading; return why it failed."""
        reason = self._reading_bad(name, values)
        self.bad[name] = self.bad[name] + 1 if reason else 0

        previous = self.previous[name]
        if (previous is not None and reason is None
                and np.array_equal(values, previous)):
            self.same[name] += 1
        else:
            self.same[name] = 0
        self.previous[name] = None if reason else values.copy()

        if self.bad[name] >= BAD_STEPS:
            return reason
        if self.same[name] >= self.frozen_window - 1:
            return "frozen"
        return None

    # -- the suspect tier: the model's own signals --------------------------

    def _suspect(self, record):
        found = {}
        doubt = record.get("doubt")
        if self.novelty is not None:
            levels = self.novelty.update(doubt)
            if self.steps > self.arm_after:
                for name, level in levels.items():
                    limit = self.novelty_alert.get(name)
                    if limit is not None and level > limit:
                        found[name] = "outside training x%.0f" % level

        parts = record.get("parts")
        if parts is not None and record.get("updated"):
            aleatoric, epistemic, unmodelled = (np.asarray(p, float)
                                                for p in parts[:3])
            ratio = unmodelled / np.maximum(aleatoric + epistemic, 1e-300)
            alpha = self.dt / SOFT_TIME_CONSTANT
            for name, cols in self.devices.items():
                self.ratio[name] += alpha * (float(ratio[cols].max())
                                             - self.ratio[name])
                limit = self.collapse.get(name)
                if (limit is not None and self.steps > self.arm_after
                        and self.ratio[name] > limit):
                    found.setdefault(name, "untrusted by filter x%.0f"
                                     % self.ratio[name])

        mean = record.get("mean")
        if mean is not None:
            mean = np.asarray(mean)
            for name, index in self.health.items():
                if max(index) >= len(mean):
                    continue                # no health states: analytic model
                level = float(np.max(mean[index]))
                if level > CEILING:
                    found.setdefault(name, "health %.1f beyond training"
                                     % level)
        return found

    # -- one step ------------------------------------------------------------

    def update(self, record, reading):
        """Take one estimator record and the raw reading it was given.

        ``reading`` is what the source produced -- None for a gap -- before
        the estimator saw it, since a reading the estimator refused is
        exactly what this has to see.
        """
        self.steps += 1
        now = float(record["time"])
        if self.last_update is None:
            self.last_update = now          # the first step
        elif self.last_step is not None and now < self.last_step:
            # The flight controller rebooted and its clock restarted. Every
            # time kept here moves onto the new clock, so a failure that
            # carries on through the reboot keeps counting rather than
            # reading as a failure that will start in an hour. The shift is
            # from the last step, gap steps included: the time a failure
            # already stood through a gap has been counted, and stays so.
            shift = now - self.last_step
            self.failed_since = {name: None if since is None else since + shift
                                 for name, since in self.failed_since.items()}
            if self.all_failed_since is not None:
                self.all_failed_since += shift
            self.last_update += shift
        self.last_step = now

        failed = {}
        if reading is not None:
            self.last_update = now
            reading = np.asarray(reading, dtype=float)
            for name, cols in self.devices.items():
                reason = self._device_failure(name, reading[cols])
                if reason is None:
                    self.failed_since[name] = None
                    continue
                if self.failed_since[name] is None:
                    self.failed_since[name] = now
                if now - self.failed_since[name] >= self.hold:
                    failed[name] = reason
        else:
            # A gap changes nothing about a device; it is the stream's
            # failure, judged below. Failures already standing still stand.
            for name in self.devices:
                since = self.failed_since[name]
                if since is not None and now - since >= self.hold:
                    failed[name] = "frozen" if self.same[name] else "bad"

        suspect = self._suspect(record)

        if len(failed) == len(self.devices):
            if self.all_failed_since is None:
                self.all_failed_since = now
        else:
            self.all_failed_since = None

        silent = now - self.last_update
        diverged = not (np.all(np.isfinite(record["mean"]))
                        and np.all(np.isfinite(record["var"])))

        if silent >= self.stream_timeout:
            state, reason = "invalid", "no sensor data for %.1f s" % silent
        elif diverged:
            state, reason = "invalid", "estimate not finite"
        elif self.all_failed_since is not None:
            state, reason = "invalid", "every sensor failed"
        elif failed:
            state, reason = "degraded", ", ".join(
                "%s %s" % kv for kv in sorted(failed.items()))
        else:
            state, reason = "ok", ""

        changed = state != self.state
        self.state = state
        return {"state": state, "level": STATES.index(state),
                "reason": reason, "failed": failed, "suspect": suspect,
                "changed": changed}


if __name__ == "__main__":
    print("INTEGRITY MONITOR ON SYNTHETIC READINGS\n")
    devices = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}
    rng = np.random.default_rng(0)
    dt = 0.02

    def healthy():
        accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.1, 3)
        gyro = rng.normal(0, 0.01, 3)
        mag = np.array([0.41, 0.0, -0.91]) + rng.normal(0, 0.02, 3)
        return np.concatenate([accel, gyro, mag / np.linalg.norm(mag)])

    def record(t, mean_ok=True):
        mean = np.zeros(12) if mean_ok else np.full(12, np.nan)
        return {"time": t, "mean": mean, "var": np.ones(12),
                "updated": True, "doubt": None, "parts": None}

    def rebooting(k):
        # The flight controller's clock: restarts at 0.5 s at step 200.
        return k * dt if k < 200 else 0.5 + (k - 200) * dt

    def scenario(label, make, steps=400, expect=None, clock=lambda k: k * dt,
                 after=None):
        """Run one case; it must end in ``expect`` -- and, given ``after``,
        hold it on every step from there, not only the last."""
        monitor = Integrity(devices, dt=dt)
        states, first = [], None
        for k in range(steps):
            reading, ok = make(k)
            status = monitor.update(record(clock(k), ok), reading)
            states.append(status["state"])
            if first is None and status["state"] == "invalid":
                first = k * dt
        final = states[-1]
        good = final == expect and (after is None or all(
            s == expect for s in states[after:]))
        print("  %-38s final %-8s first invalid %s  %s"
              % (label, final, "-" if first is None else "%.2f s" % first,
                 "" if good else "<-- expected %s" % expect))
        return good

    frozen = healthy()
    ok = True
    ok &= scenario("healthy", lambda k: (healthy(), True), expect="ok")
    ok &= scenario("gyro frozen from 2 s",
                   lambda k: (np.concatenate([healthy()[:3], frozen[3:6],
                                              healthy()[6:]])
                              if k >= 100 else healthy(), True),
                   expect="degraded")
    ok &= scenario("every device frozen from 2 s",
                   lambda k: (frozen if k >= 100 else healthy(), True),
                   expect="invalid")
    ok &= scenario("every device NaN from 2 s",
                   lambda k: (np.full(9, np.nan) if k >= 100 else healthy(),
                              True), expect="invalid")
    ok &= scenario("no readings from 2 s",
                   lambda k: (None if k >= 100 else healthy(), True),
                   expect="invalid")
    ok &= scenario("estimate goes non-finite at 2 s",
                   lambda k: (healthy(), k < 100), expect="invalid")
    ok &= scenario("one dropped reading in four",
                   lambda k: (None if k % 4 == 0 else healthy(), True),
                   expect="ok")
    ok &= scenario("every device frozen, reboot at 4 s",
                   lambda k: (frozen if k >= 100 else healthy(), True),
                   expect="invalid", clock=rebooting, after=200)
    ok &= scenario("magnetometer dead, reboot at 4 s",
                   lambda k: (np.concatenate([healthy()[:6], np.zeros(3)])
                              if k >= 100 else healthy(), True),
                   expect="degraded", clock=rebooting, after=200)
    ok &= scenario("frozen, a gap, then a reboot",
                   lambda k: (None if 160 <= k < 200
                              else frozen if k >= 100 else healthy(), True),
                   expect="invalid", clock=rebooting, after=200)
    print("\n  every scenario ended where it should: %s"
          % ("yes" if ok else "NO"))
    sys.exit(0 if ok else 1)

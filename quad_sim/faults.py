"""
Things that go wrong with an IMU or a magnetometer.

Kept separate from sensors.py for the same reason the ground robot keeps them
apart: that file is the healthy vehicle and should stay that way, so "healthy"
is never a flag that might get set wrong. It is simply what you get when this
file is not involved.

FAULTS HAPPEN TO DEVICES, NOT TO AXES

A three-axis accelerometer is one part. It warms up, loses calibration, comes
loose from its mount, or sits in a vibration mode -- and when it does, all
three of its axes go together. So a fault here degrades a whole device, and
health is carried per device rather than per channel: three devices, two modes
each, six health states.

That is the same count the ground robot ended up with, by a different route.
There it was three sensors times two modes; here it is three devices times two.

WHY BIAS IS COMMON-MODE

A bias could be modelled as an arbitrary three-vector, drawn per run. That
would be more realistic and would break the experiment, because a model given
only a severity cannot predict a direction nobody told it about. It would
learn the spread and nothing else, and the first-moment half of the
complementarity story would quietly stop being testable.

So a bias shifts every axis of the device by the same amount. That is a real
failure mode -- a temperature-dependent offset affects a whole part similarly
-- and it keeps the fault predictable from its label, which is what makes the
learned correction meaningful rather than impossible.

SEVERITY MEANS THE SAME THING ON EVERY DEVICE

Severity 1 is trouble about the size of that device's own healthy noise. An
accelerometer's noise is 0.21 m/s^2 and a magnetometer's is 0.018 unitless, so
one severity number has to mean something comparable on both. That is what the
reference scales below are for, and it is why severity can be swept
continuously rather than each device having its own units.
"""

import numpy as np

CHANNELS = ["accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z"]

DEVICES = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}

# Each device's healthy spread, used to turn a severity into real units. The
# accelerometer figure is its average over a run rather than its floor, since
# its noise grows with body rate and the floor would understate it.
REFERENCE = {"accel": 0.214, "gyro": 0.006, "mag": 0.018}


def bias(values, severity, scale, rng, dt):
    """Every axis of the device reads high by the same fixed amount."""
    return values + (severity * scale)[:, None]


def drift(values, severity, scale, rng, dt):
    """The offset starts at zero and grows across the run."""
    ramp = np.linspace(0.0, 1.0, len(values))
    return values + (severity * scale * ramp)[:, None]


def noise_inflation(values, severity, scale, rng, dt):
    """Still centred, much noisier.

    The mode a model predicting only a mean cannot represent: nothing about
    the expected reading has changed, so there is no first moment to correct.
    """
    spread = (severity * scale)[:, None] * np.ones((1, values.shape[1]))
    return values + rng.normal(0.0, 1.0, size=values.shape) * spread


def scale_error(values, severity, scale, rng, dt):
    """The device reads a fixed percentage off, so error grows with signal.

    A magnetometer with the wrong soft-iron calibration looks like this, and
    so does an accelerometer whose sensitivity has drifted.
    """
    return values * (1.0 + 0.2 * severity)[:, None]


def stuck(values, severity, scale, rng, dt):
    """The device freezes and repeats its last reading.

    Severity is the fraction of the run that is frozen, counting from the end,
    so it takes a value in [0, 1] rather than an amount.
    """
    out = values.copy()
    n = len(values)
    frozen = int(float(np.max(severity)) * n)
    if frozen > 0:
        out[n - frozen:] = values[n - frozen - 1]
    return out


def dropout(values, severity, scale, rng, dt):
    """Samples are lost at random and the previous one is held."""
    out = values.copy()
    lost = rng.random(len(values)) < float(np.max(severity))
    for k in range(1, len(values)):
        if lost[k]:
            out[k] = out[k - 1]
    return out


MODES = {"bias": bias, "drift": drift, "noise_inflation": noise_inflation,
         "scale_error": scale_error, "stuck": stuck, "dropout": dropout}

# Which moment of the measurement distribution each mode disturbs. This is the
# whole taxonomy the ground robot's results turn on: a first-moment fault moves
# the expected reading and is visible to an update that reads innovation
# direction; a second-moment fault leaves the mean alone and is only visible to
# something reading magnitude.
MOMENT = {"bias": "first", "drift": "first", "scale_error": "first",
          "noise_inflation": "second", "stuck": "neither",
          "dropout": "neither"}

TRAINED_ON = ["bias", "noise_inflation"]


def apply_fault(readings, device, mode, severity, seed=0, dt=0.02):
    """Return a copy of `readings` with one device degraded.

    readings -- the dict read_sensors gave back
    device   -- "accel", "gyro" or "mag"
    mode     -- a key of MODES
    severity -- one number for the run, or one per sample for a fault that
                arrives or worsens partway through. stuck and dropout read
                severity as a fraction of the run and take a scalar only.
    """
    if device not in DEVICES:
        raise ValueError("unknown device: %s" % device)
    if mode not in MODES:
        raise ValueError("unknown mode: %s" % mode)

    columns = DEVICES[device]
    n = len(readings[CHANNELS[columns[0]]])
    severity = np.asarray(severity, dtype=float)
    if severity.ndim == 0:
        severity = np.full(n, float(severity))
    if mode in ("stuck", "dropout") and len(np.unique(severity)) > 1:
        raise ValueError("%s takes a single severity, not one per sample"
                         % mode)

    out = dict(readings)
    if np.any(severity > 0):
        rng = np.random.default_rng(seed)
        block = np.column_stack([readings[CHANNELS[c]] for c in columns])
        broken = MODES[mode](block, severity, REFERENCE[device], rng, dt)
        for j, c in enumerate(columns):
            out[CHANNELS[c]] = broken[:, j]

    out["fault_device"] = device
    out["fault_mode"] = mode
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from sensors import read_sensors
    from trajectories import DT, random_run

    run = random_run(0)
    healthy = read_sensors(run, seed=0)

    print("Each mode on the accelerometer, severity 1.0")
    print("(that device's healthy spread is %.3f m/s^2)\n"
          % REFERENCE["accel"])
    print("  %-16s %9s %12s %14s" % ("mode", "moment", "mean shift",
                                     "extra spread"))
    print("  " + "-" * 54)

    base = np.column_stack([healthy[c] for c in CHANNELS[:3]])
    for name in MODES:
        severity = 0.4 if name in ("stuck", "dropout") else 1.0
        broken = apply_fault(healthy, "accel", name, severity, seed=0, dt=DT)
        got = np.column_stack([broken[c] for c in CHANNELS[:3]])
        change = got - base
        print("  %-16s %9s %12.4f %14.4f"
              % (name, MOMENT[name], change.mean(), change.std()))

    print("  " + "-" * 54)
    print("\n  bias moves the mean and not the spread. noise_inflation does")
    print("  the opposite -- that is the one no mean-only model can express,")
    print("  and the reason there are two health levels per device.")

    print("\n\nSeverity is continuous, not a switch:\n")
    print("  %-10s %14s" % ("severity", "mean shift"))
    for s in [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]:
        broken = apply_fault(healthy, "mag", "bias", s, seed=0, dt=DT)
        got = np.column_stack([broken[c] for c in CHANNELS[6:]])
        was = np.column_stack([healthy[c] for c in CHANNELS[6:]])
        print("  %-10.2f %14.5f" % (s, (got - was).mean()))

    print("\n\nA per-sample severity works too, for a fault that arrives:\n")
    profile = np.zeros(len(run["roll"]))
    profile[len(profile) // 2:] = 2.0
    broken = apply_fault(healthy, "gyro", "bias", profile, seed=1, dt=DT)
    got = np.column_stack([broken[c] for c in CHANNELS[3:6]])
    was = np.column_stack([healthy[c] for c in CHANNELS[3:6]])
    half = len(profile) // 2
    print("  first half  shift %.5f  (should be 0)"
          % (got - was)[:half].mean())
    print("  second half shift %.5f  (should be %.5f)"
          % ((got - was)[half:].mean(), 2.0 * REFERENCE["gyro"]))

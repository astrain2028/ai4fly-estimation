"""
Things that go wrong with a sensor.

Kept separate from sensors.py on purpose. That file is the healthy robot and
should stay that way, so "healthy" is never an option flag that might get set
wrong -- it is simply what you get when this file is not involved.

HOW SEVERITY WORKS

Every fault takes a severity, a plain number where 0 means nothing is wrong.
A severity of 1 makes the fault about the size of that sensor's own healthy
noise. That way one severity number means a comparable amount of trouble on
an encoder and on a gyro, even though their readings are on very different
scales, and severity can be swept continuously rather than a sensor being
either fine or broken.

Faults are applied to readings after the fact, so any run can be replayed
healthy and faulted with everything else held identical.
"""

import numpy as np

CHANNELS = ["left_encoder", "right_encoder", "gyro"]

# Each channel's healthy spread, used to turn a severity into real units.
# These are the same numbers the filter's hand-tuned R uses.
REFERENCE = {
    "left_encoder": 0.1805,
    "right_encoder": 0.1708,
    "gyro": 0.00799,
}


def bias(values, severity, scale, rng, dt):
    """The sensor reads consistently high by a fixed amount."""
    return values + severity * scale


def drift(values, severity, scale, rng, dt):
    """The error starts at zero and grows steadily over the run."""
    ramp = np.linspace(0.0, 1.0, len(values))
    return values + severity * scale * ramp


def noise_inflation(values, severity, scale, rng, dt):
    """The reading is still centred correctly but gets much noisier.

    This is the mode a model that predicts only a mean cannot represent:
    nothing about the expected value has changed.
    """
    return values + rng.normal(0.0, severity * scale, size=len(values))


def scale_error(values, severity, scale, rng, dt):
    """The sensor reads a fixed percentage off, so the error grows with the
    signal. A miscalibrated wheel radius looks like this."""
    return values * (1.0 + 0.2 * severity)


def stuck(values, severity, scale, rng, dt):
    """The sensor freezes and repeats its last good reading.

    Severity sets what fraction of the run is frozen, counting from the end.
    """
    out = values.copy()
    n = len(values)
    frozen = int(severity * n)
    if frozen > 0:
        out[n - frozen:] = values[n - frozen - 1]
    return out


def dropout(values, severity, scale, rng, dt):
    """The sensor misses readings at random and holds the previous one.

    Severity is roughly the fraction of samples lost.
    """
    out = values.copy()
    lost = rng.random(len(values)) < severity
    for k in range(1, len(values)):
        if lost[k]:
            out[k] = out[k - 1]
    return out


MODES = {
    "bias": bias,
    "drift": drift,
    "noise_inflation": noise_inflation,
    "scale_error": scale_error,
    "stuck": stuck,
    "dropout": dropout,
}


def apply_fault(readings, channel, mode, severity, seed=0, dt=0.02):
    """Return a copy of `readings` with one channel faulted.

    readings -- the dict that read_sensors gave back
    channel  -- which sensor: "left_encoder", "right_encoder" or "gyro"
    mode     -- a key of MODES
    severity -- 0 is healthy, 1 is trouble the size of the healthy noise.
                May be one number for the whole run, or one per sample for a
                fault that arrives or worsens partway through.

    A per-sample severity works without any change to the fault functions
    because `bias` and `noise_inflation` are pointwise in it: adding
    `severity * scale` and drawing from `normal(0, severity * scale)` both
    broadcast elementwise, and a scale of zero draws zero. `stuck` is the
    exception -- it reads severity as a fraction of the run rather than as an
    amount -- and is scalar only.
    """
    if channel not in CHANNELS:
        raise ValueError("unknown channel: %s" % channel)
    if mode not in MODES:
        raise ValueError("unknown mode: %s" % mode)

    severity = np.asarray(severity, dtype=float)
    if severity.ndim and mode == "stuck":
        raise ValueError("stuck takes a single severity, not one per sample")

    out = dict(readings)
    if np.any(severity > 0):
        rng = np.random.default_rng(seed)
        out[channel] = MODES[mode](np.asarray(readings[channel], dtype=float),
                                   severity, REFERENCE[channel], rng, dt)

    # Record what was done, so an experiment can never lose track of which
    # condition a set of readings came from.
    out["fault_channel"] = channel
    out["fault_mode"] = mode
    out["fault_severity"] = float(severity) if not severity.ndim else severity
    return out


if __name__ == "__main__":
    from trajectories import DT, random_run
    from sensors import read_sensors

    run = random_run(0, duration=20.0)
    healthy = read_sensors(run, seed=0, dt=DT)

    print("Each fault on the left encoder, severity 1.0")
    print("(healthy spread of that channel is %.4f rad/s)\n"
          % REFERENCE["left_encoder"])
    print("  %-16s %12s %12s" % ("mode", "mean shift", "extra spread"))

    base = healthy["left_encoder"]
    for name in MODES:
        broken = apply_fault(healthy, "left_encoder", name, 1.0, seed=0, dt=DT)
        change = broken["left_encoder"] - base
        print("  %-16s %12.4f %12.4f" % (name, change.mean(), change.std()))

    print("\n  bias moves the mean and nothing else.")
    print("  noise_inflation moves the spread and nothing else -- that is the")
    print("  one a model predicting only a mean has no way to express.")

    print("\nSeverity is continuous, not a switch:")
    print("  %-10s %12s" % ("severity", "mean shift"))
    for s in [0.0, 0.25, 0.5, 1.0, 2.0]:
        broken = apply_fault(healthy, "gyro", "bias", s, seed=0, dt=DT)
        print("  %-10.2f %12.5f"
              % (s, (broken["gyro"] - healthy["gyro"]).mean()))

    print("\nHealthy readings are untouched at severity 0:",
          np.array_equal(apply_fault(healthy, "gyro", "bias", 0.0)["gyro"],
                         healthy["gyro"]))

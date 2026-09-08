"""
A nine-channel IMU and magnetometer, with the noise a real one has.

THE CHANNELS

    accel_x/y/z    specific force in the body frame, m/s^2
    gyro_x/y/z     body angular rates, rad/s
    mag_x/y/z      the magnetic field direction in the body frame, unitless

Nine measurements for six states. That leaves three spare, which is what makes
a single failed channel identifiable at all -- the same redundancy argument the
ground robot rests on, with more room. There, three sensors constrained two
quantities and exactly one channel could break before the problem became
unsolvable.

WHY THE NOISE IS NOT CONSTANT

A MEMS accelerometer on a quadcopter does not have a fixed noise floor. Most of
what it reports beyond gravity is airframe vibration, and vibration tracks how
hard the vehicle is working -- rotor speed changes with manoeuvring, so the
noise grows when the vehicle is rotating quickly. That is a real effect and it
is why accelerometer-based attitude estimation degrades exactly when the
attitude is changing fastest.

Modelled here as a floor plus a term proportional to the rate magnitude. It
matters for this project because it makes the measurement covariance
state-dependent, which is the thing a heteroscedastic model can represent and
a constant R cannot. Without it, R(x) would be a constant and the
heteroscedastic half of the argument would have nothing to say.

The gyro gets a per-run bias, drawn once and held, as MEMS gyros do. The
magnetometer gets a fixed noise floor -- its errors come from hard and soft
iron distortion rather than from motion, and those are calibration problems
rather than noise.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from dynamics import magnetic_field, specific_force

CHANNELS = ["accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z"]

# Grouped by physical device. A fault is something that happens to a sensor,
# not to an axis, so health is carried per device and every axis of a degraded
# device is degraded together.
DEVICES = {"accel": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}

ACCEL_FLOOR = 0.08             # m/s^2, still air
ACCEL_VIBRATION = 0.22         # m/s^2 per rad/s of body rate
GYRO_NOISE = 0.006             # rad/s
GYRO_BIAS_SPREAD = 0.010       # rad/s, drawn once per run
MAG_NOISE = 0.018              # unitless, the field is a unit vector


def noise_levels(run):
    """The standard deviation of each channel at each moment.

    Returned rather than applied, because a fair comparison needs the true
    covariance the learned model is trying to predict. The optimal constant R
    is the average of these squared, and no constant can do better -- which is
    the baseline a state-dependent covariance has to beat.
    """
    rate = np.sqrt(run["p"] ** 2 + run["q"] ** 2 + run["r"] ** 2)
    accel = ACCEL_FLOOR + ACCEL_VIBRATION * rate

    levels = np.zeros((len(rate), len(CHANNELS)))
    levels[:, 0:3] = accel[:, None]
    levels[:, 3:6] = GYRO_NOISE
    levels[:, 6:9] = MAG_NOISE
    return levels


def ideal_readings(roll, pitch, yaw, p, q, r):
    """What the nine channels read with no noise and nothing broken.

    This is the function a measurement model has to reproduce. Six of the nine
    channels go through a rotation matrix, so six of them are trigonometric in
    the attitude. The three gyro channels are the identity.
    """
    force = specific_force(roll, pitch, yaw)
    field = magnetic_field(roll, pitch, yaw)
    rates = np.column_stack([np.atleast_1d(p), np.atleast_1d(q),
                             np.atleast_1d(r)])
    return np.hstack([force, rates, field])


def read_sensors(run, seed=0, dt=None):
    """One flight's worth of noisy readings, as a dict of channels."""
    rng = np.random.default_rng(10_000 + seed)

    clean = ideal_readings(run["roll"], run["pitch"], run["yaw"],
                           run["p"], run["q"], run["r"])
    levels = noise_levels(run)

    readings = clean + rng.normal(0.0, 1.0, size=clean.shape) * levels

    # One bias per gyro axis, drawn once and held for the whole run. A filter
    # given a run it has not seen cannot know these, which puts a floor under
    # how well any measurement model can do on those channels.
    bias = rng.normal(0.0, GYRO_BIAS_SPREAD, size=3)
    readings[:, 3:6] += bias

    out = {name: readings[:, i] for i, name in enumerate(CHANNELS)}
    out["gyro_bias"] = bias
    return out


def stack(meas):
    """The nine channels as one array the filter can step through."""
    return np.column_stack([meas[name] for name in CHANNELS])


if __name__ == "__main__":
    from trajectories import DT, DURATION, random_run

    print("Do the readings match what the geometry says?\n")
    run = random_run(0)
    meas = read_sensors(run, seed=0)
    clean = ideal_readings(run["roll"], run["pitch"], run["yaw"],
                           run["p"], run["q"], run["r"])
    got = stack(meas)

    print("  %-12s %12s %12s %12s" % ("channel", "mean error", "sd of error",
                                      "expected sd"))
    levels = noise_levels(run)
    for i, name in enumerate(CHANNELS):
        error = got[:, i] - clean[:, i]
        print("  %-12s %12.5f %12.5f %12.5f"
              % (name, error.mean(), error.std(), levels[:, i].mean()))

    print("\n  The gyro rows carry a per-run bias, so their mean error is not")
    print("  zero and should not be. Everything else should be centred.")

    print("\n\nHow much does the accelerometer noise actually vary?\n")
    rate = np.sqrt(run["p"] ** 2 + run["q"] ** 2 + run["r"] ** 2)
    print("  body rate over the run: %.2f to %.2f rad/s"
          % (rate.min(), rate.max()))
    print("  accel noise sd:         %.3f to %.3f m/s^2"
          % (levels[:, 0].min(), levels[:, 0].max()))
    print("  ratio:                  %.1fx"
          % (levels[:, 0].max() / levels[:, 0].min()))
    print("\n  A constant R has to pick one number for that whole range.")
    print("  That is the gap a state-dependent covariance is for, and on the")
    print("  ground robot the equivalent ratio was about 1.2x -- here it is")
    print("  much larger, so there is more for the model to earn.")

    print("\n\nIs a single failed channel still identifiable?\n")
    print("  %d measurements, %d states, so %d spare."
          % (len(CHANNELS), 6, len(CHANNELS) - 6))
    print("  The ground robot had exactly one spare. Three is more room, and")
    print("  whether that changes the redundancy story is worth testing")
    print("  rather than assuming.")

"""
Builds the training dataset, and lets the sensor noise be varied on purpose.

The committed data/robot_data.csv had no script behind it, which meant it
could not be regenerated or checked. This file is that script. Run with no
arguments it reproduces the shipped dataset exactly, given the same seeds.

WHY IT TAKES A NOISE-GROWTH ARGUMENT

The encoders get noisier the faster the wheels spin. How much noisier is set
by one number in sensors.py, and that number decides how much there is for a
state-dependent noise model to find. Turn it to zero and the noise is the
same everywhere, so a single hand-tuned constant is the correct answer and
nothing can beat it. Turn it up and a constant becomes a worse and worse
compromise between the quiet and the loud parts of a run.

Sweeping that number is how to ask when a learned covariance starts being
worth its cost, rather than asserting that it is. So the generator takes it
as an argument instead of reading the module-level constant.

A NOTE ON WHAT DILUTES IT

Encoders count whole ticks, and that rounding contributes a fixed spread of
about 0.0886 rad/s no matter what the wheel is doing. At low speeds that is
nearly half the total variance, and it is state-independent, so no model can
predict it. It sits underneath the part that does vary and shrinks the
difference a state-dependent model can exploit -- 33 per cent across the
speed range instead of 58. That is a property of the hardware being
simulated, not a flaw, but it explains why the effect looks small.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import dynamics
import sensors
from trajectories import DT, random_run

N_RUNS = 100
DURATION = 20.0

COLUMNS = ["run", "t", "x", "y", "heading", "speed", "turn_rate",
           "left_spin", "right_spin", "left_encoder", "right_encoder",
           "gyro", "left_noise", "right_noise", "gyro_noise", "gyro_bias"]


def build(encoder_growth=None, gyro_growth=None, n_runs=N_RUNS,
          duration=DURATION, ticks=None, radius_error=None,
          width_error=None, first_run=0):
    """Generate the dataset as a DataFrame.

    encoder_growth -- extra encoder noise per rad/s of wheel spin.
                      None keeps whatever sensors.py declares.
                      0.0 makes the noise homoscedastic.
    gyro_growth    -- the same, for the gyro against turn rate.
    ticks          -- encoder counts per wheel revolution. Raising it makes
                      the rounding finer. Rounding contributes a fixed spread
                      that no state-dependent model can predict, so a coarse
                      encoder hides part of the effect such a model exists to
                      find -- 33 per cent of variation visible instead of 56.
    radius_error   -- how far the real wheel radius is from the written-down
                      one, as a fraction. Non-zero makes the filter's
                      hand-written measurement model systematically wrong,
                      which is the normal condition on real hardware and the
                      one this simulation otherwise fails to represent.
    width_error    -- the same, for track width.
    first_run      -- which run index to start from. The index seeds both
                      the trajectory and the noise, so callers building a
                      dataset one run at a time must advance it or every
                      run comes out identical.

    These live as module-level constants in sensors.py and dynamics.py, so
    they are set around the call and put back afterwards. Not elegant, but it
    keeps the sensor and motion models in one place rather than threading
    parameters through every function that reads them.
    """
    saved = (sensors.ENCODER_NOISE_GROWTH, sensors.GYRO_NOISE_GROWTH,
             sensors.TICKS_PER_TURN,
             dynamics.RADIUS_ERROR, dynamics.WIDTH_ERROR)
    if encoder_growth is not None:
        sensors.ENCODER_NOISE_GROWTH = encoder_growth
    if gyro_growth is not None:
        sensors.GYRO_NOISE_GROWTH = gyro_growth
    if ticks is not None:
        sensors.TICKS_PER_TURN = ticks
    if radius_error is not None:
        dynamics.RADIUS_ERROR = radius_error
    if width_error is not None:
        dynamics.WIDTH_ERROR = width_error

    try:
        frames = []
        for run_id in range(first_run, first_run + n_runs):
            run = random_run(run_id, duration=duration)
            meas = sensors.read_sensors(run, seed=run_id, dt=DT)
            frames.append(pd.DataFrame({
                "run": run_id,
                "t": run["t"],
                "x": run["x"], "y": run["y"], "heading": run["heading"],
                "speed": run["speed"], "turn_rate": run["turn_rate"],
                "left_spin": run["left_spin"], "right_spin": run["right_spin"],
                "left_encoder": meas["left_encoder"],
                "right_encoder": meas["right_encoder"],
                "gyro": meas["gyro"],
                "left_noise": meas["left_noise"],
                "right_noise": meas["right_noise"],
                "gyro_noise": meas["gyro_noise"],
                "gyro_bias": meas["gyro_bias"],
            }))
    finally:
        (sensors.ENCODER_NOISE_GROWTH, sensors.GYRO_NOISE_GROWTH,
         sensors.TICKS_PER_TURN,
         dynamics.RADIUS_ERROR, dynamics.WIDTH_ERROR) = saved

    return pd.concat(frames, ignore_index=True)[COLUMNS]


def observable_range(frame, ticks=None):
    """How much the true encoder spread varies across the dataset.

    This is what a state-dependent noise model has to work with. It counts
    tick rounding, because that is part of what the sensor actually does.

    `ticks` must be the setting the data was generated WITH. build() puts the
    module constant back when it finishes, so reading it here reports on
    whatever sensors.py currently says rather than on the data in hand -- and
    then quantised and unquantised runs score identically, which is how this
    was caught.
    """
    if ticks is None:
        ticks = sensors.TICKS_PER_TURN
    tick = (2 * np.pi / ticks) / DT
    quant = tick ** 2 / 12
    spin = frame["left_spin"].abs()
    lo = np.sqrt(frame["left_noise"][spin < np.percentile(spin, 1)].mean() ** 2
                 + quant)
    hi = np.sqrt(frame["left_noise"][spin > np.percentile(spin, 99)].mean() ** 2
                 + quant)
    return float(hi / lo - 1.0)


def gyro_range(frame):
    """The same, for the gyro against turn rate.

    Needed because the two channels have separate growth constants, and a
    control condition that zeroes one while leaving the other alone is not a
    control. That mistake was made here once already.
    """
    turn = frame["turn_rate"].abs()
    lo = frame["gyro_noise"][turn < np.percentile(turn, 1)].mean()
    hi = frame["gyro_noise"][turn > np.percentile(turn, 99)].mean()
    return float(hi / lo - 1.0)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "robot_data.csv"

    print("Rebuilding the shipped dataset")
    df = build()
    print("  %d rows, %d runs, %d columns" % (len(df), df["run"].nunique(),
                                              len(df.columns)))

    if out.exists():
        old = pd.read_csv(out)
        same_shape = old.shape == df.shape
        # The csv is written rounded, so compare at the precision it stores.
        rounded = df.round(dict(zip(old.columns, [0, 2] + [4] * 5 + [4] * 5
                                    + [5, 5, 5, 6])))
        close = same_shape and np.allclose(
            old[["speed", "turn_rate", "left_spin"]].values,
            rounded[["speed", "turn_rate", "left_spin"]].values, atol=1e-3)
        print("  matches the committed file: %s" % ("yes" if close else "NO"))
        if not close:
            print("  (if this says NO, the committed csv came from different")
            print("   seeds or settings and should be regenerated from here)")

    print("\nHow much heteroscedasticity each setting produces\n")
    print("  %-16s %-12s %14s" % ("encoder growth", "ticks/rev", "spread range"))
    for growth in [0.0, 0.05, 0.15, 0.40]:
        for ticks in [sensors.TICKS_PER_TURN, 16384]:
            small = build(encoder_growth=growth, n_runs=8, duration=5.0,
                          ticks=ticks)
            print("  %-16.2f %-12d %13.1f%%"
                  % (growth, ticks, 100 * observable_range(small, ticks)))

    print("\n  At growth 0 the noise is the same everywhere and a single")
    print("  constant is the right answer, so nothing can beat a tuned R.")
    print("  That row is the control the rest of the sweep is measured")
    print("  against.")

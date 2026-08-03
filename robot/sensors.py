"""
What the robot's sensors report: two wheel encoders and a gyro.

Everything here is a healthy sensor. Faults come later, in their own file.

Three sensors for two unknowns (speed and turn rate) means there's one
spare. That's on purpose. With exactly enough sensors, a broken one just
gives you a different answer and nothing looks wrong. With a spare, the
sensors disagree with each other, and that disagreement is the clue.
"""

import numpy as np

# Noise levels. The encoders get noisier when the wheels spin fast, and the
# gyro gets noisier when the robot turns hard. That is the whole reason a
# model can learn the noise: it depends on something you can see.
ENCODER_NOISE = 0.08          # rad/s when barely moving
ENCODER_NOISE_GROWTH = 0.05   # extra noise per rad/s of wheel spin
GYRO_NOISE = 0.010            # rad/s when going straight
GYRO_NOISE_GROWTH = 0.03      # extra noise per rad/s of turning
GYRO_BIAS_SIZE = 0.005        # each gyro reads slightly off, all run long

TICKS_PER_TURN = 1024         # encoder counts per full wheel rotation


def encoder_noise_level(spin_rate):
    """Encoder noise, worse the faster the wheel spins."""
    return ENCODER_NOISE * (1 + ENCODER_NOISE_GROWTH * abs(spin_rate))


def gyro_noise_level(turn_rate):
    """Gyro noise, worse the harder the robot turns."""
    return GYRO_NOISE * (1 + GYRO_NOISE_GROWTH * abs(turn_rate))


def round_to_ticks(rate, dt):
    """Encoders count whole ticks, so they can't report just any number.

    In one time step the encoder might count 32 ticks or 33, never 32.4.
    So the spin rate it reports gets rounded to the nearest multiple of
    one tick per time step.
    """
    one_tick = (2 * np.pi / TICKS_PER_TURN) / dt
    return round(rate / one_tick) * one_tick


def read_sensors(run, seed, dt=0.02):
    """Take a run from trajectories.py and produce noisy sensor readings."""
    rng = np.random.default_rng(seed)
    n = len(run["t"])

    # The gyro's bias is picked once and stays the same for the whole run.
    # That's how real gyros behave, and it means runs can't be split up
    # randomly later on: every reading in a run shares this one offset.
    gyro_bias = rng.normal(0, GYRO_BIAS_SIZE)

    left = np.zeros(n)
    right = np.zeros(n)
    gyro = np.zeros(n)
    left_noise = np.zeros(n)
    right_noise = np.zeros(n)
    gyro_noise = np.zeros(n)

    for k in range(n):
        # left encoder
        noise = encoder_noise_level(run["left_spin"][k])
        reading = run["left_spin"][k] + rng.normal(0, noise)
        left[k] = round_to_ticks(reading, dt)
        left_noise[k] = noise

        # right encoder
        noise = encoder_noise_level(run["right_spin"][k])
        reading = run["right_spin"][k] + rng.normal(0, noise)
        right[k] = round_to_ticks(reading, dt)
        right_noise[k] = noise

        # gyro
        noise = gyro_noise_level(run["turn_rate"][k])
        gyro[k] = run["turn_rate"][k] + gyro_bias + rng.normal(0, noise)
        gyro_noise[k] = noise

    return {
        "left_encoder": left,
        "right_encoder": right,
        "gyro": gyro,
        # the real noise levels, kept so a model's guess can be graded later
        "left_noise": left_noise,
        "right_noise": right_noise,
        "gyro_noise": gyro_noise,
        "gyro_bias": gyro_bias,
    }


if __name__ == "__main__":
    from dynamics import WHEEL_RADIUS, speed_and_turn_from_wheels
    from trajectories import DT, s_path, random_run

    run = s_path()
    readings = read_sensors(run, seed=0, dt=DT)
    print("Read %d samples from 3 sensors" % len(readings["gyro"]))

    # How far off were the encoder readings?
    error = readings["left_encoder"] - run["left_spin"]
    one_tick = (2 * np.pi / TICKS_PER_TURN) / DT
    print("\nLeft encoder error")
    print("   actual spread          %.4f rad/s" % error.std())
    print("   from the noise we added %.4f rad/s" % readings["left_noise"].mean())
    print("   from tick rounding      %.4f rad/s" % (one_tick / np.sqrt(12)))
    print("   the two together explain the spread")

    # Is the noise really worse when the wheel spins fast?
    spin = np.abs(run["left_spin"])
    fast = spin > np.percentile(spin, 75)
    slow = spin < np.percentile(spin, 25)
    print("\nNoise when spinning slow: %.4f" % error[slow].std())
    print("Noise when spinning fast: %.4f" % error[fast].std())
    print("   noisier when faster, as designed")

    # Can we still work out what the robot was doing?
    left_speed = readings["left_encoder"] * WHEEL_RADIUS
    right_speed = readings["right_encoder"] * WHEEL_RADIUS
    guess_speed, guess_turn = speed_and_turn_from_wheels(left_speed, right_speed)
    speed_error = np.sqrt(np.mean((guess_speed - run["speed"]) ** 2))
    turn_error = np.sqrt(np.mean((guess_turn - run["turn_rate"]) ** 2))
    print("\nWorking backwards from the encoders")
    print("   speed off by %.4f m/s, turn rate off by %.4f rad/s"
          % (speed_error, turn_error))

    # Do the encoders and the gyro tell the same story about turning?
    agreement = np.corrcoef(guess_turn, readings["gyro"])[0, 1]
    print("\nEncoders and gyro agree on turn rate: %.4f" % agreement)
    print("   they should, since both measure it")
    print("   a broken sensor breaks this agreement, which is how you spot it")

    # Same seed, same readings
    again = read_sensors(run, seed=0, dt=DT)
    other = read_sensors(run, seed=1, dt=DT)
    print("\nSeed 0 twice gives the same readings:",
          np.allclose(again["gyro"], readings["gyro"]))
    print("Seed 1 gives different readings:",
          not np.allclose(other["gyro"], readings["gyro"]))

    # The gyro bias should vary from run to run by about GYRO_BIAS_SIZE
    biases = []
    for seed in range(200):
        r = read_sensors(random_run(seed, duration=2.0), seed=seed, dt=DT)
        biases.append(r["gyro_bias"])
    print("\nGyro bias across 200 runs: spread %.5f (set to %.5f)"
          % (np.std(biases), GYRO_BIAS_SIZE))

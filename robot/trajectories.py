"""
Drives the robot around and records where it went.

Two ways to drive it: a fixed S-shaped path for checking things work, and
random wandering for generating lots of different training runs.
"""

import numpy as np

from dynamics import step, wheel_spin_rates

DT = 0.02          # seconds per step (50 steps per second)
DURATION = 30.0    # seconds per run


def s_path_commands(t):
    """Constant speed, turning left then right then left again."""
    speed = np.full(len(t), 1.0)
    turn_rate = 0.6 * np.sin(2 * np.pi * t / 15.0)
    return speed, turn_rate


def random_commands(t, seed):
    """Random driving that still looks like a real robot.

    Adding a few sine waves with random periods gives something that
    wanders around smoothly. That matters because a real robot can't
    change speed instantly, so the commands shouldn't either.

    Speed and turning get their own separate waves. If the speed were
    always the same, you couldn't tell from the wheel readings whether a
    change came from speeding up or from turning.
    """
    rng = np.random.default_rng(seed)

    # start at 1 m/s and add three slow wobbles
    speed = np.full(len(t), 1.0)
    for i in range(3):
        period = rng.uniform(5.0, 20.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        speed = speed + 0.25 * np.sin(2 * np.pi * t / period + phase)

    # how hard this particular run turns; some runs are nearly straight,
    # others are twisty
    turn_size = rng.uniform(0.2, 1.2)

    turn_rate = np.zeros(len(t))
    for i in range(3):
        period = rng.uniform(5.0, 20.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        turn_rate = turn_rate + (turn_size / 3) * np.sin(2 * np.pi * t / period + phase)

    speed = np.clip(speed, 0.3, 1.6)
    return speed, turn_rate


def drive(speed, turn_rate, dt=DT):
    """Run the robot through a list of commands and record the path."""
    n = len(speed)
    x = np.zeros(n)
    y = np.zeros(n)
    heading = np.zeros(n)

    state = np.array([0.0, 0.0, 0.0])
    for k in range(n):
        x[k] = state[0]
        y[k] = state[1]
        heading[k] = state[2]
        state = step(state, speed[k], turn_rate[k], dt)

    left_spin, right_spin = wheel_spin_rates(speed, turn_rate)

    return {
        "t": np.arange(n) * dt,
        "x": x,
        "y": y,
        "heading": heading,
        "speed": speed,
        "turn_rate": turn_rate,
        "left_spin": left_spin,
        "right_spin": right_spin,
    }


def s_path(duration=DURATION, dt=DT):
    """The fixed S-shaped test path."""
    t = np.arange(0, duration, dt)
    speed, turn_rate = s_path_commands(t)
    return drive(speed, turn_rate, dt)


def random_run(seed, duration=DURATION, dt=DT):
    """One random run. Same seed gives the same run every time."""
    t = np.arange(0, duration, dt)
    speed, turn_rate = random_commands(t, seed)
    return drive(speed, turn_rate, dt)


if __name__ == "__main__":
    run = s_path()
    print("S path: %d steps over %.0f seconds" % (len(run["t"]), run["t"][-1]))
    print("   turn rate goes from %+.2f to %+.2f rad/s"
          % (run["turn_rate"].min(), run["turn_rate"].max()))
    print("   it turns both directions, which is what we want")

    again = s_path()
    print("   running it twice gives the same path:",
          np.allclose(run["x"], again["x"]))

    print("\nRandom runs")
    a = random_run(0)
    b = random_run(0)
    c = random_run(1)
    print("   seed 0 twice gives the same run:", np.allclose(a["x"], b["x"]))
    print("   seed 1 gives a different run:", not np.allclose(a["x"], c["x"]))

    print("\nWhat 200 random runs cover")
    all_speed = []
    all_turn = []
    for seed in range(200):
        run = random_run(seed, duration=10.0)
        all_speed.append(run["speed"])
        all_turn.append(run["turn_rate"])
    all_speed = np.concatenate(all_speed)
    all_turn = np.concatenate(all_turn)

    print("   speed goes from %.2f to %.2f m/s"
          % (all_speed.min(), all_speed.max()))
    print("   turn rate goes from %+.2f to %+.2f rad/s"
          % (all_turn.min(), all_turn.max()))

    # If speed and turning always went up together, you couldn't separate
    # their effects on the wheels. Close to zero means they're independent.
    link = np.corrcoef(all_speed, np.abs(all_turn))[0, 1]
    print("   speed and turning are linked by %.3f (want close to 0)" % link)

    sharp = np.mean(np.abs(all_turn) > 0.6)
    gentle = np.mean((np.abs(all_turn) > 0.15) & (np.abs(all_turn) <= 0.6))
    straight = np.mean(np.abs(all_turn) <= 0.15)
    print("   %.0f%% sharp turns, %.0f%% gentle turns, %.0f%% nearly straight"
          % (sharp * 100, gentle * 100, straight * 100))

    # commands should change gradually, not jump around
    fastest_speed_change = np.abs(np.diff(a["speed"])).max() / DT
    fastest_turn_change = np.abs(np.diff(a["turn_rate"])).max() / DT
    print("   speed changes at most %.2f m/s per second" % fastest_speed_change)
    print("   turning changes at most %.2f rad/s per second" % fastest_turn_change)

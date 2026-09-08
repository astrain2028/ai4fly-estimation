"""
Flights to estimate over.

WHAT A RUN HAS TO CONTAIN

Enough attitude variation that the nonlinearity is actually exercised. A
quadcopter hovering level is a linear problem -- around zero roll and pitch the
rotation matrix is close to the identity plus a small correction, and any
learned model would only ever see that corner. Fitting there and claiming the
map is nonlinear would be a knob turned to suit a conclusion.

So the rates are driven hard enough to reach thirty or forty degrees of bank,
which is ordinary for a quadcopter changing direction, and yaw is allowed to
run right around.

WHY THE RATES ARE SMOOTH RATHER THAN NOISE

Same reason the ground robot's speed is. A real vehicle is commanded by a
controller with finite bandwidth, so its angular rates change over tenths of a
second rather than between samples. Driving them with white noise would make
the increments independent, which is a random walk, and the filter's process
model would then be right for the wrong reason -- the earlier version of the
robot sim had exactly that problem and it took adding acceleration to the
state to fix it.

Here the rates come from a sum of a few sinusoids at different frequencies,
which gives smooth, aperiodic, bounded motion with a controllable amount of
energy at each timescale.

GIMBAL LOCK

Euler angles are singular at ninety degrees of pitch. Pitch is soft-limited
well below that. A serious implementation would carry a quaternion; this one
is honest about staying inside the region where angles are fine, because the
subject is the measurement model rather than the attitude representation.
"""

import numpy as np

from dynamics import step

DT = 0.02                      # 50 Hz, the same rate the ground robot runs at
DURATION = 20.0

MAX_PITCH = np.radians(45.0)   # kept well clear of the singularity at 90
RATE_SCALE = 1.2               # rad/s, roughly, at the peak of a manoeuvre
N_TONES = 4                    # sinusoids summed per axis


def _smooth_signal(rng, n_steps, dt, scale, n_tones=N_TONES):
    """A bounded, smooth, aperiodic signal.

    Frequencies are drawn from a range whose slow end takes several seconds
    per cycle and whose fast end takes about half a second. Phases are random,
    so runs from different seeds are genuinely different rather than shifted
    copies of one another.
    """
    t = np.arange(n_steps) * dt
    out = np.zeros(n_steps)
    for _ in range(n_tones):
        freq = rng.uniform(0.08, 1.8)
        phase = rng.uniform(0.0, 2 * np.pi)
        out += np.sin(2 * np.pi * freq * t + phase)
    return scale * out / n_tones


def random_run(seed=0, duration=DURATION, dt=DT):
    """One flight: attitude and body rates at every sample.

    The rates are generated first and the attitude integrated from them, so
    the two are consistent by construction -- the filter's motion model and
    the truth agree about how rates turn into angles, and any error it makes
    is its own rather than the simulator's.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)

    p = _smooth_signal(rng, n_steps, dt, RATE_SCALE)
    q = _smooth_signal(rng, n_steps, dt, RATE_SCALE)
    r = _smooth_signal(rng, n_steps, dt, RATE_SCALE * 0.7)

    roll = np.zeros(n_steps)
    pitch = np.zeros(n_steps)
    yaw = np.zeros(n_steps)

    state = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi),
                      p[0], q[0], r[0]])

    for k in range(n_steps):
        state[3], state[4], state[5] = p[k], q[k], r[k]
        roll[k], pitch[k], yaw[k] = state[0], state[1], state[2]

        # A soft limit rather than a hard clip. Clipping would put a corner in
        # the truth that no smooth motion model could follow, and the filter
        # would be scored on the simulator's discontinuity rather than on
        # anything about the sensors.
        if abs(state[1]) > MAX_PITCH:
            q[k:] *= 0.5
            state[1] = np.clip(state[1], -MAX_PITCH, MAX_PITCH)

        state = step(state, dt)

    return {"roll": roll, "pitch": pitch, "yaw": yaw,
            "p": p, "q": q, "r": r,
            "time": np.arange(n_steps) * dt}


def truth_matrix(run):
    """The six true states as one array, in filter order."""
    return np.column_stack([run["roll"], run["pitch"], run["yaw"],
                            run["p"], run["q"], run["r"]])


if __name__ == "__main__":
    print("Twenty runs of %.0f s at %d Hz\n" % (DURATION, round(1 / DT)))
    print("  %-6s %10s %10s %10s %10s"
          % ("seed", "roll sd", "pitch sd", "yaw range", "max |rate|"))

    for seed in range(5):
        run = random_run(seed)
        print("  %-6d %10.1f %10.1f %10.1f %10.2f"
              % (seed, np.degrees(run["roll"].std()),
                 np.degrees(run["pitch"].std()),
                 np.degrees(run["yaw"].max() - run["yaw"].min()),
                 max(np.abs(run["p"]).max(), np.abs(run["q"]).max())))

    print("\n  Angles in degrees, rates in rad/s.")

    print("\nIs the attitude varied enough to exercise the nonlinearity?\n")
    everything = np.concatenate([np.abs(random_run(s)["roll"])
                                 for s in range(20)])
    print("  fraction of samples past 10 degrees of roll: %.0f%%"
          % (100 * (everything > np.radians(10)).mean()))
    print("  fraction past 25 degrees:                    %.0f%%"
          % (100 * (everything > np.radians(25)).mean()))
    print("\n  Near zero the rotation matrix is nearly the identity and the")
    print("  problem is nearly linear. A study of a nonlinear map that spent")
    print("  its time there would be measuring the wrong thing.")

    print("\nDoes the integration stay clear of gimbal lock?\n")
    worst = max(np.abs(random_run(s)["pitch"]).max() for s in range(20))
    print("  largest pitch over 20 runs: %.1f degrees (limit is %.0f)"
          % (np.degrees(worst), np.degrees(MAX_PITCH)))

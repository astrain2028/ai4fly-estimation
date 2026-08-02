"""
Control profiles and trajectory rollout for the differential-drive robot.

`dynamics.py` answers what happens over one timestep given a commanded
forward speed and turn rate. This module supplies the command sequences and
integrates them into state histories. It contains no sensor model and no
noise -- measurements are a separate concern.

TWO KINDS OF TRAJECTORY

`s_path` is a deterministic serpentine used as a fixed reference. Constant
forward speed with a sinusoidal turn rate, so the vehicle sweeps smoothly
through zero curvature and the inside/outside wheel roles reverse. It exists
so that every change to the pipeline can be checked against an identical
trajectory.

`random_trajectory` produces the varied episodes that a learned measurement
model is trained on. Its design matters more than it appears: a model can
only learn relationships that are present in its training set, so the way
these commands are drawn determines what the model is capable of learning at
all.

Two properties are deliberate.

*Speed and turn rate are drawn independently.* If forward speed were held
constant across episodes, wheel speeds would be an affine function of turn
rate alone, and nothing in the data would let a model separate the two
contributions. Varying them independently makes the underlying relation
    v_left  = v - omega * W/2
    v_right = v + omega * W/2
identifiable rather than degenerate.

*Commands are smooth, not white noise.* Each profile is a small sum of
sinusoids at random frequencies and phases, which is band-limited by
construction. A physical vehicle has bounded acceleration; training on
step-to-step independent commands would teach a model relationships that
cannot occur in deployment, and would make finite-difference quantities
(used later by sensor models) meaningless.

Turn rates are drawn to cover tight turns, gentle turns, and near-straight
motion. Tight turns are where the two sides differ most and encoders are most
informative; straight motion is where they carry the least information about
heading. A training set containing only one of these regimes yields a model
that fails in the other.

Run: python trajectories.py
"""
from __future__ import annotations

import numpy as np

from dynamics import (N_STATE, TRACK_WIDTH, rk4_step, wheel_speeds,
                      wheel_rates, wrap_angle)

DT = 0.02                 # s, 50 Hz
DURATION = 30.0           # s

# command envelopes -- the ranges a random episode may occupy
V_RANGE = (0.3, 1.6)      # m/s, forward speed
OMEGA_MAX = 1.2           # rad/s, peak turn rate


def s_path_commands(t, v_nominal: float = 1.0, turn_amplitude: float = 0.6,
                    period: float = 15.0):
    """Constant speed, sinusoidal turn rate: a smooth serpentine."""
    v = np.full_like(t, float(v_nominal))
    omega = turn_amplitude * np.sin(2.0 * np.pi * t / period)
    return v, omega


def _smooth_profile(t, rng, lo, hi, n_components=3, min_period=6.0,
                    max_period=25.0):
    """A band-limited random signal on [lo, hi].

    Sum of a few sinusoids with random period and phase, rescaled to span the
    requested range. Band-limited because a real vehicle cannot change
    commands arbitrarily fast.
    """
    sig = np.zeros_like(t)
    for _ in range(n_components):
        period = rng.uniform(min_period, max_period)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        sig += np.sin(2.0 * np.pi * t / period + phase)
    sig /= n_components
    # sig is in roughly [-1, 1]; map onto [lo, hi] preserving shape
    return lo + (hi - lo) * 0.5 * (sig + 1.0)


def random_commands(t, seed: int, v_range=V_RANGE, omega_max: float = OMEGA_MAX):
    """Independent smooth profiles for forward speed and turn rate.

    The turn-rate amplitude is itself drawn per episode, so the set of
    episodes spans tight-turning runs through nearly straight ones rather
    than every episode looking the same.
    """
    rng = np.random.default_rng(seed)
    v = _smooth_profile(t, rng, *v_range)
    amp = rng.uniform(0.15, 1.0) * omega_max      # this episode's turn vigour
    omega = _smooth_profile(t, rng, -amp, amp)
    return v, omega


def rollout(v, omega, dt: float = DT, state0=(0.0, 0.0, 0.0),
            track_width: float = TRACK_WIDTH):
    """Integrate a command sequence into a state history.

    Returns a dict of arrays, all of length N. The wheel quantities are the
    noiseless truth implied by the commands -- a sensor model adds noise and
    faults on top of these, it does not replace them.
    """
    v = np.asarray(v, dtype=float)
    omega = np.asarray(omega, dtype=float)
    n = len(v)

    states = np.zeros((n, N_STATE))
    q = np.array(state0, dtype=float)
    for k in range(n):
        states[k] = q
        q = rk4_step(q, [v[k], omega[k]], dt)

    v_l, v_r = wheel_speeds(v, omega, track_width)
    rate_l, rate_r = wheel_rates(v, omega, track_width=track_width)
    return {
        "t": np.arange(n) * dt,
        "x": states[:, 0], "y": states[:, 1], "theta": states[:, 2],
        "v": v, "omega": omega,
        "v_left": v_l, "v_right": v_r,
        "rate_left": rate_l, "rate_right": rate_r,
    }


def s_path(duration: float = DURATION, dt: float = DT, **kwargs):
    t = np.arange(0.0, duration, dt)
    return rollout(*s_path_commands(t, **kwargs), dt=dt)


def random_trajectory(seed: int, duration: float = DURATION, dt: float = DT,
                      **kwargs):
    t = np.arange(0.0, duration, dt)
    return rollout(*random_commands(t, seed, **kwargs), dt=dt)


def _summarise(traj, label):
    print(f"{label:<22}"
          f"v [{traj['v'].min():.2f}, {traj['v'].max():.2f}] m/s   "
          f"omega [{traj['omega'].min():+.2f}, {traj['omega'].max():+.2f}] rad/s   "
          f"heading swept {np.ptp(np.unwrap(traj['theta'])):.1f} rad")


def _self_test():
    ok = True

    ref = s_path()
    _summarise(ref, "s_path (reference)")

    # the reference sweeps both turn directions
    both_ways = ref["omega"].min() < -0.1 and ref["omega"].max() > 0.1
    print(f"  sweeps both turn directions   {'ok' if both_ways else 'FAIL'}")
    ok &= both_ways

    # rollout must be deterministic
    same = np.allclose(s_path()["x"], ref["x"])
    print(f"  deterministic                 {'ok' if same else 'FAIL'}")
    ok &= same

    # random episodes: reproducible per seed, and different across seeds
    a1, a2, b = (random_trajectory(0), random_trajectory(0), random_trajectory(1))
    rep = np.allclose(a1["x"], a2["x"])
    diff = not np.allclose(a1["x"], b["x"])
    print(f"  seed reproducible / distinct  "
          f"{'ok' if rep and diff else 'FAIL'}")
    ok &= rep and diff

    # coverage across many episodes -- the property that decides what a model
    # can learn from this data
    print("\ncoverage over 200 random episodes:")
    trajs = [random_trajectory(s, duration=10.0) for s in range(200)]
    v_all = np.concatenate([tr["v"] for tr in trajs])
    w_all = np.concatenate([tr["omega"] for tr in trajs])
    print(f"  speed      [{v_all.min():.2f}, {v_all.max():.2f}] m/s")
    print(f"  turn rate  [{w_all.min():+.2f}, {w_all.max():+.2f}] rad/s")

    # speed and turn rate must not be collinear, or their effects on wheel
    # speeds cannot be separated by any model
    corr = float(np.corrcoef(v_all, np.abs(w_all))[0, 1])
    indep = abs(corr) < 0.15
    print(f"  |correlation| speed vs |turn| {abs(corr):.3f}  "
          f"{'ok' if indep else 'FAIL -- confounded'}")
    ok &= indep

    # all three turning regimes present
    tight = float(np.mean(np.abs(w_all) > 0.6))
    gentle = float(np.mean((np.abs(w_all) > 0.15) & (np.abs(w_all) <= 0.6)))
    straight = float(np.mean(np.abs(w_all) <= 0.15))
    print(f"  regime mix  tight {tight:.2f}  gentle {gentle:.2f}  "
          f"near-straight {straight:.2f}")
    mix_ok = min(tight, gentle, straight) > 0.05
    print(f"  all regimes represented       {'ok' if mix_ok else 'FAIL'}")
    ok &= mix_ok

    # commands stay band-limited: bounded change between steps
    dv = np.abs(np.diff(trajs[0]["v"])).max() / DT
    dw = np.abs(np.diff(trajs[0]["omega"])).max() / DT
    smooth = dv < 5.0 and dw < 5.0
    print(f"  command rates  dv/dt {dv:.2f} m/s^2, domega/dt {dw:.2f} rad/s^2  "
          f"{'ok' if smooth else 'FAIL'}")
    ok &= smooth

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    _self_test()

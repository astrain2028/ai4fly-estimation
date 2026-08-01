"""
Kinematic model of a four-wheeled differential-drive (skid-steer) robot.

This module is the motion model only. It defines how the robot's state
evolves given a commanded forward speed and turn rate, and how those commands
relate to individual wheel speeds. It generates no trajectories, simulates no
sensors, and produces no figures -- those belong in separate modules that
import this one.

STATE AND CONTROLS

    state    q = [x, y, theta]
             x, y     position in the world frame (m)
             theta    heading, counter-clockwise from the +x axis (rad)

    control  u = [v, omega]
             v        forward speed along the body axis (m/s)
             omega    turn rate (rad/s), positive counter-clockwise

The four wheels are driven as two independently controlled sides, so the
vehicle has two degrees of freedom in its commands and behaves kinematically
as a unicycle:

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = omega

This is a kinematic model, not a dynamic one: it assumes commanded speeds are
achieved instantly, and neglects mass, tyre slip, and actuator lag. That is
the appropriate level of detail for studying measurement models, where the
question is what the sensors report given the motion, not how the motion
arises. Skid-steer vehicles do slip appreciably during turns on real
surfaces; if that becomes relevant it belongs here as an explicit extension,
not as unmodelled error elsewhere.

WHEEL SPEEDS

A turn requires the two sides to travel at different speeds. With track width
W (the lateral separation between left and right wheels):

    v_left  = v - omega * W/2
    v_right = v + omega * W/2

During a left turn (omega > 0) the left wheels are on the inside of the arc
and cover less ground, so v_left is the smaller of the two. The roles reverse
in a right turn. This difference is the physical basis for inferring motion
from wheel encoders, and inverting it recovers the commands:

    v     = (v_right + v_left) / 2
    omega = (v_right - v_left) / W

Angular rate of a wheel follows from its linear speed and radius r:

    wheel_rate = wheel_speed / r        (rad/s)
"""
from __future__ import annotations

import numpy as np

# --- vehicle geometry -----------------------------------------------------
WHEEL_RADIUS = 0.10       # m
TRACK_WIDTH = 0.45        # m, lateral separation between left and right wheels

N_STATE = 3               # [x, y, theta]
N_CONTROL = 2             # [v, omega]


def wrap_angle(theta):
    """Wrap an angle to [-pi, pi] so heading never accumulates unbounded."""
    return np.arctan2(np.sin(theta), np.cos(theta))


def wheel_speeds(v, omega, track_width: float = TRACK_WIDTH):
    """Linear speeds of the left and right wheel pairs, in m/s.

    Returns (v_left, v_right). Accepts scalars or arrays.
    """
    half = 0.5 * track_width
    return v - omega * half, v + omega * half


def wheel_rates(v, omega, wheel_radius: float = WHEEL_RADIUS,
                track_width: float = TRACK_WIDTH):
    """Angular rates of the left and right wheels, in rad/s.

    This is the quantity a wheel encoder actually observes.
    """
    v_l, v_r = wheel_speeds(v, omega, track_width)
    return v_l / wheel_radius, v_r / wheel_radius


def commands_from_wheel_speeds(v_left, v_right, track_width: float = TRACK_WIDTH):
    """Inverse of `wheel_speeds`: recover (v, omega) from the two sides.

    Exact for this kinematic model, and the reason encoders are informative
    about motion at all.
    """
    return 0.5 * (v_right + v_left), (v_right - v_left) / track_width


def derivative(q, u):
    """Time derivative of the state: q_dot = f(q, u)."""
    theta = q[2]
    v, omega = u[0], u[1]
    return np.array([v * np.cos(theta), v * np.sin(theta), omega])


def rk4_step(q, u, dt: float):
    """Advance the state one timestep with fourth-order Runge-Kutta.

    Euler integration accumulates visible heading error on curved paths. Since
    this state is the ground truth every later result is measured against,
    that error would masquerade as a bias in the data rather than a defect of
    the integrator.
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    k1 = derivative(q, u)
    k2 = derivative(q + 0.5 * dt * k1, u)
    k3 = derivative(q + 0.5 * dt * k2, u)
    k4 = derivative(q + dt * k3, u)
    q_next = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q_next[2] = wrap_angle(q_next[2])
    return q_next


def _self_test():
    """Property checks on the model. No plots, no trajectories -- these
    verify the relationships the rest of the project depends on."""
    ok = True

    # driving straight leaves both sides equal and the heading unchanged
    v_l, v_r = wheel_speeds(1.0, 0.0)
    ok &= np.isclose(v_l, v_r) and np.isclose(v_l, 1.0)
    q = rk4_step([0.0, 0.0, 0.0], [1.0, 0.0], 0.1)
    ok &= np.allclose(q, [0.1, 0.0, 0.0], atol=1e-12)
    print(f"straight line          {'ok' if ok else 'FAIL'}")

    # left turn: inside (left) wheel is slower
    v_l, v_r = wheel_speeds(1.0, 0.5)
    left_ok = v_l < v_r
    # right turn: roles reverse
    v_l2, v_r2 = wheel_speeds(1.0, -0.5)
    right_ok = v_l2 > v_r2
    print(f"inside/outside swap    {'ok' if left_ok and right_ok else 'FAIL'}"
          f"   (left turn {v_l:.3f} < {v_r:.3f} m/s)")
    ok &= left_ok and right_ok

    # the inverse mapping recovers the commands exactly
    v_in, w_in = 0.8, -0.35
    v_out, w_out = commands_from_wheel_speeds(*wheel_speeds(v_in, w_in))
    inv_ok = np.isclose(v_in, v_out) and np.isclose(w_in, w_out)
    print(f"encoder inverse        {'ok' if inv_ok else 'FAIL'}")
    ok &= inv_ok

    # turning in place: no translation, pure rotation
    q = rk4_step([0.0, 0.0, 0.0], [0.0, 1.0], 0.25)
    spin_ok = np.allclose(q[:2], 0.0) and np.isclose(q[2], 0.25)
    print(f"turn in place          {'ok' if spin_ok else 'FAIL'}")
    ok &= spin_ok

    # A full circle returns to the start. dt is chosen to divide the period
    # exactly -- taking int(period/dt) steps would leave a fractional step
    # uncovered, and the resulting gap (v * leftover) would look like
    # integrator error when it is really just truncation in the test.
    v, omega, n_steps = 1.0, 0.5, 10_000
    dt = (2.0 * np.pi / omega) / n_steps
    q = np.array([0.0, 0.0, 0.0])
    for _ in range(n_steps):
        q = rk4_step(q, [v, omega], dt)
    closure = float(np.hypot(q[0], q[1]))
    circ_ok = closure < 1e-9
    print(f"closed circle          {'ok' if circ_ok else 'FAIL'}"
          f"   (returns within {closure:.2e} m)")
    ok &= circ_ok

    # the circle's radius must be v/omega
    v, omega, n_steps = 1.0, 0.5, 10_000
    dt = (2.0 * np.pi / omega) / n_steps
    q, xs, ys = np.array([0.0, 0.0, 0.0]), [], []
    for _ in range(n_steps):
        q = rk4_step(q, [v, omega], dt)
        xs.append(q[0]); ys.append(q[1])
    measured_r = 0.5 * (max(xs) - min(xs))
    rad_ok = np.isclose(measured_r, v / omega, atol=1e-9)
    print(f"turn radius = v/omega  {'ok' if rad_ok else 'FAIL'}"
          f"   ({measured_r:.6f} vs {v/omega:.6f} m)")
    ok &= rad_ok

    # heading stays wrapped
    q = np.array([0.0, 0.0, 3.0])
    for _ in range(100):
        q = rk4_step(q, [0.0, 1.0], 0.1)
    wrap_ok = -np.pi <= q[2] <= np.pi
    print(f"heading wrapped        {'ok' if wrap_ok else 'FAIL'}")
    ok &= wrap_ok

    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    _self_test()

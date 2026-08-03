"""
Motion model for a four-wheeled robot that steers by driving its left and
right sides at different speeds.

State is [x, y, heading]. Commands are forward speed and turn rate.
"""

import numpy as np

WHEEL_RADIUS = 0.10    # meters
TRACK_WIDTH = 0.45     # meters, from the left wheels to the right wheels


def wheel_speeds(speed, turn_rate):
    """How fast each side travels, in m/s.

    Turning left puts the left wheels on the inside of the curve, so they
    cover less ground and move slower than the right ones.
    """
    half = TRACK_WIDTH / 2
    left = speed - turn_rate * half
    right = speed + turn_rate * half
    return left, right


def wheel_spin_rates(speed, turn_rate):
    """How fast each wheel spins, in rad/s. This is what an encoder measures."""
    left, right = wheel_speeds(speed, turn_rate)
    return left / WHEEL_RADIUS, right / WHEEL_RADIUS


def speed_and_turn_from_wheels(left_speed, right_speed):
    """Work backwards from wheel speeds to the robot's speed and turn rate."""
    speed = (right_speed + left_speed) / 2
    turn_rate = (right_speed - left_speed) / TRACK_WIDTH
    return speed, turn_rate


def derivative(state, speed, turn_rate):
    """How fast x, y, and heading are changing right now."""
    heading = state[2]
    dx = speed * np.cos(heading)
    dy = speed * np.sin(heading)
    dheading = turn_rate
    return np.array([dx, dy, dheading])


def step(state, speed, turn_rate, dt):
    """Move the robot forward one time step.

    Takes four estimates of the derivative and averages them (Runge-Kutta 4).
    Using just one estimate is simpler but drifts on curves, and this state
    is the truth that everything else gets compared against.
    """
    k1 = derivative(state, speed, turn_rate)
    k2 = derivative(state + 0.5 * dt * k1, speed, turn_rate)
    k3 = derivative(state + 0.5 * dt * k2, speed, turn_rate)
    k4 = derivative(state + dt * k3, speed, turn_rate)

    new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # keep heading between -pi and pi so it doesn't grow forever
    new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
    return new_state


if __name__ == "__main__":
    print("Drive straight at 1 m/s for 0.1 s")
    state = step(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, 0.1)
    print("   ended at x=%.3f y=%.3f   (should be x=0.1, y=0)"
          % (state[0], state[1]))

    print("\nTurn left at 1 m/s, 0.5 rad/s")
    left, right = wheel_speeds(1.0, 0.5)
    print("   left wheels %.3f m/s, right wheels %.3f m/s" % (left, right))
    print("   left is slower, which is right for a left turn")

    print("\nTurn right at 1 m/s, -0.5 rad/s")
    left, right = wheel_speeds(1.0, -0.5)
    print("   left wheels %.3f m/s, right wheels %.3f m/s" % (left, right))
    print("   now right is slower, so the sides swapped")

    print("\nGo from wheel speeds back to commands")
    left, right = wheel_speeds(0.8, -0.35)
    speed, turn = speed_and_turn_from_wheels(left, right)
    print("   started with speed=0.80 turn=-0.35")
    print("   recovered    speed=%.2f turn=%.2f" % (speed, turn))

    print("\nSpin in place at 1 rad/s for 0.25 s")
    state = step(np.array([0.0, 0.0, 0.0]), 0.0, 1.0, 0.25)
    print("   position %.3f, %.3f   heading %.3f rad"
          % (state[0], state[1], state[2]))
    print("   it turned but didn't move, which is correct")

    print("\nDrive a full circle at 1 m/s, 0.5 rad/s")
    print("   the circle should have radius speed/turn_rate = 2.00 m")
    # one full circle takes 2*pi/turn_rate seconds; pick the number of steps
    # first so dt divides that time evenly, otherwise the loop stops just
    # short of the full circle and it looks like the math is off
    steps = 10000
    dt = (2 * np.pi / 0.5) / steps
    state = np.array([0.0, 0.0, 0.0])
    xs = []
    for i in range(steps):
        state = step(state, 1.0, 0.5, dt)
        xs.append(state[0])
    radius = (max(xs) - min(xs)) / 2
    distance_from_start = np.sqrt(state[0] ** 2 + state[1] ** 2)
    print("   measured radius %.3f m" % radius)
    print("   ended %.9f m from where it started" % distance_from_start)

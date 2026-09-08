"""
Attitude kinematics for a quadcopter, which is where the nonlinearity lives.

WHY THIS FOLDER EXISTS

The ground robot's measurement map turned out to be exactly linear. Its
encoders report

    left  = (speed - turn_rate * W/2) / r
    right = (speed + turn_rate * W/2) / r
    gyro  =  turn_rate

which is a three-by-two matrix, six numbers, and superposition holds to 1e-14.
That was checked rather than assumed, and it has an uncomfortable consequence:
an ordinary least-squares fit beats the trained network on every channel, by
three to seven per cent. The network approximates a straight line with a
piecewise-linear ReLU stack and pays for the kinks.

So the claim "a learned measurement model earns its place" cannot be tested
there. Whatever the learned arms buy on that robot, it is not nonlinearity --
it is having sensor health among their inputs, which is a different argument
and one the linear map does not touch.

A quadcopter's attitude problem is not like that. What an accelerometer reads
depends on where gravity points in the body frame, and what a magnetometer
reads depends on where north points in the body frame. Both go through a
rotation matrix built from sines and cosines of the attitude, so the map from
state to measurement is trigonometric and genuinely not expressible as a
matrix.

That makes this the setting where the learned-map question can actually be
asked. Whether the answer is yes is measured in linearity.py, not assumed
here.

WHAT IS MODELLED AND WHAT IS NOT

Six states:

    [roll, pitch, yaw, p, q, r]

the three Euler angles and the three body-frame angular rates. Position and
velocity are left out on purpose. They would need GPS to be observable, GPS is
a linear measurement, and adding a linear channel to a study about
nonlinearity would only dilute it.

Near-hover is assumed, so the accelerometer sees gravity plus a small
disturbance rather than large translational acceleration. That is the regime
attitude estimation is usually posed in, and it keeps the nonlinearity in the
rotation rather than in the flight dynamics.
"""

import numpy as np

GRAVITY = 9.81                 # m/s^2

# Earth's magnetic field as a unit vector in the world frame, at roughly the
# inclination of Boulder, Colorado -- about 66 degrees below horizontal,
# pointing north. Only the direction matters; a magnetometer is calibrated to
# unit length before it reaches an estimator.
MAG_INCLINATION = np.radians(66.0)
MAG_WORLD = np.array([np.cos(MAG_INCLINATION), 0.0, -np.sin(MAG_INCLINATION)])


def rotation_world_to_body(roll, pitch, yaw):
    """The matrix taking a world-frame vector into the body frame.

    The usual aerospace 3-2-1 sequence: yaw about z, then pitch about the new
    y, then roll about the new x. Transposed relative to the body-to-world
    form, which is what a sensor reading needs -- the sensor sits in the body
    and measures world quantities as they appear from there.

    Written for a single attitude. Vectorised over many at once by
    rotate_many, which is what the filter actually calls.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp * cy, cp * sy, -sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
    ])


def rotate_many(roll, pitch, yaw, vector):
    """Take one world vector into the body frame, for many attitudes at once.

    Shape (n,) angles and a (3,) vector give back (n, 3). Doing this without a
    Python loop matters for the same reason it did on the robot: the filter
    evaluates every sigma point at every step, and a loop over twenty-five
    points inside a fifty-hertz update is not free.
    """
    roll = np.atleast_1d(roll)
    pitch = np.atleast_1d(pitch)
    yaw = np.atleast_1d(yaw)

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = vector

    return np.column_stack([
        (cp * cy) * vx + (cp * sy) * vy + (-sp) * vz,
        (sr * sp * cy - cr * sy) * vx + (sr * sp * sy + cr * cy) * vy
        + (sr * cp) * vz,
        (cr * sp * cy + sr * sy) * vx + (cr * sp * sy - sr * cy) * vy
        + (cr * cp) * vz,
    ])


def euler_rates(roll, pitch, p, q, r):
    """How the Euler angles change, given the body rates.

    Body rates are not the derivatives of the Euler angles. The three rotation
    axes are not orthogonal to each other -- yaw is measured about the world's
    vertical, pitch about an intermediate axis, roll about the body's own --
    so converting between them takes a matrix that depends on the attitude
    itself:

        d(roll)  = p + sin(roll) tan(pitch) q + cos(roll) tan(pitch) r
        d(pitch) =            cos(roll) q -            sin(roll) r
        d(yaw)   = sin(roll)/cos(pitch) q + cos(roll)/cos(pitch) r

    The tangent and the division by cos(pitch) are why this blows up at
    pitch = +-90 degrees, which is gimbal lock. A quadcopter in ordinary
    flight does not go near it, and trajectories.py keeps well clear, but it
    is the reason a serious implementation would carry a quaternion instead.
    """
    tp = np.tan(pitch)
    cp = np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)

    d_roll = p + sr * tp * q + cr * tp * r
    d_pitch = cr * q - sr * r
    d_yaw = (sr * q + cr * r) / cp
    return d_roll, d_pitch, d_yaw


def step(state, dt):
    """Move one attitude state forward by dt.

    The rates are held constant across the step and the angles integrated
    from them. That is the same first-order scheme robot/dynamics.py uses, and
    it is accurate enough at fifty hertz for rates that change on the scale of
    a second.
    """
    roll, pitch, yaw, p, q, r = state[:6]
    d_roll, d_pitch, d_yaw = euler_rates(roll, pitch, p, q, r)

    return np.array([
        roll + d_roll * dt,
        pitch + d_pitch * dt,
        yaw + d_yaw * dt,
        p, q, r,
    ])


def specific_force(roll, pitch, yaw, disturbance=None):
    """What an ideal accelerometer reads, in the body frame.

    An accelerometer does not measure acceleration. It measures specific
    force: the sum of every force except gravity, divided by mass. Sitting
    still on a bench it reads +g upward, because the bench is pushing up and
    gravity is the one thing it cannot feel.

    Near hover the vehicle's own acceleration is small, so what remains is
    gravity as seen from the body, which depends entirely on attitude. That is
    the nonlinearity: three sines and cosines multiplied together.
    """
    reading = rotate_many(roll, pitch, yaw, np.array([0.0, 0.0, -GRAVITY]))
    if disturbance is not None:
        reading = reading + disturbance
    return -reading


def magnetic_field(roll, pitch, yaw):
    """What an ideal magnetometer reads, in the body frame.

    The same rotation applied to a different world vector. Two channels
    measuring two fixed directions from a moving frame is what makes attitude
    observable at all -- gravity alone leaves yaw undetermined, since rotating
    about the vertical does not change where down is.
    """
    return rotate_many(roll, pitch, yaw, MAG_WORLD)


if __name__ == "__main__":
    print("Check 1: the rotation matrix is orthonormal\n")
    for angles in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-0.9, 0.7, -2.4)]:
        R = rotation_world_to_body(*angles)
        print("  roll %5.2f pitch %5.2f yaw %5.2f   R'R = I: %s   det = %+.6f"
              % (angles + (np.allclose(R.T @ R, np.eye(3)),
                           np.linalg.det(R))))

    print("\nCheck 2: rotate_many agrees with the single-attitude form\n")
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, size=(50, 3))
    v = np.array([0.2, -0.5, 0.9])
    one = np.array([rotation_world_to_body(*row) @ v for row in a])
    many = rotate_many(a[:, 0], a[:, 1], a[:, 2], v)
    print("  max difference %.2e" % np.abs(one - many).max())

    print("\nCheck 3: what the sensors read when the vehicle is level\n")
    f = specific_force(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    m = magnetic_field(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    print("  accelerometer  [%+.3f %+.3f %+.3f]   should be [0 0 %+.2f]"
          % (f[0], f[1], f[2], GRAVITY))
    print("  magnetometer   [%+.3f %+.3f %+.3f]   should point north and down"
          % (m[0], m[1], m[2]))

    print("\nCheck 4: is the measurement map nonlinear?\n")
    b = rng.uniform(-0.8, 0.8, size=(200, 3))
    c = rng.uniform(-0.8, 0.8, size=(200, 3))

    def h(angles):
        return np.hstack([specific_force(angles[:, 0], angles[:, 1],
                                         angles[:, 2]),
                          magnetic_field(angles[:, 0], angles[:, 1],
                                         angles[:, 2])])

    lhs = h(3.0 * b + 2.0 * c)
    rhs = 3.0 * h(b) + 2.0 * h(c)
    print("  max |h(3a+2b) - (3h(a)+2h(b))| = %.4f" % np.abs(lhs - rhs).max())
    print("  the same check on the ground robot gives 1.4e-14, because that")
    print("  map is a matrix. This one is not, and that is the point.")

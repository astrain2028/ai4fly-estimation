"""
An Unscented Kalman Filter for the robot.

The filter estimates five things:

    [x, y, heading, speed, turn_rate]

Position and heading move according to the kinematics. Speed and turn rate
have no model at all -- they are assumed to drift slowly, and the encoders
and gyro are what pin them down. That is the point of the filter: it works
out how fast the robot is going from the wheel readings.

WHY UNSCENTED AND NOT PLAIN KALMAN

A plain Kalman filter needs the motion to be a matrix multiply. Ours has
cos(heading) and sin(heading) in it, so there is no such matrix once the
heading is uncertain.

The unscented filter deals with that by picking a handful of sample points
around the current estimate, pushing each one through the real motion model,
and rebuilding an average and a spread from where they land. No derivatives,
no linearising.
"""

import numpy as np

from dynamics import step, wheel_spin_rates


def sigma_points(mean, cov, alpha=1.0, beta=2.0, kappa=0.0):
    """Pick 2n+1 sample points that carry the same mean and spread as the
    Gaussian described by (mean, cov).

    These are not random draws. They are chosen so their weighted average is
    exactly `mean` and their weighted spread is exactly `cov`.
    """
    n = len(mean)
    lam = alpha ** 2 * (n + kappa) - n

    # cholesky gives a matrix that, times its own transpose, is cov.
    # Its columns are the directions to step along, scaled by the spread.
    spread = np.linalg.cholesky((n + lam) * cov)

    points = np.zeros((2 * n + 1, n))
    points[0] = mean
    for i in range(n):
        points[1 + i] = mean + spread[:, i]
        points[1 + n + i] = mean - spread[:, i]

    w_mean = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    w_cov = w_mean.copy()
    w_mean[0] = lam / (n + lam)
    w_cov[0] = w_mean[0] + (1 - alpha ** 2 + beta)
    return points, w_mean, w_cov


def rebuild(points, w_mean, w_cov, extra_noise):
    """Turn a cloud of points back into an average and a spread."""
    mean = (w_mean[:, None] * points).sum(axis=0)
    offset = points - mean
    cov = np.zeros((len(mean), len(mean)))
    for i in range(len(points)):
        cov = cov + w_cov[i] * np.outer(offset[i], offset[i])
    return mean, cov + extra_noise


def move_state(state, dt):
    """The motion model the filter uses.

    Position and heading follow the kinematics, driven by the speed and turn
    rate the filter currently believes. Those last two are left alone -- the
    process noise Q is what lets them change.
    """
    x, y, heading, speed, turn_rate = state
    moved = step(np.array([x, y, heading]), speed, turn_rate, dt)
    return np.array([moved[0], moved[1], moved[2], speed, turn_rate])


def expected_readings(states):
    """What the sensors should report for each state.

    Takes ALL the sigma points at once, shape (n_points, 5), and returns
    shape (n_points, 3). Handling them in one go matters when this function
    is a neural network: calling it eleven separate times per step is far
    slower than calling it once with eleven rows.
    """
    states = np.atleast_2d(states)
    speed, turn_rate = states[:, 3], states[:, 4]
    left, right = wheel_spin_rates(speed, turn_rate)
    return np.column_stack([left, right, turn_rate])


class UKF:
    """The filter itself.

    Q is how much the motion model is distrusted, R how much the sensors are.
    Pass a different R at each update to make sensor trust vary with time --
    which is the whole point of a learned noise model.
    """

    def __init__(self, Q, R, move=move_state, measure=expected_readings,
                 alpha=1.0, beta=2.0, kappa=0.0):
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.move = move
        self.measure = measure
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

    def predict(self, mean, cov, dt):
        """Push the estimate forward in time."""
        points, w_mean, w_cov = sigma_points(mean, cov, self.alpha,
                                             self.beta, self.kappa)
        moved = np.array([self.move(p, dt) for p in points])
        return rebuild(moved, w_mean, w_cov, self.Q)

    def update(self, mean, cov, reading, R=None):
        """Correct the estimate using one set of sensor readings.

        Returns the corrected estimate, plus the innovation and its
        covariance -- those two are what a consistency check needs, so they
        come back rather than being thrown away.
        """
        if R is None:
            R = self.R

        points, w_mean, w_cov = sigma_points(mean, cov, self.alpha,
                                             self.beta, self.kappa)

        # what each sample point says the sensors should read, all at once
        predicted = np.asarray(self.measure(points))
        z_mean, S = rebuild(predicted, w_mean, w_cov, R)

        # how state and measurement vary together
        cross = np.zeros((len(mean), len(z_mean)))
        for i in range(len(points)):
            cross = cross + w_cov[i] * np.outer(points[i] - mean,
                                                predicted[i] - z_mean)

        gain = cross @ np.linalg.inv(S)
        innovation = np.asarray(reading) - z_mean

        new_mean = mean + gain @ innovation
        new_cov = cov - gain @ S @ gain.T
        return new_mean, new_cov, innovation, S

    def run(self, readings, start_mean, start_cov, dt, R_series=None):
        """Filter a whole run of measurements.

        R_series, if given, supplies a different R for each time step.
        """
        n_steps = len(readings)
        means = np.zeros((n_steps, len(start_mean)))
        covs = np.zeros((n_steps, len(start_mean), len(start_mean)))
        innovations = np.zeros((n_steps, readings.shape[1]))
        innovation_covs = np.zeros((n_steps, readings.shape[1],
                                    readings.shape[1]))

        mean = np.asarray(start_mean, dtype=float)
        cov = np.asarray(start_cov, dtype=float)

        for k in range(n_steps):
            mean, cov = self.predict(mean, cov, dt)
            R = None if R_series is None else R_series[k]
            mean, cov, innovation, S = self.update(mean, cov, readings[k], R)

            means[k] = mean
            covs[k] = cov
            innovations[k] = innovation
            innovation_covs[k] = S

        return means, covs, innovations, innovation_covs


def nis(innovations, innovation_covs):
    """Normalised innovation squared, one number per time step.

    If the filter's idea of its own uncertainty is right, these should
    average about the number of sensors.
    """
    out = np.zeros(len(innovations))
    for k in range(len(innovations)):
        out[k] = innovations[k] @ np.linalg.solve(innovation_covs[k],
                                                  innovations[k])
    return out


if __name__ == "__main__":
    from trajectories import DT, random_run
    from sensors import read_sensors

    # ---- check 1: on a linear problem the UKF must be exactly right ----
    print("Check 1: linear motion, where the answer is known exactly")
    A = np.array([[1.0, 0.5], [0.0, 1.0]])
    mean0 = np.array([2.0, -1.0])
    cov0 = np.array([[0.30, 0.05], [0.05, 0.20]])
    pts, wm, wc = sigma_points(mean0, cov0)
    moved = np.array([A @ p for p in pts])
    got_mean, got_cov = rebuild(moved, wm, wc, np.zeros((2, 2)))
    print("   mean       matches: %s" % np.allclose(got_mean, A @ mean0))
    print("   covariance matches: %s" % np.allclose(got_cov, A @ cov0 @ A.T))
    print("   (sigma points reproduce a linear map perfectly, as they must)")

    # ---- check 2: run it on a real simulated drive ----
    print("\nCheck 2: filtering an actual run")
    run = random_run(seed=0, duration=20.0)
    meas = read_sensors(run, seed=0, dt=DT)
    readings = np.column_stack([meas["left_encoder"],
                                meas["right_encoder"],
                                meas["gyro"]])

    # How much to distrust the motion model, per step. Position and heading
    # follow the kinematics almost exactly, so those get almost nothing.
    # Speed and turn rate have no model at all, so Q is what lets them move.
    #
    # These two numbers are not guesses. Speed changes by at most 0.0068 m/s
    # in one step and turn rate by 0.0100 rad/s, so squaring those gives the
    # variance the filter should expect per step. Picking Q by hand instead
    # gets it wrong by orders of magnitude, and the NIS check below is what
    # catches that.
    Q = np.diag([1e-9, 1e-9, 1e-9, 2e-5, 1e-4])

    # how much to distrust each sensor -- roughly their true noise
    R = np.diag([0.15 ** 2, 0.15 ** 2, 0.011 ** 2])

    start_mean = np.array([0.0, 0.0, 0.0, run["speed"][0], run["turn_rate"][0]])
    start_cov = np.diag([0.01, 0.01, 0.01, 0.10, 0.10])

    ukf = UKF(Q, R)
    means, covs, innovations, S = ukf.run(readings, start_mean, start_cov, DT)

    speed_error = np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2))
    turn_error = np.sqrt(np.mean((means[:, 4] - run["turn_rate"]) ** 2))
    print("   speed off by     %.4f m/s" % speed_error)
    print("   turn rate off by %.4f rad/s" % turn_error)

    raw_turn = (readings[:, 1] - readings[:, 0]) * 0.10 / 0.45
    print("   for comparison, turn rate straight off the encoders: %.4f"
          % np.sqrt(np.mean((raw_turn - run["turn_rate"]) ** 2)))

    values = nis(innovations, S)
    print("\n   average NIS %.2f  (should be near 3, the number of sensors)"
          % values.mean())
    print("   Too low means the filter is under-confident: it expected more")
    print("   surprise than it got, usually because Q or R is too big.")
    print("   Too high means over-confident, which is the dangerous side --")
    print("   everything downstream believes an estimate it should not.")

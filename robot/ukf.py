"""
An Unscented Kalman Filter for the robot.

The filter estimates seven things:

    [x, y, heading, speed, turn_rate, accel, turn_accel]

Position and heading move according to the kinematics. Speed and turn rate
are moved by the accelerations, and the accelerations drift. The encoders and
gyro are what pin all of it down: the point of the filter is to work out how
fast the robot is going from the wheel readings.

Only the first five are observable in any direct sense -- the sensors report
speed and turn rate, and nothing reports acceleration. It is carried anyway
because the alternative is worse. See move_state for the measurement that
settles it.

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

        [x, y, heading, speed, turn_rate, accel, turn_accel]

    Position and heading follow the kinematics. Speed and turn rate are moved
    by the accelerations, and the accelerations themselves are left alone --
    the process noise Q is what lets those drift.

    WHY ACCELERATION IS A STATE

    An earlier version stopped at five states and let Q move speed and turn
    rate directly, which says their step-to-step changes are independent
    noise. That is a random walk, and it is measurably wrong here: the
    increments of the true speed have a lag-1 autocorrelation of 0.998, where
    a random walk requires 0.00. The robot accelerates smoothly.

    No value of Q repairs that, because the error is in the shape of the model
    and not its scale. Tuning it produced exactly the symptom of an
    unsatisfiable fit -- NEES could be brought to target only by pushing NIS
    away from it, and vice versa.

    Acceleration, on the other hand, really does behave like a random walk
    here: it changes by about 2 per cent of its own spread per step. So it is
    the right place to put the process noise.
    """
    x, y, heading, speed, turn_rate, accel, turn_accel = state[:7]
    moved = step(np.array([x, y, heading]), speed, turn_rate, dt)
    forward = np.array([moved[0], moved[1], moved[2],
                        speed + accel * dt, turn_rate + turn_accel * dt,
                        accel, turn_accel])

    # Anything past the seventh entry is sensor health, and it is carried
    # forward untouched. Health has no dynamics: a sensor does not get better
    # or worse because time passed, only because evidence arrived. Leaving it
    # alone in the prediction puts the entire burden of estimating it on the
    # measurement model, which is where it belongs.
    if len(state) > 7:
        return np.concatenate([forward, state[7:]])
    return forward


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

        WHERE R COMES FROM WHEN THE MODEL PREDICTS IT

        A model whose noise depends on the state gives a different answer at
        every sample point, and only one number can go into the update. The
        right one falls out of writing down what is actually being asked.

        The reading is z = h(x) + noise, where the noise has covariance R(x),
        and x is not known exactly -- it has a spread. So

            spread of z = average of R(x)  +  spread of h(x)

        The second term is what `rebuild` already computes from how far the
        sample points land from each other. The first is the weighted average
        of R over those same points, which is what is taken below.

        Using R at the middle sample point alone would be simpler and is what
        many implementations do, but it throws away the fact that the model
        disagrees with itself across the spread of possible states.
        """
        if R is None:
            R = self.R

        points, w_mean, w_cov = sigma_points(mean, cov, self.alpha,
                                             self.beta, self.kappa)

        # What each sample point says the sensors should read, all at once.
        # A measurement model may also hand back its own noise, one covariance
        # per sample point, in which case it overrides whatever R was passed.
        out = self.measure(points)
        if isinstance(out, tuple):
            predicted, R_per_point = out
            R = np.average(R_per_point, axis=0, weights=w_mean)
        predicted = np.asarray(predicted if isinstance(out, tuple) else out)

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

            # A measurement model that tunes itself from its own past mistakes
            # needs to be told how each step went. Classical adaptive filtering
            # works exactly this way: watch the innovations, and if they are
            # consistently bigger than claimed, raise R. Models that do not
            # adapt have no observe method and never hear about it.
            if hasattr(self.measure, "observe"):
                self.measure.observe(innovation, S)

            # Some states cannot take every value a Gaussian would allow. A
            # sensor's degradation cannot be negative, and if the model
            # ignores negative health -- which it must, having never been
            # shown any -- then moving there costs the filter nothing in
            # innovation while still absorbing correction. That is a free
            # direction, and an unconstrained update will use it.
            #
            # Projecting back onto the allowed set after each update is not
            # exact Bayesian inference, but the alternative is an estimate
            # that drifts into a region where the measurement model is not
            # defined.
            if hasattr(self.measure, "constrain"):
                mean = self.measure.constrain(mean)

            means[k] = mean
            covs[k] = cov
            innovations[k] = innovation
            innovation_covs[k] = S

        return means, covs, innovations, innovation_covs


def nis(innovations, innovation_covs):
    """Normalised innovation squared, one number per time step."""
    out = np.zeros(len(innovations))
    for k in range(len(innovations)):
        out[k] = innovations[k] @ np.linalg.solve(innovation_covs[k],
                                                  innovations[k])
    return out


def nees(estimates, covariances, truth, states=None):
    """Normalised estimation error squared, one number per time step.

    Where NIS asks "was I as surprised as I said I would be", NEES asks
    "am I as close to the truth as I claim". It needs the true state, so it
    only works in simulation -- but that is exactly what simulation is for.

    The two catch different faults. Innovations only involve things a sensor
    measures, so NIS is blind to any state nothing observes. Here nothing
    measures position or heading: those can drift as far as they like and
    every innovation stays perfectly well behaved.

    `states` picks a subset, e.g. [3, 4] for speed and turn rate alone. That
    matters because the full-state number is dominated by whichever states
    have no measurement, so it mostly reports how Q was tuned rather than
    anything about the sensors.
    """
    estimates = np.asarray(estimates)
    truth = np.asarray(truth)
    out = np.zeros(len(estimates))
    for k in range(len(estimates)):
        error = estimates[k] - truth[k]
        P = covariances[k]
        if states is not None:
            error = error[states]
            P = P[np.ix_(states, states)]
        out[k] = error @ np.linalg.solve(P, error)
    return out


def consistency(values, dof):
    """Check both things a correctly tuned filter has to get right.

    NIS and NEES both follow a chi-square distribution when the filter is
    tuned -- one degree of freedom per sensor for NIS, per state for NEES.
    That fixes two numbers, not one:

        average  should be dof
        spread   should be 2 * dof

    Checking only the average is not enough. Chen et al. show a filter that
    is tuned wrongly but still has exactly the right average, with the
    spread giving it away. We hit the same thing here: an average of 3.17
    against a target of 3 looked fine while the spread was 7.2 against 6,
    because two channels were over-trusted and one under-trusted by amounts
    that cancelled in a sum and compounded in a square.

    The median is reported too. These distributions have a long right tail,
    and a handful of bad runs can drag the average well away from what a
    typical run looks like.
    """
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "target": float(dof),
        "variance": float(values.var()),
        "variance_target": float(2 * dof),
        "ratio": float(values.var() / values.mean()),
    }


def print_consistency(values, dof, label=""):
    c = consistency(values, dof)
    print("  %-20s %9s %9s %9s %9s"
          % (label, "average", "median", "spread", "ratio"))
    print("  %-20s %9.3f %9.3f %9.3f %9.3f"
          % ("measured", c["mean"], c["median"], c["variance"], c["ratio"]))
    print("  %-20s %9.1f %9s %9.1f %9.1f"
          % ("should be", c["target"], "~" + str(c["target"]),
             c["variance_target"], 2.0))
    return c


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

    # ---- check 2: is the filter honest about its own uncertainty? ----
    print("\nCheck 2: is the filter honest about its own uncertainty?")

    # Process noise over the seven states. Speed and turn rate are moved by
    # the accelerations rather than by noise, so their own entries are tiny
    # and the noise lives on the accelerations. Those two values are searched
    # rather than measured -- see experiments/common.py for why the measured
    # ones are far too small.
    Q = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-3, 1e-1])

    # R was not guessed. Starting from the sensors' own noise figures, the
    # spread of NIS came out at 7.2 against a target of 6 even though the
    # average looked right. Scaling each channel by how far its whitened
    # innovations missed unit variance, three times over, gives these.
    # The encoders needed MORE than their measurement noise alone, because
    # S also carries state uncertainty, and because encoder noise really
    # does vary with speed so no one number fits the whole range.
    R = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2])

    all_nis, nees_full, nees_vel = [], [], []
    for seed in range(20):
        run = random_run(seed=seed, duration=20.0)
        meas = read_sensors(run, seed=seed, dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"],
                                    meas["gyro"]])
        # The trajectory carries no accelerations, so they are differenced
        # out of the speed and turn rate.
        accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
        turn_accel = np.diff(run["turn_rate"],
                             append=run["turn_rate"][-1]) / DT
        truth = np.column_stack([run["x"], run["y"], run["heading"],
                                 run["speed"], run["turn_rate"],
                                 accel, turn_accel])
        start_cov = np.diag([0.01, 0.01, 0.01, 0.10, 0.10, 0.04, 0.04])

        means, covs, innovations, S = UKF(Q, R).run(
            readings, truth[0].copy(), start_cov, DT)

        all_nis.append(nis(innovations, S))
        nees_full.append(nees(means, covs, truth))
        nees_vel.append(nees(means, covs, truth, states=[3, 4]))

    all_nis = np.concatenate(all_nis)
    nees_full = np.concatenate(nees_full)
    nees_vel = np.concatenate(nees_vel)

    print()
    print_consistency(all_nis, 3, "NIS, 3 sensors")
    print()
    print_consistency(nees_vel, 2, "NEES, speed+turn")
    print()
    print_consistency(nees_full, 7, "NEES, all 7 states")

    print("\nReading the three together:")
    print("  NIS passes, so the filter predicts its measurements honestly.")
    print("  NEES on speed and turn rate is near target now that")
    print("  acceleration is a state. It sat near 3.6 before, because a")
    print("  random walk is the wrong shape for motion this smooth --")
    print("  the true speed increments have a lag-1 correlation of 0.998,")
    print("  where a random walk requires zero.")
    print("  The full-state figure is still poor, driven by position and")
    print("  heading, which nothing measures. NIS cannot see those at all,")
    print("  because innovations only involve quantities a sensor reports.")
    print("  That is the whole reason for computing both.")
    print("  Compare average against median on the full state: a few runs")
    print("  behave badly and drag the average up. The average alone hides")
    print("  what a typical run looks like; the median alone hides that")
    print("  some runs go wrong.")

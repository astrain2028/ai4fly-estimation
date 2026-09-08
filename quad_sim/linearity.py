"""
Is there anything here for a nonlinear model to learn?

THE QUESTION THIS FOLDER EXISTS TO ANSWER

On the ground robot the measurement map is exactly linear, and an ordinary
least-squares fit beat the trained network on every channel by three to seven
per cent. The network was approximating a straight line with a piecewise-linear
ReLU stack and paying for the kinks. So on that problem "we learned a nonlinear
measurement model" describes the architecture rather than anything the data
required.

Before building a learned model here it is worth asking whether this problem is
any different, and the question can be settled without training anything. Fit
the best possible linear map from state to reading, and compare what it leaves
behind against the sensor noise floor.

    residual near the noise floor   the map is effectively linear over this
                                    operating range, a matrix is the right
                                    model, and this folder has no premise

    residual well above the floor   there is structure a linear map cannot
                                    reach, and a nonlinear model has
                                    something to do

This is the honest order to do it in. Training first and reporting that the
network works would not distinguish the two cases, because a network can fit a
linear function too.

WHY THE ANSWER IS NOT OBVIOUS IN ADVANCE

Six of the nine channels go through a rotation matrix, which is trigonometric,
so the map is certainly not a matrix. But "not a matrix" and "far from a
matrix over the range actually flown" are different claims. Near level flight
sin(x) is close to x and cos(x) is close to 1, so a quadcopter that mostly
hovers would present a nearly linear problem however trigonometric the
underlying geometry is.

That is why trajectories.py drives the attitude hard enough to spend a third
of its samples past ten degrees of bank. Whether that is enough is what gets
measured here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from sensors import CHANNELS, ideal_readings, noise_levels, read_sensors, stack
from trajectories import DT, random_run, truth_matrix

TRAIN_RUNS = range(0, 40)
TEST_RUNS = range(100, 110)


def gather(seeds):
    """Truth and readings over several runs, stacked."""
    states, readings, floors = [], [], []
    for seed in seeds:
        run = random_run(seed)
        states.append(truth_matrix(run))
        readings.append(stack(read_sensors(run, seed=seed)))
        floors.append(noise_levels(run))
    return (np.vstack(states), np.vstack(readings), np.vstack(floors))


def fit_linear(states, readings):
    """The best linear map from state to reading, with an intercept."""
    design = np.hstack([states, np.ones((len(states), 1))])
    weights, *_ = np.linalg.lstsq(design, readings, rcond=None)
    return weights


def apply_linear(weights, states):
    return np.hstack([states, np.ones((len(states), 1))]) @ weights


def main():
    train_x, train_y, _ = gather(TRAIN_RUNS)
    test_x, test_y, test_floor = gather(TEST_RUNS)

    weights = fit_linear(train_x, train_y)
    predicted = apply_linear(weights, test_x)

    # The truth without noise, so the linear fit's error can be separated
    # into "noise it could never have predicted" and "structure it missed".
    clean = np.vstack([ideal_readings(*truth_matrix(random_run(s)).T[:6])
                       for s in TEST_RUNS])

    print("HOW MUCH OF THIS MAP IS LINEAR?\n")
    print("%d runs fitted, %d held out. The best linear map from six states"
          % (len(list(TRAIN_RUNS)), len(list(TEST_RUNS))))
    print("to nine channels, against the noise it cannot help.\n")
    print("  %-10s %12s %12s %12s %10s"
          % ("channel", "noise floor", "linear rmse", "unexplained", "ratio"))
    print("  " + "-" * 62)

    ratios = []
    for i, name in enumerate(CHANNELS):
        floor = test_floor[:, i].mean()
        rmse = np.sqrt(np.mean((predicted[:, i] - test_y[:, i]) ** 2))
        # What the linear map missed about the noiseless function itself.
        missed = np.sqrt(np.mean((predicted[:, i] - clean[:, i]) ** 2))
        ratios.append(missed / floor)
        print("  %-10s %12.5f %12.5f %12.5f %9.1fx"
              % (name, floor, rmse, missed, missed / floor))

    print("  " + "-" * 62)
    print("\n  'unexplained' is how far the linear fit sits from the true")
    print("  noiseless reading. 'ratio' compares that to the noise floor.")
    print("  Below 1 means the linear map is inside the noise and nothing")
    print("  could tell the difference. Well above 1 means there is real")
    print("  structure left on the table.")

    worst = max(ratios)
    print("\n\nVERDICT\n")
    print("  largest ratio across the nine channels: %.1fx" % worst)
    if worst < 1.0:
        print("\n  The linear map is within the noise on every channel. This")
        print("  problem is effectively linear over the range flown, and a")
        print("  learned nonlinear model would have nothing to add. Either")
        print("  fly harder attitudes or drop the premise.")
    else:
        print("\n  A linear map leaves structure well above the noise floor.")
        print("  That is what the ground robot could not offer -- there the")
        print("  same fit was exact to 1e-14 and least squares beat the")
        print("  network. Here a nonlinear model has something to do, and")
        print("  whether it does it is the next thing to measure.")

    print("\n\nWHERE THE NONLINEARITY LIVES\n")
    print("  %-10s %10s" % ("channel", "ratio"))
    order = np.argsort(ratios)[::-1]
    for i in order:
        bar = "#" * int(min(ratios[i], 40))
        print("  %-10s %9.1fx  %s" % (CHANNELS[i], ratios[i], bar))
    print("\n  The magnetometer is worst because yaw runs right around, so")
    print("  its sines and cosines are exercised over their whole range. The")
    print("  accelerometer only sees roll and pitch, which stay under about")
    print("  forty degrees, and over that range sin(x) is close enough to x")
    print("  that accel_x and accel_y sit inside the noise. Only accel_z,")
    print("  which carries cos(roll)cos(pitch), clears it.")
    print("\n  The gyro rows are not quite zero, and that is not curvature.")
    print("  Rates are measured directly, so their part of the map is the")
    print("  identity and a linear fit recovers it exactly. What is left is")
    print("  the per-run gyro bias: the fit is trained on runs whose biases")
    print("  do not average to zero, so it absorbs a little of that into the")
    print("  intercept. A bias no measurement model can predict is a floor")
    print("  under those channels rather than evidence about the map.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
TEST 1 -- IS THE MAP RIGHT?

Asks what a hand-written measurement model is worth when the vehicle is
not quite the vehicle it was written for.

Everywhere else in this project the analytic model wins, and it should: the
simulator generates readings from the same equations the model uses, so the
model is not an approximation of the truth, it is the truth. Nothing learned
from data can beat that, and the healthy comparison confirms it -- the
heteroscedastic arm ties and no arm does better.

That is not a property of measurement models. It is a property of a
simulation in which somebody wrote down the vehicle exactly.

Real wheels are not exactly the radius on the drawing. Tyres wear, pressure
changes, and the track width is a ruler measurement between two contact
patches that are not points. So the equations are slightly wrong about the
vehicle they are attached to, and being slightly wrong about a multiplier
shows up as a bias that grows with speed. At 12 rad/s of wheel spin, a 3 per
cent radius error is 0.33 rad/s -- close to twice the sensor noise, entirely
systematic, and invisible to anyone who trusts the equations.

A learned model has no such loyalty. It fits whatever relationship the data
actually shows, which is the relationship the real vehicle has. So the
question here is how much calibration error it takes before learning the
measurement model beats deriving it.

WHAT THIS DOES AND DOES NOT ARGUE

It argues for learning the measurement map. It does not, on its own, argue
for a state-dependent covariance -- a plain network fits the same biased
relationship equally well, and the two learned arms should move together.
Separating those two claims is why the plain arm is here.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from common import (NEES_DOF, NIS_DOF, best_constant_R, filter_once,
                    NEES_STATES, two_moment)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import dynamics
import make_dataset
import sensors
from trajectories import DT, random_run
from ukf import expected_readings

from heteroscedasticity import train_heteroscedastic

# How wrong the written-down wheel radius is. Zero reproduces the condition
# every other experiment here runs under.
ERRORS = [0.0, 0.01, 0.03]

TRAIN_RUNS = 100
EVAL_RUNS = 10


def train_plain(frame):
    """A learned mean with no covariance of its own.

    Included to separate two claims that are easy to confuse. If the plain
    and heteroscedastic arms move together as calibration error grows, the
    gain is coming from learning the map, not from learning the noise.
    """
    import torch.nn as nn

    torch.manual_seed(0)
    inputs = ["x", "y", "heading", "speed", "turn_rate"]
    outputs = ["left_encoder", "right_encoder", "gyro"]

    x = torch.tensor(frame[inputs].values, dtype=torch.float32)
    y = torch.tensor(frame[outputs].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    xs, ys = (x - x_mean) / x_std, (y - y_mean) / y_std

    model = nn.Sequential(nn.Linear(5, 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for _ in range(60):
        order = torch.randperm(len(xs))
        for start in range(0, len(xs), 512):
            rows = order[start:start + 512]
            loss = loss_fn(model(xs[rows]), ys[rows])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    model.eval()

    def measure(states):
        states = np.atleast_2d(states)[:, :5]
        with torch.no_grad():
            out = model((torch.tensor(states, dtype=torch.float32) - x_mean)
                        / x_std) * y_std + y_mean
        return out.numpy().astype(float)

    return measure


def evaluate(measure, R, error, n_runs=EVAL_RUNS):
    """Filter runs from a vehicle that is `error` off its written-down size."""
    saved = dynamics.RADIUS_ERROR
    dynamics.RADIUS_ERROR = error
    try:
        speed, turn, nis_all, nees_all = [], [], [], []
        for seed in range(200, 200 + n_runs):
            run = random_run(seed, duration=20.0)
            meas = sensors.read_sensors(run, seed=seed, dt=DT)
            accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
            turn_accel = np.diff(run["turn_rate"],
                                 append=run["turn_rate"][-1]) / DT
            truth = np.column_stack([run["x"], run["y"], run["heading"],
                                     run["speed"], run["turn_rate"],
                                     accel, turn_accel])
            result = filter_once(measure, meas, truth, R)
            speed.append(result["speed_rmse"])
            turn.append(result["turn_rmse"])
            nis_all.append(result["nis"])
            nees_all.append(result["nees"])
    finally:
        dynamics.RADIUS_ERROR = saved

    return {"speed": float(np.mean(speed)), "turn": float(np.mean(turn)),
            "nis": np.concatenate(nis_all), "nees": np.concatenate(nees_all)}


class quiet:
    def __enter__(self):
        import io
        self._real, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *_):
        sys.stdout = self._real


def main():
    print("CALIBRATION ERROR")
    print("How far the real wheel radius is from the one the analytic model")
    print("was built with. The learned arms are retrained on data from the")
    print("real vehicle at each level, which is what collecting data means.\n")

    header = ("%-16s %-24s %8s %8s %9s %9s"
              % ("radius error", "arm", "speed", "turn", "NIS", "NEES"))
    print(header)
    print("-" * len(header))

    for error in ERRORS:
        frame = make_dataset.build(n_runs=TRAIN_RUNS, radius_error=error)
        R_const = best_constant_R(frame)

        with quiet():
            learned_mean = train_plain(frame)
            learned_both = train_heteroscedastic(frame)

        bias = (dynamics.WHEEL_RADIUS * error / dynamics.WHEEL_RADIUS) * 11.3
        arms = [("analytic (written down)", expected_readings, R_const),
                ("plain network", learned_mean, R_const),
                ("heteroscedastic", learned_both, R_const)]

        first = True
        for label, measure, R in arms:
            out = evaluate(measure, R, error)
            nis_mean, _, _, _ = two_moment(out["nis"], NIS_DOF)
            nees_mean, _, _, _ = two_moment(out["nees"], NEES_DOF)
            tag = ("%.0f%% (%.2f rad/s)" % (100 * error, bias)) if first else ""
            first = False
            print("%-16s %-24s %8.4f %8.4f %9.3f %9.3f"
                  % (tag, label, out["speed"], out["turn"], nis_mean,
                     nees_mean))
        print()

    print("-" * len(header))
    print("targets: NIS %.1f, NEES %.1f" % (NIS_DOF, NEES_DOF))
    print()
    print("The bias in brackets is what the analytic model is wrong by at a")
    print("typical wheel speed, against a sensor noise near 0.18 rad/s.")
    print()
    print("At zero error the analytic model is exactly correct and should")
    print("win. If it stops winning as the error grows, that is the argument")
    print("for learning the map -- and if the two learned arms move together,")
    print("the argument is for learning the map and not for learning the")
    print("noise, which is a different claim needing different evidence.")


if __name__ == "__main__":
    main()

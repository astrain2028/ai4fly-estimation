"""
TEST 2 -- IS THE NOISE STRUCTURE RIGHT?

Asks when a state-dependent noise model starts being worth its cost.

A learned covariance is only useful if the real noise varies with something
the model can see. How much it varies here is set by one number in sensors.py
-- the extra encoder noise per rad/s of wheel spin -- so that number can be
swept, and the answer becomes a curve rather than an assertion.

At the bottom of the sweep the noise is the same everywhere. A single
constant is then the correct answer, and a learned model cannot do better
than tie. That row is the control: it is where the method should fail, and
including it is what makes the rest of the sweep mean anything. Turning the
knob up until the learned model wins and reporting only that point would be
choosing the conditions to suit the conclusion.

Three arms, because they fail differently.

    analytic + best constant R   one number, chosen optimally for each level
    adaptive R                   estimated online from innovations, with lag
    heteroscedastic              predicted from the state, no lag, no tuning

The constant is recomputed at every level rather than held fixed, so it is
never handicapped: it always gets the best single value available to it,
which is the average variance over the run. Beating a badly chosen constant
would prove nothing.

WHAT THIS DOES NOT COVER

Only the aleatoric half. The epistemic term answers a different question --
whether the model has seen states like this before -- and inside the training
envelope it contributes almost nothing, so it will not show up here. Testing
that half needs states the model was not trained on, which is a second axis
crossed with this one, and a separate experiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util

import numpy as np
import torch

from common import NEES_DOF, NIS_DOF, P0, Q, filter_once, two_moment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import sensors
from trajectories import DT, random_run
from ukf import expected_readings

import make_dataset


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bhr = _load(ROOT / "models" / "bhr" / "train.py", "bhr_train_sweep")
adaptive = _load(ROOT / "models" / "adaptive" / "measurement.py",
                 "adaptive_sweep")

# Both sensors have their own growth constant, and the sweep scales both
# together. Scaling only the encoders leaves the gyro varying with turn rate,
# so the bottom row would not be homoscedastic and the control would not be a
# control -- which is exactly what happened on the first attempt here.
SCALES = [0.0, 1.0, 3.0, 8.0]
BASE_ENCODER = sensors.ENCODER_NOISE_GROWTH      # 0.05, the shipped setting
BASE_GYRO = sensors.GYRO_NOISE_GROWTH            # 0.03

EVAL_RUNS = 10
TRAIN_RUNS = 60          # fewer than the shipped 100, to keep the sweep short


def best_constant_R(frame):
    """The best single covariance for this noise level.

    A constant R cannot track anything, so the most it can do is match the
    average variance. Anything else is worse by construction, which is why
    this is computed rather than searched.
    """
    tick = (2 * np.pi / sensors.TICKS_PER_TURN) / DT
    quant = tick ** 2 / 12
    return np.diag([
        frame["left_noise"].pow(2).mean() + quant,
        frame["right_noise"].pow(2).mean() + quant,
        frame["gyro_noise"].pow(2).mean() + frame["gyro_bias"].var(),
    ])


def train_heteroscedastic(frame):
    """Train the heteroscedastic model in memory and wrap it for the filter.

    Nothing is written to disk. The shipped bhr_model.pt belongs to the
    default noise level and must not be overwritten by a sweep.
    """
    torch.manual_seed(0)
    runs = frame["run"].unique()
    val_runs = set(runs[:max(1, len(runs) // 5)])
    is_val = frame["run"].isin(val_runs)
    train_df, val_df = frame[~is_val], frame[is_val]

    x = torch.tensor(train_df[bhr.INPUTS].values, dtype=torch.float32)
    y = torch.tensor(train_df[bhr.OUTPUTS].values, dtype=torch.float32)
    xv = torch.tensor(val_df[bhr.INPUTS].values, dtype=torch.float32)
    yv = torch.tensor(val_df[bhr.OUTPUTS].values, dtype=torch.float32)

    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)

    model = bhr.make_model()
    bhr.train(model, (x - x_mean) / x_std, (y - y_mean) / y_std,
              (xv - x_mean) / x_std, (yv - y_mean) / y_std)
    model.eval()

    def measure(states):
        # The filter carries accelerations after speed and turn rate; the
        # model was trained on the five the sensors depend on.
        states = np.atleast_2d(states)[:, :len(bhr.INPUTS)]
        with torch.no_grad():
            raw = model((torch.tensor(states, dtype=torch.float32) - x_mean)
                        / x_std)
            eta1, eta2 = bhr.split_outputs(raw, False)
            mean_s, var_s = bhr.to_mean_and_var(eta1, eta2)
        readings = (mean_s * y_std + y_mean).numpy().astype(float)
        variances = (var_s * y_std ** 2).numpy().astype(float)
        R = np.zeros((len(states), 3, 3))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    return measure


def evaluate(measure, R, scale, n_runs=EVAL_RUNS):
    """Filter runs generated at this noise level.

    The sensor constants have to be set here too, not only during training.
    Evaluating a model trained at one noise level on runs generated at
    another would compare nothing meaningful.
    """
    saved = (sensors.ENCODER_NOISE_GROWTH, sensors.GYRO_NOISE_GROWTH)
    sensors.ENCODER_NOISE_GROWTH = BASE_ENCODER * scale
    sensors.GYRO_NOISE_GROWTH = BASE_GYRO * scale
    try:
        speed, nis_all, nees_all = [], [], []
        for seed in range(100, 100 + n_runs):
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
            nis_all.append(result["nis"])
            nees_all.append(result["nees"])
    finally:
        (sensors.ENCODER_NOISE_GROWTH, sensors.GYRO_NOISE_GROWTH) = saved

    return {"speed": float(np.mean(speed)),
            "nis": np.concatenate(nis_all),
            "nees": np.concatenate(nees_all)}


class quiet:
    """Swallow training chatter so the table stays readable."""

    def __enter__(self):
        import io
        self._real, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *_):
        sys.stdout = self._real


def main():
    print("HETEROSCEDASTICITY SWEEP")
    print("Both sensors' noise growth scaled together, 1.0 being the shipped")
    print("setting and 0.0 being noise that does not vary at all.")
    print("%d runs per point, model retrained at every level.\n" % EVAL_RUNS)

    header = ("%-8s %-11s %-24s %8s %9s %9s %9s"
              % ("scale", "variation", "arm", "speed", "NIS", "NIS var",
                 "NEES"))
    print(header)
    print("-" * len(header))

    for scale in SCALES:
        frame = make_dataset.build(encoder_growth=BASE_ENCODER * scale,
                                   gyro_growth=BASE_GYRO * scale,
                                   n_runs=TRAIN_RUNS)
        enc = 100 * make_dataset.observable_range(frame)
        gyr = 100 * make_dataset.gyro_range(frame)
        R_const = best_constant_R(frame)

        with quiet():
            learned = train_heteroscedastic(frame)

        arms = [
            ("analytic + best const R", expected_readings, R_const),
            ("adaptive R", adaptive.AdaptiveR(R_const), R_const),
            ("heteroscedastic", learned, R_const),
        ]

        first = True
        for label, measure, R in arms:
            out = evaluate(measure, R, scale)
            nis_mean, nis_var, _, _ = two_moment(out["nis"], NIS_DOF)
            nees_mean, _, _, _ = two_moment(out["nees"], NEES_DOF)
            tag = ("%.1f" % scale) if first else ""
            span = ("enc %.0f%%" % enc) if first else ""
            if first:
                first = False
            print("%-8s %-11s %-24s %8.4f %9.3f %9.3f %9.3f"
                  % (tag, span, label, out["speed"], nis_mean, nis_var,
                     nees_mean))
        print("%-8s %-11s" % ("", "gyro %.0f%%" % gyr))

    print("-" * len(header))
    print("targets: NIS %.1f, NIS var %.1f, NEES %.1f"
          % (NIS_DOF, 2.0 * NIS_DOF, NEES_DOF))
    print()
    print("Scale 0.0 is the control. The noise is the same everywhere, a")
    print("single constant is exactly right, and the learned model has")
    print("nothing to find -- it should tie at best and may lose slightly to")
    print("the cost of having to learn what it was told for free.")
    print()
    print("Read down the scale column rather than across the arms. The")
    print("question is where the constant stops being good enough, and")
    print("whether the learned model holds its consistency as the noise")
    print("varies more.")


if __name__ == "__main__":
    main()

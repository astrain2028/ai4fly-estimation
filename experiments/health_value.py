"""
Does knowing which sensor is bad actually make the estimate better?

The health arm can recover a fault's severity to within a few per cent. That
is not the same as being worth having. The point of estimating health is to
stop trusting a sensor that is lying, and whether that improves the state
estimate is a separate question that has not been asked.

There is a precedent for asking it. Outside its training envelope the Laplace
term improved the filter's innovation consistency from 3.26 to 2.83 against a
target of 3, while NEES over the same steps stayed at 160 against a target of
2. Better-calibrated uncertainty, no better estimate. Nothing rules out the
same pattern here.

TWO QUESTIONS, ONE FILE

The first is accuracy under fault: with a sensor degraded, does the
health-conditioned arm estimate speed and turn rate better than a filter that
cannot see health at all?

The second is whether the health arm inherits what makes a learned map worth
having. Test 1 showed the analytic model collapses under a one per cent
wheel-radius error while a learned one is untouched. The health arm learns its
map from data too, so it ought to inherit that -- but ought is not a
measurement, and the case for putting it at the centre of a deployed system
rests on it.

WHY THE ARMS NEED DIFFERENT FILTERS

The health arm carries ten states; everything else carries seven. The extra
three are the health levels, which no other arm has anywhere to put. So each
arm is run with the filter it was built for, and the comparison is on the
quantities they share -- speed and turn rate, which sit at indices 3 and 4 in
both.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from common import NEES_DOF, NEES_STATES, NIS_DOF, P0, Q, best_constant_R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import dynamics
import make_dataset
import make_faulted
import sensors
from faults import apply_fault
from trajectories import DT, random_run
from ukf import (UKF, expected_readings, nis as nis_of, nees as nees_of,
                 rebuild, sigma_points)

from heteroscedasticity import train_heteroscedastic


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health_arm = _load(ROOT / "models" / "health" / "measurement.py", "health_m")
health_train = _load(ROOT / "models" / "health" / "train.py", "health_t")
bhr_arm = _load(ROOT / "models" / "bhr" / "measurement.py", "bhr_m")

EVAL_RUNS = 8
SEVERITIES = [0.0, 1.0, 2.0, 3.0]
ERRORS = [0.0, 0.01, 0.03]
CAL_RUNS = 60           # runs per calibration level; smaller to stay quick


class quiet:
    def __enter__(self):
        import io
        self._real, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *_):
        sys.stdout = self._real


def run_filter(measure, meas, run, wide, R):
    """Filter one run with whichever state size the arm needs.

    `wide` selects the ten-state filter that carries health. Speed and turn
    rate are at the same indices either way, so the scores are comparable.
    """
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    accel = np.diff(run["speed"], append=run["speed"][-1]) / DT
    turn_accel = np.diff(run["turn_rate"], append=run["turn_rate"][-1]) / DT
    truth = np.column_stack([run["x"], run["y"], run["heading"],
                             run["speed"], run["turn_rate"],
                             accel, turn_accel])

    if wide:
        Q_use, P_use = health_arm.filter_settings(Q, P0)
        start = np.zeros(health_arm.N_STATES)
        truth = np.column_stack(
            [truth, np.zeros((len(truth), health_arm.N_STATES - 7))])
    else:
        Q_use, P_use = Q, P0
        start = np.zeros(7)
    start[:5] = truth[0, :5]

    if hasattr(measure, "reset"):
        measure.reset()
    means, covs, innov, S = UKF(Q_use, R, measure=measure).run(
        readings, start, P_use, DT)

    return {
        "speed": float(np.sqrt(np.mean((means[:, 3] - truth[:, 3]) ** 2))),
        "turn": float(np.sqrt(np.mean((means[:, 4] - truth[:, 4]) ** 2))),
        "nis": nis_of(innov, S),
        "nees": nees_of(means, covs, truth, states=NEES_STATES),
    }


def sweep_faults(arms, R):
    """Speed and turn error as one sensor is degraded further and further."""
    print("QUESTION 1 -- DOES IT ESTIMATE THE STATE ANY BETTER?\n")
    print("Left encoder degraded for the whole run. Speed error in m/s.\n")

    for mode in ["bias", "noise_inflation"]:
        print("  %s" % mode)
        print("  %-22s" % "severity"
              + "".join("%12.1f" % s for s in SEVERITIES))
        print("  " + "-" * (22 + 12 * len(SEVERITIES)))

        for label, measure, wide in arms:
            row = []
            for severity in SEVERITIES:
                errors = []
                for seed in range(500, 500 + EVAL_RUNS):
                    run = random_run(seed, duration=20.0)
                    meas = sensors.read_sensors(run, seed=seed, dt=DT)
                    if severity > 0:
                        meas = apply_fault(meas, "left_encoder", mode,
                                           severity, seed=seed, dt=DT)
                    errors.append(run_filter(measure, meas, run, wide,
                                             R)["speed"])
                row.append(np.mean(errors))
            print("  %-22s" % label + "".join("%12.4f" % v for v in row))
        print()

    print("  A filter that cannot see health has to average a lying sensor in")
    print("  with the honest ones. One that can should discount it. If the")
    print("  health row does not separate from the others as severity rises,")
    print("  then estimating health is a diagnostic and not an improvement.")


def sweep_calibration(R_healthy):
    """Does the health arm inherit a learned map's immunity to bad geometry?"""
    print("\n\nQUESTION 2 -- DOES IT SURVIVE A MIS-SPECIFIED VEHICLE?\n")
    print("Wheel radius differs from the value the analytic model uses.")
    print("Learned arms retrained on data from the real vehicle each time.\n")
    print("  %-26s %10s %10s %10s"
          % ("", "0%", "1%", "3%"))
    print("  " + "-" * 58)

    rows = {"analytic": [], "heteroscedastic": [], "health-conditioned": []}

    for error in ERRORS:
        healthy = make_dataset.build(n_runs=CAL_RUNS, radius_error=error)
        R_const = best_constant_R(healthy)

        with quiet():
            hetero = train_heteroscedastic(healthy)
            faulted = make_faulted.build(n_runs=CAL_RUNS, radius_error=error)
            health = train_health(faulted)

        arms = [("analytic", expected_readings, False, R_const),
                ("heteroscedastic", hetero, False, R_const),
                ("health-conditioned", health, True, R_const)]

        saved = dynamics.RADIUS_ERROR
        dynamics.RADIUS_ERROR = error
        try:
            for label, measure, wide, R_use in arms:
                errors = []
                for seed in range(600, 600 + EVAL_RUNS):
                    run = random_run(seed, duration=20.0)
                    meas = sensors.read_sensors(run, seed=seed, dt=DT)
                    errors.append(run_filter(measure, meas, run, wide,
                                             R_use)["speed"])
                rows[label].append(np.mean(errors))
        finally:
            dynamics.RADIUS_ERROR = saved

    for label, values in rows.items():
        print("  %-26s" % label + "".join("%10.4f" % v for v in values))

    print("\n  The analytic model should collapse. If the health arm tracks")
    print("  the heteroscedastic one, it inherits what makes a learned map")
    print("  worth having, and belongs at the centre of a deployed stack.")
    print("  If it does not, something about conditioning on health costs it")
    print("  that immunity, and the stack should be arranged differently.")


def mean_gain(measure, wide, mode, severity, R, seeds=range(500, 504)):
    """How much the left encoder is allowed to move the speed estimate.

    This is K[speed, left_encoder], the entry of the Kalman gain that decides
    what that one sensor does to that one state. It is the quantity the whole
    exercise is about: a filter that has worked out a sensor is lying should
    let it push the estimate around less.

    The filter is stepped by hand here rather than through UKF.run, because
    run() does not hand back the gain and the gain is the point.
    """
    totals = []
    for seed in seeds:
        run = random_run(seed, duration=20.0)
        meas = sensors.read_sensors(run, seed=seed, dt=DT)
        if severity > 0:
            meas = apply_fault(meas, "left_encoder", mode, severity,
                               seed=seed, dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])

        if wide:
            Q_use, P_use = health_arm.filter_settings(Q, P0)
            mean = np.zeros(health_arm.N_STATES)
        else:
            Q_use, P_use = Q, P0
            mean = np.zeros(7)
        mean[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                    run["speed"][0], run["turn_rate"][0]]

        f = UKF(Q_use, R, measure=measure)
        cov = P_use.copy()
        per_step = []
        for k in range(len(readings)):
            mean, cov = f.predict(mean, cov, DT)

            points, w_mean, w_cov = sigma_points(mean, cov, f.alpha, f.beta,
                                                 f.kappa)
            out = f.measure(points)
            predicted = np.asarray(out[0] if isinstance(out, tuple) else out)
            R_step = (np.average(out[1], axis=0, weights=w_mean)
                      if isinstance(out, tuple) else R)
            z_mean, S = rebuild(predicted, w_mean, w_cov, R_step)
            cross = sum(w_cov[i] * np.outer(points[i] - mean,
                                            predicted[i] - z_mean)
                        for i in range(len(points)))
            per_step.append(abs((cross @ np.linalg.inv(S))[3, 0]))

            mean, cov, _, _ = f.update(mean, cov, readings[k], None)
            if hasattr(f.measure, "constrain"):
                mean = f.measure.constrain(mean)

        # Second half only, once the filter has settled.
        totals.append(np.mean(per_step[len(per_step) // 2:]))
    return float(np.mean(totals))


def sweep_gain(arms, R):
    """Does the filter actually rely on the bad sensor less?"""
    print("\n\nQUESTION 3 -- DOES IT RELY ON THE BAD SENSOR LESS?\n")
    print("Gain from the left encoder onto the speed estimate. Lower means")
    print("that sensor is being allowed to move the answer less.\n")

    for mode in ["bias", "noise_inflation"]:
        print("  %s" % mode)
        print("  %-22s" % "severity"
              + "".join("%12.1f" % s for s in SEVERITIES))
        print("  " + "-" * (22 + 12 * len(SEVERITIES)))
        for label, measure, wide in arms:
            row = [mean_gain(measure, wide, mode, s, R) for s in SEVERITIES]
            print("  %-22s" % label + "".join("%12.4f" % v for v in row))
        print()

    print("  A constant covariance cannot respond at all, so its row is flat")
    print("  by construction -- it trusts a ruined sensor exactly as much as")
    print("  a good one. The question is whether the health row falls, and by")
    print("  enough. At severity 3 the surviving bias is larger than the")
    print("  sensor's own noise, so the gain ought to fall by something like")
    print("  a factor of three, not a few per cent.")


def train_health(frame):
    """Train the health-conditioned model in memory, on this frame."""
    torch.manual_seed(0)
    bhr = health_train.bhr
    x = torch.tensor(frame[health_train.INPUTS].values, dtype=torch.float32)
    y = torch.tensor(frame[health_train.OUTPUTS].values, dtype=torch.float32)
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)
    xs, ys = (x - x_mean) / x_std, (y - y_mean) / y_std

    model = health_train.make_model()
    bhr.train(model, xs, ys, xs, ys)
    model.eval()

    take, n_out = health_arm.TAKE, len(health_train.OUTPUTS)

    def measure(states):
        states = np.atleast_2d(states)
        picked = states[:, take].copy()
        picked[:, 5:] = np.clip(picked[:, 5:], 0.0, None)
        t = torch.tensor(picked, dtype=torch.float32)
        with torch.no_grad():
            e1, e2 = bhr.split_outputs(model((t - x_mean) / x_std), False)
            mean_s, var_s = bhr.to_mean_and_var(e1, e2)
        readings = (mean_s * y_std + y_mean).numpy().astype(float)
        variances = (var_s * y_std ** 2).numpy().astype(float)
        R = np.zeros((len(states), n_out, n_out))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    def constrain(mean):
        mean = np.array(mean, dtype=float)
        mean[health_arm.HEALTH_STATES] = np.maximum(
            mean[health_arm.HEALTH_STATES], 0.0)
        return mean

    measure.constrain = constrain
    return measure


def main():
    R = best_constant_R()
    arms = [
        ("analytic + constant R", expected_readings, False),
        ("heteroscedastic", bhr_arm.load_measurement_model(), False),
        ("health-conditioned", health_arm.load_measurement_model(), True),
    ]
    sweep_faults(arms, R)
    sweep_gain(arms, R)
    sweep_calibration(R)


if __name__ == "__main__":
    main()

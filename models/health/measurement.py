"""
Supplies the health-conditioned model to a filter that carries health.

The state is ten entries now:

    [x, y, heading, speed, turn_rate, accel, turn_accel,
     m_left, m_right, m_gyro]

and this model reads eight of them -- the five the sensors depend on, and the
three health levels. Acceleration is skipped because no sensor reports it.

WHAT CHANGES IN THE FILTER

Nothing, mechanically. The update is the same. What changes is that the
sample points now spread over health as well as over position and speed, so
the predicted readings differ between points that differ only in how healthy
they think a sensor is. That difference is what fills the health rows of the
state-measurement covariance, and those rows are what let the update move
health:

    m_hat  <-  m_hat  +  P_mz S^-1 nu

Every other arm in this project has P_mz identically zero, because their
predicted readings do not depend on health -- they have no health to depend
on. That is the entire difference, and it is why none of them responds to a
fault.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np
import torch


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_train = _load(Path(__file__).parent / "train.py", "health_train")
bhr = _train.bhr
make_model = _train.make_model
INPUTS, OUTPUTS = _train.INPUTS, _train.OUTPUTS

# Which entries of the filter's state this model wants, in order:
# the five vehicle states, then the three health levels.
TAKE = [0, 1, 2, 3, 4, 7, 8, 9]

N_STATES = 10
HEALTH_STATES = [7, 8, 9]


def load_measurement_model(path=None):
    """Load the trained model and return a function the filter can call.

    Takes states of shape (n, 10) and returns readings of shape (n, 3) with a
    covariance per state. Health below zero is clipped on the way in: the
    filter's sample points can wander negative, and a negative degradation is
    not a thing the model was ever shown.
    """
    if path is None:
        path = Path(__file__).parent / "health_model.pt"
    saved = torch.load(path, weights_only=False)

    model = make_model()
    model.load_state_dict(saved["weights"])
    model.eval()

    x_mean, x_std = saved["x_mean"], saved["x_std"]
    y_mean, y_std = saved["y_mean"], saved["y_std"]

    def measure(states):
        states = np.atleast_2d(states)
        picked = states[:, TAKE].copy()
        picked[:, 5:] = np.clip(picked[:, 5:], 0.0, None)

        x = torch.tensor(picked, dtype=torch.float32)
        with torch.no_grad():
            eta1, eta2 = bhr.split_outputs(model((x - x_mean) / x_std), False)
            mean_s, var_s = bhr.to_mean_and_var(eta1, eta2)

        readings = (mean_s * y_std + y_mean).numpy().astype(float)
        variances = (var_s * y_std ** 2).numpy().astype(float)

        R = np.zeros((len(states), len(OUTPUTS), len(OUTPUTS)))
        for k in range(len(states)):
            R[k] = np.diag(variances[k])
        return readings, R

    return measure


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    import numpy as np
    from common import Q, P0, best_constant_R
    from faults import apply_fault
    from sensors import read_sensors
    from trajectories import DT, random_run
    from ukf import UKF, nis as nis_of

    measure = load_measurement_model()

    # The filter's settings, widened for health. Health gets a small process
    # noise so the estimate can move at all, and a generous starting spread
    # because nothing is known about it before any measurement arrives.
    Q10 = np.zeros((N_STATES, N_STATES))
    Q10[:7, :7] = Q
    for i in HEALTH_STATES:
        Q10[i, i] = 1e-4
    P10 = np.zeros((N_STATES, N_STATES))
    P10[:7, :7] = P0
    for i in HEALTH_STATES:
        P10[i, i] = 1.0

    R_const = best_constant_R()

    print("CAN THE FILTER ESTIMATE HEALTH?\n")
    print("Left encoder faulted for the whole run, filter started at m = 0.")
    print("A bias changes what the model predicts; noise inflation does not.\n")
    print("%-18s %10s %14s %14s %10s"
          % ("mode", "true m", "estimated m", "other channels", "NIS"))
    print("-" * 70)

    for mode in ["none", "bias", "noise_inflation"]:
        for severity in ([0.0] if mode == "none" else [1.0, 3.0]):
            estimates, others, nis_all = [], [], []
            for seed in range(400, 406):
                run = random_run(seed, duration=20.0)
                meas = read_sensors(run, seed=seed, dt=DT)
                if mode != "none":
                    meas = apply_fault(meas, "left_encoder", mode, severity,
                                       seed=seed, dt=DT)
                readings = np.column_stack([meas["left_encoder"],
                                            meas["right_encoder"],
                                            meas["gyro"]])
                start = np.zeros(N_STATES)
                start[:3] = [run["x"][0], run["y"][0], run["heading"][0]]
                start[3], start[4] = run["speed"][0], run["turn_rate"][0]

                means, covs, innov, S = UKF(Q10, R_const, measure=measure).run(
                    readings, start, P10, DT)

                # The second half only: the filter needs time to settle.
                half = len(means) // 2
                estimates.append(means[half:, 7].mean())
                others.append(means[half:, 8:10].mean())
                nis_all.append(nis_of(innov, S).mean())

            print("%-18s %10.2f %14.3f %14.3f %10.2f"
                  % (mode, severity, np.mean(estimates), np.mean(others),
                     np.mean(nis_all)))

    print("-" * 70)
    print("\nThe 'other channels' column is the control. Health estimated on")
    print("the two sensors that were never touched should stay near zero; if")
    print("it rises with the fault, the filter is detecting that something is")
    print("wrong without being able to say which sensor, which is a weaker")
    print("claim than the one this formulation is meant to support.")

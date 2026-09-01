"""
Faults the learned model has never seen.

THE TEST THAT MATTERS MOST AND FLATTERS LEAST

The health-conditioned model is trained on two kinds of degradation, bias and
noise inflation. faults.py defines six. So the obvious question is what
happens on the other four, and the honest expectation is that it does badly:
a model conditioned on health has learned what the degradations in its
training set do to a reading, and a degradation outside that set is an input
it has no basis for.

Ovadia et al. give the general form of this. Predictive uncertainty degrades
under distributional shift, and calibration that holds in-distribution fails
under even mild shift. There is no reason a learned fault model should be
exempt.

The classical alternative has the opposite character. Covariance matching
does not know what a fault is. It watches innovations, and when they are
consistently larger than predicted it raises the covariance -- regardless of
why, and regardless of whether anyone has seen that failure before. It should
therefore be roughly as good on an unfamiliar fault as on a familiar one,
which is a property no learned model has.

WHAT IS EXPECTED, SO THAT THE RESULT CAN DISAGREE

    drift          a bias that grows during the run. Bias-like, so the model
                   may partly transfer, though it was trained on constants.
    scale_error    a multiplicative error, so a bias that grows with speed.
                   Also bias-like.
    stuck          the reading freezes. Nothing like either training mode.
    dropout        readings are lost and the last is held. Nothing like
                   either training mode.

A gradient rather than a cliff would be the interesting outcome: the model
transferring to what resembles its training and failing on what does not.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from common import NIS_DOF, P0, Q, best_constant_R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robot"))

import sensors
from faults import apply_fault
from trajectories import DT, random_run
from ukf import UKF, expected_readings, nis as nis_of


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


health = _load(ROOT / "models" / "health" / "measurement.py", "health_h")
combined = _load(ROOT / "models" / "combined" / "measurement.py", "combined_h")
adaptive = _load(ROOT / "models" / "adaptive" / "measurement.py", "adaptive_h")

TRAINED_ON = ["bias", "noise_inflation"]
HELD_OUT = ["drift", "scale_error", "stuck", "dropout"]

# stuck and dropout take severity as a fraction of the run, so 3.0 is
# meaningless for them; everything is evaluated at a severity each mode can
# actually express.
SEVERITY = {"bias": 2.0, "noise_inflation": 2.0, "drift": 2.0,
            "scale_error": 2.0, "stuck": 0.5, "dropout": 0.3}

EVAL_RUNS = 8


def score(measure, wide, mode, severity, R, n_runs=EVAL_RUNS):
    speed, nis_all = [], []
    for seed in range(900, 900 + n_runs):
        run = random_run(seed, duration=20.0)
        meas = sensors.read_sensors(run, seed=seed, dt=DT)
        if severity > 0:
            meas = apply_fault(meas, "left_encoder", mode, severity,
                               seed=seed, dt=DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])

        if wide:
            Q_use, P_use = health.filter_settings(Q, P0)
            start = np.zeros(health.N_STATES)
        else:
            Q_use, P_use = Q, P0
            start = np.zeros(7)
        start[:5] = [run["x"][0], run["y"][0], run["heading"][0],
                     run["speed"][0], run["turn_rate"][0]]

        if hasattr(measure, "reset"):
            measure.reset()
        means, _, innov, S = UKF(Q_use, R, measure=measure).run(
            readings, start, P_use, DT)

        speed.append(np.sqrt(np.mean((means[:, 3] - run["speed"]) ** 2)))
        nis_all.append(nis_of(innov, S).mean())

    return float(np.mean(speed)), float(np.mean(nis_all))


def main():
    R = best_constant_R()
    arms = [
        ("analytic + constant R", expected_readings, False),
        ("adaptive R", adaptive.AdaptiveR(R), False),
        ("health-conditioned", health.load_measurement_model(), True),
        ("combined", combined.load_measurement_model(), True),
    ]

    print("HELD-OUT FAULT TYPES\n")
    print("The learned model saw bias and noise inflation in training and")
    print("none of the other four. Speed error in m/s; healthy is the")
    print("no-fault reference for the same arm.\n")

    modes = ["none"] + TRAINED_ON + HELD_OUT
    header = "  %-24s" % "" + "".join("%11s" % m[:10] for m in modes)
    print(header)
    print("  " + "-" * (24 + 11 * len(modes)))

    table = {}
    for label, measure, wide in arms:
        row = []
        for mode in modes:
            severity = 0.0 if mode == "none" else SEVERITY[mode]
            error, _ = score(measure, wide, mode, severity, R)
            row.append(error)
        table[label] = row
        print("  %-24s" % label + "".join("%11.4f" % v for v in row))

    print("  " + "-" * (24 + 11 * len(modes)))

    print("\n\nCOST OF THE FAULT, RELATIVE TO EACH ARM'S OWN HEALTHY ERROR\n")
    print("  %-24s" % "" + "".join("%11s" % m[:10] for m in modes[1:]))
    print("  " + "-" * (24 + 11 * (len(modes) - 1)))
    for label, row in table.items():
        healthy = row[0]
        print("  %-24s" % label
              + "".join("%10.1fx" % (v / healthy) for v in row[1:]))

    print("  " + "-" * (24 + 11 * (len(modes) - 1)))
    print("\nThe left two columns are what the model was trained on and the")
    print("right four are not. A learned arm that holds up across all six is")
    print("transferring; one that does well on the left and badly on the")
    print("right has learned the faults rather than the idea of a fault.")
    print("\nThe adaptive row should be roughly flat, because it does not know")
    print("what any of these are -- it only knows its innovations grew.")


if __name__ == "__main__":
    main()

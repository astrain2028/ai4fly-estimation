"""
Why covariance matching loses on the quadcopter.

The anomaly
-----------

On the ground robot, Mehra's estimator beats the analytic model on bias
faults. On the quadcopter it is worse than the analytic model on bias faults
on both devices tested -- 2.850 against 2.229 on the accelerometer, 2.168
against 1.992 on the magnetometer -- and the README records that rather than
explaining it.

The hypothesis
--------------

Mehra's estimator is a subtraction. The innovation covariance the filter
predicts is

    S = scatter + R

where scatter is the spread of the predicted measurements over the sigma
points. The estimator observes the innovations, forms their empirical
covariance, subtracts the scatter the filter claimed, and takes what is left
as R:

    R_est = E[nu nu'] - scatter

That is exact when the scatter is right. Jiang, Shi and Moura (README [27])
prove that nonlinear Kalman filters systematically underestimate the
posterior covariance when the measurement map is nonlinear, and the scatter
is a projection of that covariance through the map. If the scatter is
underestimated, the subtraction leaves too much for R, the estimator inflates
it, and the filter trusts its sensors less than it should. That would cost
accuracy on every fault, and it is the only mechanism proposed here that
would not arise on the ground robot, whose map is linear.

The measurement
---------------

Run the analytic model with the best constant R on healthy runs of both
vehicles, and compare per channel:

    predicted    mean over the run of  diag(S - R)      what the UKF claims
    empirical    var(nu) - diag(R)                      what actually happened

The ratio empirical / predicted is what Mehra's residual is biased by. Near 1
on both vehicles and the hypothesis is wrong. Near 1 on the robot and well
above 1 on the quadcopter, and it explains the anomaly -- and predicts that
any innovation-based adaptive method inherits the same bias on any nonlinear
map, which is worth knowing beyond this repository.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loader import load_module

SEEDS = range(2000, 2008)
RESULTS = ROOT / "results"
RECORDS = []


def _borrow(folder, names):
    """Load a simulator's modules under unique names, off the shared path."""
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / folder))
    for n in names:
        sys.modules.pop(n, None)
    try:
        mods = {n: load_module(ROOT / folder / ("%s.py" % n),
                               "%s_%s_scatter" % (folder, n)) for n in names}
    finally:
        sys.path[:] = saved
        for n in names:
            sys.modules.pop(n, None)
    return mods


def innovations(ukf_module, move, measure, Q, R, P0, readings, start, dt):
    """Innovations and the scatter the filter claimed, step by step."""
    f = ukf_module.UKF(Q, R, move=move, measure=measure)
    mean, cov = start.copy(), P0.copy()
    nus, scatters = [], []
    for k in range(len(readings)):
        mean, cov = f.predict(mean, cov, dt)
        mean, cov, nu, S = f.update(mean, cov, readings[k])
        nus.append(nu)
        scatters.append(np.diag(S) - np.diag(R))
    return np.array(nus), np.array(scatters)


def robot():
    m = _borrow("robot", ["dynamics", "trajectories", "sensors", "ukf"])
    sys.path.insert(0, str(ROOT / "experiments"))
    common = load_module(ROOT / "experiments" / "common.py", "common_scatter")
    R = common.best_constant_R()
    rows = []
    for seed in SEEDS:
        run = m["trajectories"].random_run(seed, duration=20.0)
        meas = m["sensors"].read_sensors(run, seed=seed, dt=m["trajectories"].DT)
        readings = np.column_stack([meas["left_encoder"],
                                    meas["right_encoder"], meas["gyro"]])
        accel = np.diff(run["speed"], append=run["speed"][-1]) / m["trajectories"].DT
        turn = np.diff(run["turn_rate"], append=run["turn_rate"][-1]) / m["trajectories"].DT
        start = np.array([run["x"][0], run["y"][0], run["heading"][0],
                          run["speed"][0], run["turn_rate"][0],
                          accel[0], turn[0]])
        nus, sc = innovations(m["ukf"], m["ukf"].move_state,
                              m["ukf"].expected_readings, common.Q, R,
                              common.P0, readings, start, m["trajectories"].DT)
        rows.append((nus, sc))
    return rows, np.diag(R), ["left_enc", "right_enc", "gyro"]


def quad():
    m = _borrow("quad_sim", ["dynamics", "trajectories", "sensors"])
    ukf = _borrow("robot", ["dynamics", "ukf"])["ukf"]
    dyn, traj, sens = m["dynamics"], m["trajectories"], m["sensors"]

    def move(state, dt):
        return dyn.step(np.asarray(state, dtype=float)[:6], dt)

    def measure(states):
        states = np.atleast_2d(states)
        force = dyn.specific_force(states[:, 0], states[:, 1], states[:, 2])
        field = dyn.magnetic_field(states[:, 0], states[:, 1], states[:, 2])
        return np.hstack([force, states[:, 3:6], field])

    Q = np.diag([1e-8, 1e-8, 1e-8, 2e-3, 2e-3, 2e-3])
    P0 = np.diag([0.02, 0.02, 0.30, 0.05, 0.05, 0.05])
    total = np.zeros(9)
    for seed in range(20):
        total += (sens.noise_levels(traj.random_run(seed)) ** 2).mean(axis=0)
    R = np.diag(total / 20)

    rows = []
    for seed in SEEDS:
        run = traj.random_run(seed)
        readings = sens.stack(sens.read_sensors(run, seed=seed))
        start = traj.truth_matrix(run)[0].copy()
        nus, sc = innovations(ukf, move, measure, Q, R, P0, readings, start,
                              traj.DT)
        rows.append((nus, sc))
    return rows, np.diag(R), list(sens.CHANNELS)


def report(label, rows, R_diag, names):
    nus = np.vstack([r[0] for r in rows])
    scat = np.vstack([r[1] for r in rows])
    predicted = scat.mean(axis=0)
    empirical = nus.var(axis=0) - R_diag
    for i, name in enumerate(names):
        RECORDS.append({"vehicle": label.split(" --")[0].strip().lower(),
                        "channel": name, "claimed": predicted[i],
                        "empirical": empirical[i],
                        "ratio": empirical[i] / predicted[i]})

    print("\n%s\n" % label)
    print("  %-10s %12s %12s %10s" % ("channel", "predicted", "empirical",
                                       "ratio"))
    print("  " + "-" * 48)
    ratios = []
    for i, name in enumerate(names):
        ratio = empirical[i] / predicted[i] if predicted[i] > 0 else np.nan
        ratios.append(ratio)
        print("  %-10s %12.3e %12.3e %9.2fx" % (name, predicted[i],
                                                empirical[i], ratio))
    print("  " + "-" * 48)
    return np.array(ratios)


def main():
    print("DOES THE UKF UNDERESTIMATE ITS OWN MEASUREMENT SCATTER?")
    print("\nHealthy runs, analytic model, best constant R, %d seeds. The"
          % len(list(SEEDS)))
    print("scatter is what Mehra's estimator subtracts; if the filter claims")
    print("less than there is, the residual it leaves for R is too large.")

    r_ratio = report("GROUND ROBOT -- linear map", *robot())
    q_ratio = report("QUADCOPTER -- nonlinear map", *quad())

    print("\n\nVERDICT\n")
    print("  robot      median ratio %.2fx" % np.nanmedian(r_ratio))
    print("  quadcopter median ratio %.2fx" % np.nanmedian(q_ratio))
    if np.nanmedian(q_ratio) > 1.5 and np.nanmedian(r_ratio) < 1.5:
        print("\n  The quadcopter's filter claims markedly less measurement")
        print("  scatter than the innovations actually show, and the robot's")
        print("  does not. Mehra's subtraction is therefore biased on the")
        print("  quad: it attributes the missing scatter to sensor noise and")
        print("  inflates R, which is the anomaly. Any innovation-based")
        print("  adaptive method inherits this on a nonlinear map.")
    elif np.nanmedian(q_ratio) < 1.5:
        print("\n  The scatter is not underestimated on the quadcopter, so")
        print("  this is not why covariance matching loses there. The anomaly")
        print("  stays open and the hypothesis in the README should go.")
    else:
        print("\n  Both vehicles show the effect, so it does not explain why")
        print("  only the quadcopter's adaptive arm loses to the baseline.")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(RECORDS).to_csv(RESULTS / "scatter.csv", index=False,
                                 float_format="%.6g")
    print("\nWrote %d rows to results/scatter.csv" % len(RECORDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())

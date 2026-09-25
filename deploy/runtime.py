"""
The estimator as it would run on the vehicle.

What this is
------------

A loop that takes timestamped sensor readings from a source, runs one filter
step per reading, and writes a log. The filter is a plain function call in a
plain loop -- not a callback, not a subscriber, not a coroutine -- because the
cost claim this project makes is per step and stops being measurable once
middleware scheduling is mixed into it.

The source is anything that yields ``(timestamp, reading)``. Today that is a
simulated run replayed with the gaps and jitter a real link has; on the
vehicle it is a MAVLink stream. The estimator does not know which, which is
the point: everything downstream of the source is what gets tested here and
what ships unchanged.

Only numpy is imported. The measurement model comes from the weights that
deploy/export.py wrote, and the filter is robot/ukf.py, which has no
dependency beyond numpy either.

Two things a simulator never has to handle
------------------------------------------

**The interval is measured, not assumed.** Every experiment in this
repository steps at exactly 0.02 s. A real link delivers readings late, early,
and occasionally in bursts, so ``dt`` comes from the timestamps. A step whose
interval is far from nominal is flagged in the log rather than silently
integrated, since a filter that integrated a 0.5 s gap as 0.02 s would be
confidently wrong about where the vehicle is.

**A missing reading is not a reading.** When the source yields ``None`` the
filter predicts and does not update. Reusing the previous reading instead --
the easy mistake -- is indistinguishable from a ``stuck`` fault to every
mechanism in this project, and would be scored as one. The log records which
steps were predict-only so a downstream consumer can tell a quiet link from a
frozen sensor.

What is logged
--------------

One row per step: timestamp, measured interval, whether an update happened,
the full state estimate, the diagonal of its covariance, the innovation, and
NIS. That is what NIS-based self-assessment needs, and NIS is the only
self-assessment available on hardware, since NEES needs a true state nobody
has in flight.

Vehicle
-------

The ground robot, because those are the weights that exist and a bench test
comes before a flight. The vehicle enters through four arguments -- motion
model, measurement model, state layout, nominal rate -- so quad_sim slots in
by supplying its own.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from loader import load_module

# A step whose interval differs from nominal by more than this factor is
# flagged. The filter still integrates it -- the alternative is dropping data
# -- but the flag lets a consumer discount the estimate around a gap.
INTERVAL_TOLERANCE = 2.0


def _ukf():
    """robot/ukf.py, loaded with robot/ importable for its own dynamics import."""
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "robot"))
    try:
        return load_module(ROOT / "robot" / "ukf.py", "runtime_ukf")
    finally:
        sys.path[:] = saved


ukf = _ukf()


class Estimator:
    """One vehicle's filter, stepped by timestamp.

    ``move`` and ``measure`` are the motion and measurement models the UKF
    takes as arguments. ``health_states`` are the state indices held
    non-negative after each update; empty for an arm that carries no health.
    """

    def __init__(self, move, measure, Q, R, start_mean, start_cov,
                 dt_nominal, health_states=()):
        self.filter = ukf.UKF(Q, R, move=move, measure=measure)
        self.measure = measure
        self.mean = np.asarray(start_mean, dtype=float).copy()
        self.cov = np.asarray(start_cov, dtype=float).copy()
        self.start_cov = self.cov.copy()
        self.dt_nominal = float(dt_nominal)
        self.health_states = list(health_states)
        self.last_time = None
        self.repairs = 0

    def _repaired(self, cov):
        """A symmetric positive-definite covariance close to ``cov``.

        The unscented transform takes a Cholesky factor every step, and a
        covariance that has drifted asymmetric or lost definiteness to
        rounding makes it raise. In a simulation that ends the run; on the
        vehicle it must not. Eigenvalues are floored, and if that is not
        enough the covariance is reset to its starting value, which loses
        the filter's confidence but not its state.
        """
        cov = 0.5 * (cov + cov.T)
        values, vectors = np.linalg.eigh(cov)
        if values.min() > 0 and values.min() > 1e-12 * values.max():
            return cov
        floor = max(1e-12 * values.max(), 1e-12)
        return (vectors * np.maximum(values, floor)) @ vectors.T

    def _advance(self, dt, reading):
        """One predict and, with a reading, one update. May raise."""
        mean, cov = self.filter.predict(self.mean, self.cov, dt)
        if reading is None:
            return mean, cov, None, None
        mean, cov, innovation, S = self.filter.update(
            mean, cov, np.asarray(reading, dtype=float))
        return mean, cov, innovation, S

    def step(self, timestamp, reading):
        """Advance to ``timestamp`` and, if ``reading`` is not None, update.

        Returns a dict describing what happened, which is one log row.
        """
        if self.last_time is None:
            dt = self.dt_nominal
        else:
            dt = float(timestamp - self.last_time)
        self.last_time = timestamp

        irregular = (dt <= 0.0
                     or dt > INTERVAL_TOLERANCE * self.dt_nominal
                     or dt < self.dt_nominal / INTERVAL_TOLERANCE)
        if dt <= 0.0:
            dt = self.dt_nominal      # a repeated or reversed timestamp

        started = time.perf_counter()
        self.cov = 0.5 * (self.cov + self.cov.T)
        repaired = 0
        try:
            mean, cov, innovation, S = self._advance(dt, reading)
        except np.linalg.LinAlgError:
            repaired = 1
            self.cov = self._repaired(self.cov)
            try:
                mean, cov, innovation, S = self._advance(dt, reading)
            except np.linalg.LinAlgError:
                repaired = 2
                self.cov = self.start_cov.copy()
                mean, cov, innovation, S = self._advance(dt, reading)
        self.mean, self.cov = mean, cov
        self.repairs += int(repaired > 0)

        nis = None
        if reading is not None:
            if hasattr(self.measure, "observe"):
                self.measure.observe(innovation, S)
            if self.health_states:
                self.mean[self.health_states] = np.maximum(
                    self.mean[self.health_states], 0.0)
            nis = float(innovation @ np.linalg.solve(S, innovation))
        elapsed = time.perf_counter() - started

        return {
            "time": float(timestamp),
            "dt": dt,
            "updated": reading is not None,
            "irregular": bool(irregular),
            "repaired": repaired,
            "ms": 1000.0 * elapsed,
            "mean": self.mean.copy(),
            "var": np.diag(self.cov).copy(),
            "innovation": innovation,
            "nis": nis,
        }


def run(source, estimator, log_path, state_names, channel_names):
    """Drive ``estimator`` from ``source`` and write one csv row per step."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = log_header(state_names, channel_names)

    rows = 0
    with open(log_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for timestamp, reading in source:
            record = estimator.step(timestamp, reading)
            writer.writerow(log_row(record, channel_names))
            rows += 1
    return rows


def log_header(state_names, channel_names):
    """Column names for one log row, shared with deploy/pi_main.py."""
    return (["time", "dt", "updated", "irregular", "repaired", "ms"]
            + list(state_names)
            + ["var_%s" % s for s in state_names]
            + ["innov_%s" % c for c in channel_names]
            + ["nis"])


def log_row(record, channel_names):
    """One estimator record as the strings that go in its log row."""
    innov = (record["innovation"] if record["innovation"] is not None
             else [""] * len(channel_names))
    return ([record["time"], "%.6f" % record["dt"], int(record["updated"]),
             int(record["irregular"]), record["repaired"],
             "%.4f" % record["ms"]]
            + ["%.6g" % v for v in record["mean"]]
            + ["%.6g" % v for v in record["var"]]
            + ["%.6g" % v if v != "" else "" for v in innov]
            + ["%.4f" % record["nis"] if record["nis"] is not None else ""])


def replay(readings, dt, seed=0, drop_fraction=0.0, jitter_fraction=0.0):
    """Turn an array of readings into a source with the defects of a real link.

    ``drop_fraction`` of readings arrive as None. Timestamps are jittered by a
    zero-mean fraction of ``dt``, so the measured interval varies around the
    nominal one the way it does over a serial link.
    """
    rng = np.random.default_rng(seed)
    n = len(readings)
    times = np.arange(n) * dt
    if jitter_fraction > 0:
        times = times + rng.normal(0.0, jitter_fraction * dt, size=n)
        times = np.maximum.accumulate(times)        # keep them monotone
    dropped = rng.random(n) < drop_fraction
    for k in range(n):
        yield times[k], (None if dropped[k] else readings[k])


def ground_robot(weights=None):
    """Everything the estimator needs for the ground robot, numpy only.

    Returns (move, measure, Q, R, start_mean, start_cov, dt, health_states,
    state_names, channel_names). The learned measurement model is the health
    arm's weights, exported by deploy/export.py; the analytic fallback is used
    if they are absent, so the loop can be exercised before anything is
    trained.
    """
    export = load_module(HERE / "export.py", "runtime_export")
    dynamics = load_module(ROOT / "robot" / "dynamics.py", "runtime_dynamics")

    n_vehicle, n_states = 7, 13
    health_states = list(range(7, 13))
    weights = HERE / "weights" / "health.npz" if weights is None else weights

    def move(state, dt):
        return ukf.move_state(state, dt)

    if Path(weights).exists():
        take = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
        measure = export.load(weights, take=take, n_vehicle=5)
        layered = load_module(ROOT / "models" / "layered" / "measurement.py",
                              "runtime_layered")
        measure = layered.Layered(measure, health_states=health_states,
                                  n_channels=3)
    else:
        measure = ukf.expected_readings
        n_states, health_states = n_vehicle, []

    Q = np.zeros((n_states, n_states))
    Q[:7, :7] = np.diag([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-3, 1e-1])
    P0 = np.zeros((n_states, n_states))
    P0[:7, :7] = np.diag([0.01, 0.01, 0.01, 0.10, 0.10, 0.04, 0.04])
    for i in health_states:
        Q[i, i], P0[i, i] = 1e-6, 0.1

    R = np.diag([0.1805 ** 2, 0.1708 ** 2, 0.00799 ** 2]) * 0.85

    names = ["x", "y", "heading", "speed", "turn_rate", "accel", "turn_accel",
             "bias_left", "bias_right", "bias_gyro",
             "noise_left", "noise_right", "noise_gyro"][:n_states]
    channels = ["left_encoder", "right_encoder", "gyro"]
    return (move, measure, Q, R, np.zeros(n_states), P0, 0.02,
            health_states, names, channels)


def quadcopter(weights=None):
    """Everything the estimator needs for the quadcopter, numpy only.

    Same return shape as ground_robot. This is the vehicle
    deploy/mavlink_source.py feeds: nine channels from SCALED_IMU in the order
    quad_sim trained on. The motion and analytic measurement models are built
    from quad_sim/dynamics.py directly rather than through
    quad_sim/measurement.py, because that module imports the training code and
    with it torch, and the vehicle-side process is meant to carry neither.
    """
    export = load_module(HERE / "export.py", "runtime_export_quad")
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "quad_sim"))
    try:
        dynamics = load_module(ROOT / "quad_sim" / "dynamics.py",
                               "runtime_quad_dynamics")
        sensors = load_module(ROOT / "quad_sim" / "sensors.py",
                              "runtime_quad_sensors")
        trajectories = load_module(ROOT / "quad_sim" / "trajectories.py",
                                   "runtime_quad_trajectories")
    finally:
        sys.path[:] = saved

    n_vehicle, n_states = 6, 12
    health_states = list(range(6, 12))
    weights = HERE / "weights" / "quad.npz" if weights is None else weights

    def move(state, dt):
        moved = dynamics.step(np.asarray(state, dtype=float)[:n_vehicle], dt)
        if len(state) > n_vehicle:
            return np.concatenate([moved, state[n_vehicle:]])
        return moved

    def analytic(states):
        states = np.atleast_2d(states)
        force = dynamics.specific_force(states[:, 0], states[:, 1],
                                        states[:, 2])
        field = dynamics.magnetic_field(states[:, 0], states[:, 1],
                                        states[:, 2])
        return np.hstack([force, states[:, 3:6], field])

    if Path(weights).exists():
        measure = export.load(weights, take=list(range(n_states)),
                              n_vehicle=n_vehicle)
        layered = load_module(ROOT / "models" / "layered" / "measurement.py",
                              "runtime_layered_quad")
        measure = layered.Layered(measure, health_states=health_states,
                                  n_channels=9)
    else:
        measure = analytic
        n_states, health_states = n_vehicle, []

    # The same Q, P0 and health settings quad_sim/measurement.py uses,
    # restated so this file does not import it.
    Q = np.zeros((n_states, n_states))
    Q[:6, :6] = np.diag([1e-8, 1e-8, 1e-8, 2e-3, 2e-3, 2e-3])
    P0 = np.zeros((n_states, n_states))
    P0[:6, :6] = np.diag([0.02, 0.02, 0.30, 0.05, 0.05, 0.05])
    for i in health_states:
        Q[i, i], P0[i, i] = 1e-6, 0.1

    # The best constant R: average measurement variance over healthy flights,
    # as quad_sim/measurement.py computes it.
    total = np.zeros(9)
    for seed in range(20):
        total += (sensors.noise_levels(trajectories.random_run(seed)) ** 2
                  ).mean(axis=0)
    R = np.diag(total / 20)

    names = ["roll", "pitch", "yaw", "p", "q", "r",
             "bias_accel", "bias_gyro", "bias_mag",
             "noise_accel", "noise_gyro", "noise_mag"][:n_states]
    return (move, measure, Q, R, np.zeros(n_states), P0,
            trajectories.DT, health_states, names, list(sensors.CHANNELS))


if __name__ == "__main__":
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "robot"))
    from trajectories import DT, random_run
    from sensors import read_sensors
    sys.path[:] = saved

    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = ground_robot()
    using_learned = len(health_states) > 0
    print("GROUND ROBOT, %s MEASUREMENT MODEL, NUMPY ONLY\n"
          % ("LEARNED" if using_learned else "ANALYTIC"))

    run_data = random_run(0, duration=20.0)
    meas = read_sensors(run_data, seed=0, dt=DT)
    readings = np.column_stack([meas["left_encoder"], meas["right_encoder"],
                                meas["gyro"]])
    start[:5] = [run_data["x"][0], run_data["y"][0], run_data["heading"][0],
                 run_data["speed"][0], run_data["turn_rate"][0]]

    log = HERE / "logs" / "replay_check.csv"
    print("Replaying one simulated run with 5% dropped readings and 20%")
    print("timing jitter, the defects a serial link has and a simulator")
    print("does not.\n")

    source = replay(readings, dt, seed=1, drop_fraction=0.05,
                    jitter_fraction=0.2)
    estimator = Estimator(move, measure, Q, R, start, P0, dt,
                          health_states=health_states)
    rows = run(source, estimator, log, names, channels)

    import pandas as pd
    frame = pd.read_csv(log)
    updated = frame["updated"].astype(bool)
    speed_err = np.sqrt(np.mean((frame["speed"] - run_data["speed"]) ** 2))
    print("  steps logged            %d" % rows)
    print("  predict-only steps      %d  (%.1f%%)"
          % ((~updated).sum(), 100 * (~updated).mean()))
    print("  flagged irregular       %d" % frame["irregular"].sum())
    print("  speed error             %.4f m/s" % speed_err)
    print("  mean NIS on updates     %.2f  (want 3)"
          % frame.loc[updated, "nis"].mean())
    print("  median ms per step      %.3f" % frame["ms"].median())
    print("\n  Wrote %s" % log.relative_to(ROOT))

    print("\n  Dropped readings show as predict-only rows with no innovation,")
    print("  which is what distinguishes a quiet link from a stuck sensor in")
    print("  the log. A filter that reused the last reading instead would")
    print("  look identical to a frozen channel to every detector here.")

    # The quadcopter: the vehicle the MAVLink adapter actually feeds. Loaded
    # under a borrowed path because quad_sim's module names collide with the
    # robot's, and the robot's are already in sys.modules from above.
    for name in ("trajectories", "sensors", "dynamics"):
        sys.modules.pop(name, None)
    saved = list(sys.path)
    sys.path.insert(0, str(ROOT / "quad_sim"))
    import trajectories as qtraj
    import sensors as qsens
    sys.path[:] = saved

    (move, measure, Q, R, start, P0, dt,
     health_states, names, channels) = quadcopter()
    print("\n\nQUADCOPTER, %s MEASUREMENT MODEL, NUMPY ONLY\n"
          % ("LEARNED" if health_states else "ANALYTIC"))

    flight = qtraj.random_run(0)
    meas = qsens.read_sensors(flight, seed=0)
    readings = qsens.stack(meas)
    start[:6] = qtraj.truth_matrix(flight)[0]

    log = HERE / "logs" / "replay_check_quad.csv"
    source = replay(readings, dt, seed=1, drop_fraction=0.05,
                    jitter_fraction=0.2)
    estimator = Estimator(move, measure, Q, R, start, P0, dt,
                          health_states=health_states)
    rows = run(source, estimator, log, names, channels)

    frame = pd.read_csv(log)
    updated = frame["updated"].astype(bool)
    truth = qtraj.truth_matrix(flight)
    att = np.degrees(frame[["roll", "pitch"]].values - truth[:, :2])
    print("  steps logged            %d" % rows)
    print("  predict-only steps      %d" % (~updated).sum())
    print("  attitude error          %.3f deg (roll and pitch)"
          % np.sqrt(np.mean(att ** 2)))
    print("  mean NIS on updates     %.2f  (want 9)"
          % frame.loc[updated, "nis"].mean())
    print("  median ms per step      %.3f" % frame["ms"].median())
    print("\n  Wrote %s" % log.relative_to(ROOT))
    print("\n  This is the estimator deploy/mavlink_source.py feeds: nine")
    print("  channels from SCALED_IMU, in this order, into this loop.")

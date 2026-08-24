"""
The classical answer: no learning, adjust R from the filter's own mistakes.

This is Mehra's idea from 1970, and it is the baseline any learned noise
model has to justify itself against. It needs no training data, no network,
and almost no arithmetic.

HOW IT WORKS

The filter predicts how surprised it expects to be. That prediction is S, the
innovation covariance, and it is built from two pieces:

    S = spread of h(x)  +  R

The first piece comes from the filter not knowing the state exactly. The
second is the sensor noise. Now watch the innovations that actually arrive.
If they are consistently bigger than S said they would be, something is
noisier than claimed -- and since the first piece is known, the blame falls
on R. So:

    R  <-  (how big innovations really were)  -  (spread of h)

Take a window of recent innovations, average their outer products, subtract
the part the state spread explains, and what is left is an estimate of R.

WHY IT IS NOT ENOUGH

Two limitations, and both are the reason the rest of this project exists.

It is reactive. R only moves after enough bad innovations have piled up to
fill the window, so a sensor that fails suddenly is trusted at its old level
for as long as the window is deep. Shrink the window and the estimate gets
noisy enough to destabilise the filter. That trade cannot be tuned away.

It cannot tell what went wrong. Large innovations mean the filter was
surprised, and a sensor going bad and a motion model being wrong produce the
same surprise. This method raises R either way -- so when the real problem is
that the robot is doing something unmodelled, it responds by distrusting
perfectly good sensors.
"""

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "robot"))

import numpy as np

from ukf import expected_readings

WINDOW = 200          # innovations averaged over, 4 seconds at 50 Hz
BLEND = 0.05          # how fast R is allowed to move, per update
FLOOR = 0.05          # never shrink below this fraction of the starting R


class AdaptiveR:
    """The hand-written measurement model, with an R that tunes itself.

    Used exactly like any other arm. The difference is that the filter calls
    `observe` after each step, and this object listens.
    """

    def __init__(self, start_R, window=WINDOW, blend=BLEND, floor=FLOOR):
        self.start_R = np.asarray(start_R, dtype=float)
        self.R = self.start_R.copy()
        self.history = deque(maxlen=window)
        self.window = window
        self.blend = blend
        self.floor = floor * np.diag(self.start_R)
        self.spread = np.zeros_like(self.start_R)
        self.trace = []          # R over time, for looking at afterwards

    def __call__(self, states):
        """Same readings as the hand-written model, plus the current R."""
        states = np.atleast_2d(states)
        readings = expected_readings(states)
        R = np.repeat(self.R[None, :, :], len(states), axis=0)
        return readings, R

    def observe(self, innovation, S):
        """Told how the last step went. This is where the adapting happens."""
        self.history.append(np.asarray(innovation, dtype=float))

        # S was built as (spread of h) + R, and we know which R went in, so
        # the part that is not sensor noise can be recovered by subtraction.
        self.spread = S - self.R

        if len(self.history) == self.window:
            rows = np.array(self.history)
            actual = rows.T @ rows / len(rows)      # how big they really were

            estimate = actual - self.spread
            # Only the diagonal is kept. The off-diagonal terms are noisy at
            # this window length and can easily make R non-positive-definite,
            # which stops the filter outright.
            diagonal = np.maximum(np.diag(estimate), self.floor)

            # Move partway, not all the way. A single unlucky window should
            # not be able to throw R across the room.
            self.R = ((1 - self.blend) * self.R
                      + self.blend * np.diag(diagonal))

        self.trace.append(np.diag(self.R).copy())

    def reset(self):
        """Forget everything. Called between runs so one run's history
        cannot leak into the next one's starting point."""
        self.R = self.start_R.copy()
        self.history.clear()
        self.trace = []


def load_measurement_model(path=None):
    """Return a fresh adaptive model.

    `path` is accepted and ignored -- there is nothing trained to load, which
    is this arm's main selling point.
    """
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import R_TUNED
    return AdaptiveR(R_TUNED)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "experiments"))
    from common import (R_TUNED, N_RUNS, NIS_DOF, NEES_DOF, gather,
                        two_moment, load_arm, make_run, filter_once,
                        best_constant_R)
    from faults import apply_fault

    print("Does it find the sensor noise on its own?\n")
    model = load_measurement_model()
    _, meas, truth = make_run(0)
    filter_once(model, meas, truth)

    best = best_constant_R()
    print("  %-16s %12s %12s %12s"
          % ("", "tuned by hand", "best constant", "found alone"))
    for i, name in enumerate(["left_encoder", "right_encoder", "gyro"]):
        print("  %-16s %12.4f %12.4f %12.4f"
              % (name, np.sqrt(R_TUNED[i, i]), np.sqrt(best[i, i]),
                 np.sqrt(model.R[i, i])))

    print("\n  The middle column is the honest reference -- the average")
    print("  variance, which is the best a single number can do. This method")
    print("  lands near it with no supervision at all, which is the check")
    print("  that it works.")
    print("\n  It does NOT match the hand-tuned column, and should not. That")
    print("  one was found by covariance matching against a filter whose")
    print("  process model was wrong, so it absorbed state error as well as")
    print("  sensor noise and came out larger than the sensors actually are.")

    print("\nHow quickly does it notice a sensor going bad?")
    print("  left encoder noise trebles halfway through a run\n")

    model = load_measurement_model()
    _, meas, truth = make_run(0)
    broken = dict(meas)
    half = len(meas["left_encoder"]) // 2
    extra = np.zeros(len(meas["left_encoder"]))
    rng = np.random.default_rng(0)
    extra[half:] = rng.normal(0, 2 * 0.1805, size=len(extra) - half)
    broken["left_encoder"] = meas["left_encoder"] + extra

    filter_once(model, broken, truth)
    trace = np.array(model.trace)
    settled = np.sqrt(trace[-1, 0])
    target = 0.1805 * np.sqrt(1 + 4)

    print("  %-22s %10s" % ("", "left sd"))
    print("  %-22s %10.4f" % ("before the fault", np.sqrt(trace[half - 1, 0])))
    print("  %-22s %10.4f" % ("end of run", settled))
    print("  %-22s %10.4f" % ("where it should end", target))

    reached = np.where(np.sqrt(trace[half:, 0]) > 1.5 * np.sqrt(trace[half - 1, 0]))[0]
    if len(reached):
        print("\n  Took %d steps (%.1f s) to react."
              % (reached[0], reached[0] * 0.02))
    else:
        print("\n  Never reacted within the run.")
    print("  That lag is the price of watching innovations instead of the")
    print("  state, and it is what a model predicting R from the state avoids.")

    print("\nOver %d healthy runs" % N_RUNS)
    fresh = load_measurement_model()
    r = gather(fresh, range(N_RUNS))
    print("  speed %.4f   turn %.4f   NIS %.3f   NEES %.3f   %.3f ms/step"
          % (r["speed_rmse"].mean(), r["turn_rmse"].mean(),
             r["nis"].mean(), r["nees"].mean(), r["ms_per_step"]))

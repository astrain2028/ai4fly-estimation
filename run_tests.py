"""
Runs every file's self-test and reports which ones fail.

Each file in this project ends with a block that exercises what it defines --
the filter checks its own consistency, the sensor model checks its noise
calibration, each arm filters some runs and prints a table. Those are the
tests. What was missing was anything that ran them together.

That gap cost something real. Widening the filter's state from five entries
to seven broke the self-test blocks in four of the seven arms, all in the
same way, and the repository carried them in that state through a push. Each
file still parsed and imported cleanly, so nothing short of executing them
would have noticed.

    python run_tests.py            everything
    python run_tests.py robot      only files under robot/
    python run_tests.py bhr gp     only files whose path contains these

Slow ones are skipped by default because a check nobody runs is worse than no
check at all. Pass --all to include them.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ordered so that a failure appears at the lowest layer it belongs to. If the
# motion model is broken there is no point reading seven arm failures caused
# by it.
TESTS = [
    "robot/dynamics.py",
    "robot/trajectories.py",
    "robot/sensors.py",
    "robot/faults.py",
    "robot/ukf.py",
    "experiments/common.py",
    "models/fixed/measurement.py",
    "models/adaptive/measurement.py",
    "models/plain/measurement.py",
    "models/resnet/measurement.py",
    "models/bhr/measurement.py",
    "models/gp/measurement.py",
    "models/ensemble/measurement.py",
    "models/health/measurement.py",
    "models/combined/measurement.py",
    "models/doubt/measurement.py",
    "models/mmae/measurement.py",
]

# Minutes each, and they retrain models or sweep parameters. Worth running
# deliberately, not on every check.
SLOW = [
    "robot/make_dataset.py",
    "robot/make_faulted.py",
    "models/bhr/laplace.py",
    "models/doubt/laplace.py",
    "models/health/train.py",
    "experiments/healthy.py",
    "experiments/degradation.py",
    "experiments/calibration.py",
    "experiments/heteroscedasticity.py",
    "experiments/envelope.py",
    "experiments/health_value.py",
    "experiments/complementarity.py",
    "experiments/heldout.py",
    "experiments/redundancy.py",
    "experiments/timing.py",
    "experiments/tune.py",
    "experiments/bakeoff.py",
    "experiments/figures.py",
]

TIMEOUT = 900


def run(path):
    """Execute one file and report how it went."""
    started = time.time()
    try:
        done = subprocess.run([sys.executable, str(ROOT / path)],
                              capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - started, ""
    if done.returncode == 0:
        return "ok", time.time() - started, ""

    # The last few lines of a traceback say more than the first few.
    tail = [line for line in done.stderr.strip().splitlines() if line.strip()]
    return "FAILED", time.time() - started, "\n".join(tail[-3:])


def main():
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    tests = TESTS + (SLOW if "--all" in sys.argv else [])
    if words:
        tests = [t for t in tests if any(w in t for w in words)]

    if not tests:
        print("nothing matched %s" % words)
        return 1

    print("Running %d self-tests\n" % len(tests))
    failures = []
    for path in tests:
        if not (ROOT / path).exists():
            print("  %-38s missing" % path)
            continue
        status, seconds, detail = run(path)
        print("  %-38s %-8s %6.1fs" % (path, status, seconds))
        if status != "ok":
            failures.append((path, detail))

    print()
    if failures:
        print("%d of %d failed\n" % (len(failures), len(tests)))
        for path, detail in failures:
            print("--- %s" % path)
            print("%s\n" % (detail or "(no output)"))
        return 1

    print("all %d passed" % len(tests))
    if "--all" not in sys.argv:
        print("(%d slow tests skipped -- pass --all to include them)"
              % len(SLOW))
    return 0


if __name__ == "__main__":
    sys.exit(main())

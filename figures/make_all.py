"""
Redraw every figure, in both themes, from the csv files in results/.

Each figure is its own file and runs on its own; this only runs them in the
order the README presents them. A figure whose csv is missing says so and
exits cleanly, so a partial set of experiments gives a partial set of
figures rather than a failure.

    python figures/make_all.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ORDER = ["trace", "crossing", "moments", "quad", "scatter", "health_readout",
         "frozen", "cost"]


def main():
    failed = []
    for name in ORDER:
        path = HERE / ("%s.py" % name)
        if not path.exists():
            print("%-16s missing" % name)
            continue
        done = subprocess.run([sys.executable, str(path)], capture_output=True,
                              text=True)
        status = "ok" if done.returncode == 0 else "FAILED"
        print("%-16s %s" % (name, status))
        if done.returncode != 0:
            failed.append(name)
            print("\n".join(done.stderr.strip().splitlines()[-3:]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

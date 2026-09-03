"""
The figures, drawn from results/bakeoff.csv rather than from memory.

WHY THE CSV AND NOT A FRESH RUN

Everything here reads the file bakeoff.py wrote. Nothing is recomputed. That
is deliberate: a figure that reruns the experiment can disagree with the table
it sits next to, and then there is no way to tell which is right. One sweep,
one file, and both the table and the plots are views of it.

THE FIGURE THAT MATTERS

complementarity.png is the argument in one image. Two panels, one per fault
mode, speed error against severity:

    bias    -- the reading is shifted, so the innovation carries direction.
               A model conditioned on health can learn the shift and cancel
               it, and covariance matching can only widen its uncertainty.
    noise   -- the reading stays centred and spreads. There is no direction
               to read, dh/dm is zero, and no first-order update can move
               health at all. Only the size of the innovations sees it.

The two single-mechanism lines should cross between the panels: health below
adaptive on the left, above it on the right. That crossing is the claim, and
it is the sort of thing a table states and a figure shows.

The combined line should track the lower of the two in both panels. Where it
does not is as informative as where it does.

CONSISTENCY, WHICH IS THE HALF THAT SURVIVES TO HARDWARE

consistency.png plots NIS against severity on a log scale, with the target of
3 marked. NEES needs the true state and so exists only in simulation; on a
vehicle without motion capture, NIS is the entire self-assessment available.
An arm whose NIS climbs under a fault is one that does not know it is in
trouble, and that is worth seeing as a slope rather than reading as a row.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on a headless machine
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# The four arms the argument is about. The others are controls or variants and
# would only crowd the panels; every one of them is in the csv.
#
# health is drawn wide and semi-transparent, combined narrow and solid over
# the top. On the bias panel the two are identical to four decimals -- the
# multiplier sits at its floor because health has already absorbed the fault --
# and with equal linewidths the red vanishes entirely under the green. A
# reader would see three lines and conclude health was missing rather than
# that it coincides, which is the opposite of what the panel is for.
LINES = [
    ("analytic + best const R", "#888888", "o", "--", 1.6, 1.0),
    ("adaptive R (Mehra)", "#1f77b4", "s", "-", 1.8, 1.0),
    ("health-conditioned", "#d62728", "^", "-", 4.0, 0.45),
    ("combined", "#2ca02c", "D", "-", 1.6, 1.0),
]

# Where a log axis is used. Matplotlib's default decade ticks put a single
# label on this range, which is unreadable for a quantity whose target is 3.
LOG_TICKS = [1, 2, 3, 5, 10, 20]

PANELS = [("bias", "bias: the reading shifts"),
          ("noise_inflation", "noise: the reading spreads")]

# The channel whose figures keep the plain names. Anything else is suffixed,
# so two sweeps cannot land on the same file.
DEFAULT_CHANNEL = "left_encoder"


def load(name="bakeoff.csv"):
    """The sweep, or None if nobody has run it.

    A missing csv is not a failure -- there is simply nothing to draw yet --
    so main returns 0 in that case and run_tests.py does not report it as a
    broken file.
    """
    path = RESULTS / name
    if not path.exists():
        print("No %s yet -- run experiments/bakeoff.py first."
              % path.relative_to(ROOT))
        return None
    return pd.read_csv(path)


def series(frame, arm, mode, column):
    """One arm's curve for one fault mode, healthy included at severity 0."""
    healthy = frame[(frame["arm"] == arm) & (frame["mode"] == "none")]
    rows = frame[(frame["arm"] == arm) & (frame["mode"] == mode)]
    grouped = rows.groupby("severity")[column].mean()
    return ([0.0] + list(grouped.index),
            [healthy[column].mean()] + list(grouped.values))


def named(stem, channel):
    """Output name for one figure, carrying the channel it came from.

    The obvious version hardcodes the filenames, and then drawing the gyro
    sweep silently overwrites the encoder figures with a different experiment
    under the same name. Two sweeps, two sets of files.
    """
    if channel == DEFAULT_CHANNEL:
        return "%s.png" % stem
    return "%s_%s.png" % (stem, channel)


def draw(frame, column, ylabel, filename, title, log=False, target=None):
    present = [line[:4] for line in LINES if line[0] in set(frame["arm"])]
    if not present:
        print("  none of the four arms are in the csv, skipping %s" % filename)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (mode, heading) in zip(axes, PANELS):
        for arm, colour, marker, style in present:
            width, alpha = dict((l[0], (l[4], l[5])) for l in LINES)[arm]
            x, y = series(frame, arm, mode, column)
            ax.plot(x, y, style, color=colour, marker=marker, markersize=5,
                    linewidth=width, alpha=alpha, label=arm)
        if target is not None:
            ax.axhline(target, color="black", linewidth=0.8, alpha=0.5)
            ax.text(0.02, target, " target %g" % target, va="bottom",
                    fontsize=8, alpha=0.7)
        ax.set_title(heading, fontsize=10)
        ax.set_xlabel("severity")
        ax.grid(alpha=0.25, linewidth=0.6)
        if log:
            ax.set_yscale("log")
            ax.set_yticks(LOG_TICKS)
            ax.set_yticklabels([str(t) for t in LOG_TICKS])
            ax.minorticks_off()

    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    path = RESULTS / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  wrote %s" % path.relative_to(ROOT))


def main():
    frame = load(sys.argv[1] if len(sys.argv) > 1 else "bakeoff.csv")
    if frame is None:
        return 0

    channel = frame["channel"].iloc[0]
    print("Drawing from %d runs, fault on the %s\n"
          % (len(frame), channel))

    draw(frame, "speed_rmse", "speed error, m/s",
         named("complementarity", channel),
         "Each mechanism owns one fault family (%s)" % channel)

    draw(frame, "nis", "NIS", named("consistency", channel),
         "Does the filter know how wrong it is? (%s)" % channel,
         log=True, target=3.0)

    print("\nRead the left panel against the right. If health sits below")
    print("adaptive on the left and above it on the right, the two mechanisms")
    print("are covering different faults rather than the same one twice --")
    print("which is the whole claim, and the reason for carrying both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

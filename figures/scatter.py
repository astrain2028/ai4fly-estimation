"""
Why covariance matching over-inflates R on the quadcopter: its UKF claims
far less measurement scatter than its innovations turn out to show.

Reads results/scatter.csv (experiments/scatter.py). For each sensor channel,
on healthy runs of the analytic model with the best constant R, the csv holds
the scatter the filter claims over its sigma points, the scatter the
innovations actually show, and their ratio. The ratio is recomputed from the
other two columns and checked, so its direction is never assumed; the
medians per vehicle are the only other thing computed here.

    python figures/scatter.py
"""

import numpy as np
from matplotlib.ticker import NullLocator
from matplotlib.transforms import blended_transform_factory

import style

VEHICLES = [("ground robot", "Ground robot"), ("quadcopter", "Quadcopter")]

# Which channels read the state through a nonlinear map. The quadcopter's
# accelerometer and magnetometer read gravity and the magnetic field rotated
# into the body frame -- products of sines and cosines of attitude
# (quad_sim/dynamics.py: specific_force, magnetic_field). Its gyros read the
# body rates straight out of the state, and every robot channel is a linear
# combination of speed and turn rate (robot/dynamics.py: wheel_speeds).
NONLINEAR = ("accel", "mag")
SENSOR = {"accel": "accelerometer", "mag": "magnetometer", "gyro": "gyro"}
NAMES = {"left_enc": "left encoder", "right_enc": "right encoder"}

ROW = 0.25            # inches per channel row
GROUP_GAP = 0.75      # extra rows between the two vehicles' blocks
TICKS = (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def pretty(channel):
    return NAMES.get(channel, channel.replace("_", " "))


def times(r):
    """A ratio as the figure prints it: two significant figures, then ×."""
    if r >= 10:
        return "%.0f×" % r
    if r >= 1:
        return "%.1f×" % r
    return "%.2f×" % r


def load():
    data = style.read("scatter.csv")
    ratio = data["empirical"] / data["claimed"]
    # The csv's ratio must be empirical / claimed: > 1 means the filter claims
    # less scatter than it sees. Refuse to draw if it was written the other way.
    if not np.allclose(ratio, data["ratio"], rtol=1e-4):
        raise SystemExit("results/scatter.csv: ratio is not empirical/claimed")
    data["ratio"] = ratio
    return data


def layout(blocks):
    """Row positions, top to bottom: a header row per vehicle, then its
    channels. Returns the header rows, the channel rows and the last row."""
    y, heads, rows = 0.0, [], []
    for b, (_, part, _) in enumerate(blocks):
        if b:
            y += GROUP_GAP
        heads.append(y)
        y += 1
        rows.append(list(y + np.arange(len(part))))
        y += len(part)
    return heads, rows, y - 1


def draw(theme):
    data = load()
    blocks = []
    for key, name in VEHICLES:
        part = data[data["vehicle"] == key].sort_values("ratio")
        blocks.append((name, part, float(part["ratio"].median())))
    robot_med, quad_med = blocks[0][2], blocks[1][2]
    worst = data.loc[data["ratio"].idxmax()]
    worst_sensor = SENSOR.get(worst["channel"].split("_")[0], worst["channel"])

    heads, rows, last = layout(blocks)
    y_top, y_bottom = -0.75, last + 0.65

    W = style.WIDTH
    ph = (y_bottom - y_top) * ROW
    H = 1.04 + 0.30 + 0.28 + ph + 0.56 + 0.89
    fig = style.figure(H, theme)

    # The headline is computed from the medians, and says what they say --
    # that they are medians, since one vehicle's channels span two decades.
    if quad_med > 1.5 and robot_med < 1.5:
        title = ("On the quadcopter the filter sees a median %s the scatter "
                 "it claims; on the robot, %s"
                 % (times(quad_med), times(robot_med)))
    elif quad_med <= 1.5:
        title = ("Neither filter under-claims its scatter much: medians %s "
                 "and %s" % (times(robot_med), times(quad_med)))
    else:
        title = ("Both filters under-claim their scatter: medians robot %s, "
                 "quadcopter %s" % (times(robot_med), times(quad_med)))
    # The subtitle ranks only what the csv holds: how far each claim is off.
    # How far R inflates depends on each channel's R, which it does not hold.
    used = style.header(
        fig, theme, title,
        "One dot per sensor channel, healthy runs: the scatter the "
        "innovations show, divided by the scatter the UKF claims.\n"
        "Right of 1×, covariance matching books the missing scatter as "
        "sensor noise and raises R. The claim is furthest off on the %s."
        % worst_sensor)

    # No blue here: nothing on this figure is covariance matching's output.
    # The stems measure the claim's error as a ratio, which is not how far R
    # moves, so they are neutral.
    ink_dot, gray_dot = theme.ink, theme.muted
    stem = style.mix(theme.muted, theme.surface, 0.42)
    style.legend_row(fig, theme, [
        ("accelerometer, magnetometer: nonlinear in attitude", ink_dot, "dot"),
        ("encoders, gyros: linear in the state", gray_dot, "dot"),
        ("stem: how far off the claim is", stem),
    ], used + 0.14, gap=0.30)

    left, right = 1.50, 0.40
    top = used + 0.30 + 0.28
    ax = fig.add_axes([left / W, (H - top - ph) / H, (W - left - right) / W,
                       ph / H])
    style.axes_style(ax, theme, grid="x")

    lo = data["ratio"].min() / 1.45
    hi = data["ratio"].max() * 2.1
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y_bottom, y_top)
    ticks = [t for t in TICKS if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g×" % t for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    ax.set_xlabel("empirical scatter ÷ claimed scatter, log scale",
                  fontsize=8.5, color=theme.muted, labelpad=6)

    # Right of 1×: the filter claims less than it sees. The dark theme's wash
    # is mixed most of the way back toward the surface: at full strength it
    # is lighter than the grid, which then vanishes inside it, and a region
    # this large reads as a second panel.
    wash = theme.wash if theme.name == "light" else style.mix(
        theme.surface, theme.wash, 0.35)
    ax.axvspan(1, hi, color=wash, linewidth=0, zorder=0)
    ax.axvline(1, color=theme.ink2, linewidth=style.px(1), zorder=2)
    ax.text(1, y_top, "claim matches reality", ha="center", va="bottom",
            fontsize=7.5, color=theme.ink2)
    ax.text(lo, y_top, " over-claims", ha="left", va="bottom", fontsize=7.5,
            color=theme.muted)
    inset = dict(xytext=(-style.px(6), 0), textcoords="offset points",
                 annotation_clip=False)
    ax.annotate("under-claims: the gap lands in R", (hi, y_top), ha="right",
                va="bottom", fontsize=7.5, color=theme.ink2, **inset)

    # The mechanism, stated where the robot's block leaves the wash empty.
    ax.annotate("Covariance matching:  R = innovation covariance seen "
                "\N{MINUS SIGN} scatter claimed.\nScatter the filter fails "
                "to claim is booked as sensor noise.",
                (hi, (rows[0][0] + rows[0][-1]) / 2), ha="right",
                va="center", fontsize=8.5, color=theme.ink2, linespacing=1.5,
                **inset)

    label_x = blended_transform_factory(fig.transFigure, ax.transData)
    for (name, part, med), head, ys in zip(blocks, heads, rows):
        ax.text(0.28 / W, head, name, transform=label_x, ha="left",
                va="center", fontsize=10, fontweight="semibold",
                color=theme.ink)
        for yy, (_, r) in zip(ys, part.iterrows()):
            ax.text(0.44 / W, yy, pretty(r["channel"]), transform=label_x,
                    ha="left", va="center", fontsize=8.5, color=theme.ink2)
            ax.plot([1, r["ratio"]], [yy, yy], color=stem,
                    linewidth=style.px(2), zorder=3, solid_capstyle="butt")
            nonlinear = r["channel"].startswith(NONLINEAR)
            style.dot(ax, [r["ratio"]], [yy], theme,
                      ink_dot if nonlinear else gray_dot, size=9, zorder=5)

        # The vehicle's median: named at the block's head on a patch of
        # whatever it sits on, with a caret pointing down into the block. Not
        # a line -- a full-height hairline reads as a second reference rule.
        under = wash if med > 1 else theme.surface
        ax.text(med, head, "median %s" % times(med), ha="center",
                va="center", fontsize=8, color=theme.ink2, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", linewidth=0,
                          facecolor=under))
        ax.plot([med], [head + 0.5], linestyle="none", marker="v",
                markersize=style.px(8), color=theme.ink2,
                markeredgecolor=under, markeredgewidth=style.px(1.5),
                zorder=6)

    # The extreme, named with its value.
    b = [i for i, (key, _) in enumerate(VEHICLES)
         if key == worst["vehicle"]][0]
    yy = rows[b][list(blocks[b][1]["channel"]).index(worst["channel"])]
    ax.annotate(times(worst["ratio"]), (worst["ratio"], yy),
                xytext=(style.px(9), 0), textcoords="offset points",
                ha="left", va="center", fontsize=9.5, fontweight="semibold",
                color=theme.ink, annotation_clip=False)

    counts = [(int((part["ratio"] > 1).sum()), len(part))
              for _, part, _ in blocks]
    style.footnote(
        fig, theme,
        "Healthy runs of the analytic model with the best constant R, pooled "
        "over seeds. Claimed: mean diag(S \N{MINUS SIGN} R), the spread of "
        "the sigma points; empirical:\ninnovation variance \N{MINUS SIGN} "
        "diag(R). Right of 1×: %d of %d quadcopter channels, %d of %d robot "
        "channels. Left of 1× the subtraction takes too much, and R comes\n"
        "out low. The ratio ranks how far each claim is off, not how far R "
        "inflates: covariance matching adds empirical \N{MINUS SIGN} claimed "
        "to R, so the inflation\ndepends on each channel's own R, which this "
        "csv does not hold. Jiang, Shi and Moura prove that nonlinear Kalman "
        "filters underestimate their posterior\ncovariance; the claimed "
        "scatter is that covariance seen through the measurement map. "
        "Source: results/scatter.csv, experiments/scatter.py."
        % (counts[1] + counts[0]),
        bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("scatter", draw)

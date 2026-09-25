"""
An honest negative result: on the quadcopter the health entries do not read
as fault severities.

On the ground robot a bias entry settles near the severity injected, which
is what makes it a diagnostic. Here each panel is one device. A working bias
readout would put every filled dot on the diagonal: near 0 on a healthy
flight, at the injected severity under a fault. The accelerometer and gyro
bias entries move, but to a quarter or a third of the severity; the
magnetometer bias entry sits near the same value with a fault or without.
Noise entries cannot see a pure noise fault on either vehicle (dh/dm = 0),
so near 0 under a noise fault is their expected reading, not a quadcopter
failure -- those faults belong to the covariance-matching residual. The
magnetometer noise entry, which should be blind, rises anyway. The entries
still earn their place by improving accuracy (quad_sim/bakeoff.py); they
cannot be alarmed on, which is why deploy/pi_main.py leaves the level alert
off for this vehicle.

Reads results/quad_health_readout.csv (quad_sim/health_readout.py). Nothing
is recomputed; the counts in the text are tallies of its columns.

    python figures/health_readout.py
"""

import math
import textwrap

import numpy as np

import style

DEVICES = [("accel", "Accelerometer"), ("gyro", "Gyroscope"),
           ("mag", "Magnetometer")]
JITTER = 0.24          # half-width of the healthy-flight spread, severity units
NEAR_ZERO = 0.25       # a final value this low is "near 0" in the verdicts


# ---------------------------------------------------------------- data

def load():
    """The csv, with each row's device, entry kind, severity and seed.

    On a healthy row the listed entry is the one longest above 1, so above_1
    is that entry's own run. above_2 and above_3 are the longest run of ANY
    of the six entries (max over entries in quad_sim/health_readout.py), so
    they need not belong to the listed entry.
    """
    data = style.read("quad_health_readout.csv")
    data["kind"] = data["entry"].str.split("_").str[0]          # bias | noise
    data["device"] = data["entry"].str.split("_").str[1]
    data["healthy"] = data["condition"].str.startswith("healthy")
    words = data["condition"].str.split()
    data["severity"] = np.where(data["healthy"], 0.0,
                                words.str[-1].astype(float))
    data["seed"] = np.where(data["healthy"], words.str[-1].astype(float),
                            np.nan)
    return data


def rows_of(frame, device, kind):
    """One device's bias or noise rows, in order of severity."""
    return frame[(frame["device"] == device)
                 & (frame["kind"] == kind)].sort_values("severity")


def level_of(column):
    """The level an ``above_N`` column counts steps over."""
    return int(column.split("_")[1])


def span(values):
    """'95–97' for a spread of percentages, or one number if they agree."""
    lo, hi = int(round(min(values))), int(round(max(values)))
    return "%d" % lo if lo == hi else "%d–%d" % (lo, hi)


def listing(numbers, word="flight"):
    """'flight 3' or 'flights 1, 7 and 8'."""
    items = ["%d" % n for n in numbers]
    if len(items) == 1:
        return "%s %s" % (word, items[0])
    return "%ss %s and %s" % (word, ", ".join(items[:-1]), items[-1])


def number_word(n):
    return {1: "one", 2: "two", 3: "three"}.get(n, "%d" % n)


def verdicts(faulted, healthy, alert):
    """The two-line reading under each panel title.

    The numbers come from the csv. The words are checked against it too, so
    a rerun that changes what the data says stops here instead of printing a
    sentence the dots no longer support.
    """
    def share(device):              # bias entry's final, % of the severity
        b = rows_of(faulted, device, "bias")
        return 100 * b["final"] / b["severity"]

    def calm_max(device, kind):     # highest healthy-flight dot of that entry
        h = rows_of(healthy, device, kind)
        return h["final"].max() if len(h) else -np.inf

    out = {}
    s = share("accel")
    assert s.max() < 100
    assert (rows_of(faulted, "accel", "noise")["final"] < NEAR_ZERO).all()
    out["accel"] = ("bias reads %s%% of the severity;\n"
                    "noise stays near 0, as it must" % span(s))

    s = share("gyro")
    assert s.max() < 100
    assert calm_max("gyro", "bias") > rows_of(faulted, "gyro",
                                              "bias")["final"].max()
    out["gyro"] = ("bias reads %s%% of the severity,\n"
                   "and more on a healthy flight" % span(s))

    bias = rows_of(faulted, "mag", "bias")["final"]
    assert calm_max("mag", "bias") > alert and (bias > alert).all()
    assert (rows_of(faulted, "mag", "noise")["final"] > alert).all()
    out["mag"] = ("bias sits near %.1f with a fault or without;\n"
                  "noise, which should be blind, rises anyway"
                  % bias.mean())
    return out


# ---------------------------------------------------------------- marks
# style.py draws filled dots only; a noise entry is a hollow ring, so the
# ring, a text-only version of end_labels and a legend that can show both
# live here.

def filled(ax, x, y, theme, colour):
    # 10 px with a 2 px surface ring: an 8 px disc that stays legible where
    # it overlaps a line or a neighbour.
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(10),
            color=colour, markeredgecolor=theme.surface,
            markeredgewidth=style.px(2), zorder=6, clip_on=False)


def hollow(ax, x, y, theme, colour):
    # A surface disc underneath clears lines from the centre and gives the
    # ring the same 2 px halo a filled dot has.
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(13),
            color=theme.surface, markeredgewidth=0, zorder=5, clip_on=False)
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(7.5),
            markerfacecolor=theme.surface, markeredgecolor=colour,
            markeredgewidth=style.px(2), zorder=6, clip_on=False)


def mark(ax, kind, x, y, theme):
    (filled if kind == "bias" else hollow)(ax, x, y, theme, theme.orange)


def text_labels(ax, items, theme, dx=10, min_gap=14, size=8.5):
    """style.end_labels without the dots: the marks are already drawn and
    may be hollow. Texts move apart vertically only as far as they must."""
    fig = ax.figure
    scale = fig.dpi * fig.get_figwidth() / 900.0
    pts = [ax.transData.transform((x, y)) for x, y, _ in items]
    order = sorted(range(len(items)), key=lambda i: pts[i][1])
    placed, last = {}, -np.inf
    for i in order:
        placed[i] = max(pts[i][1], last + min_gap * scale)
        last = placed[i]
    shift = (np.mean([pts[i][1] for i in order])
             - np.mean([placed[i] for i in order]))
    inv = ax.transData.inverted()
    for i, (x, y, text) in enumerate(items):
        tx, ty = inv.transform((pts[i][0] + dx * scale, placed[i] + shift))
        ax.text(tx, ty, text, ha="left", va="center", fontsize=size,
                color=theme.ink, clip_on=False)


def legend(fig, theme, items, y_in, left=0.28, gap=0.30):
    """style.legend_row with extra keys: a hollow ring, a diagonal and a
    level rule."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    renderer = fig.canvas.get_renderer()
    x, y = left, 1 - y_in / H
    for label, kind in items:
        if kind in ("bias", "noise"):
            key = fig.add_axes([x / W, y - 0.1 / H, 0.2 / W, 0.2 / H])
            key.set_axis_off()
            key.set_xlim(-1, 1)
            key.set_ylim(-1, 1)
            mark(key, kind, [0], [0], theme)
            x += 0.22
        elif kind == "diag":
            fig.add_artist(style.plt.Line2D(
                [x / W, (x + 0.18) / W], [y - 0.08 / H, y + 0.08 / H],
                color=theme.ink2, linewidth=style.px(1.2),
                transform=fig.transFigure))
            x += 0.28
        else:                                            # the alert level
            fig.add_artist(style.plt.Line2D(
                [x / W, (x + 0.24) / W], [y, y], color=theme.muted,
                linewidth=style.px(1), transform=fig.transFigure))
            x += 0.32
        text = fig.text(x / W, y, label, ha="left", va="center", fontsize=9,
                        color=theme.ink2)
        x += text.get_window_extent(renderer).width / fig.dpi + gap


# ---------------------------------------------------------------- figure

def draw(theme):
    data = load()
    healthy = data[data["healthy"]]
    faulted = data[~data["healthy"]]
    alert = level_of("above_1")         # the robot's alert level, pi_main.py
    high = level_of("above_2")

    # Tallies for the text, straight from the csv's columns.
    fired = int((healthy["above_1"] > 0).sum())      # any entry crossed 1
    bias_faults = faulted[faulted["kind"] == "bias"]
    missed = int((bias_faults["above_1"] == 0).sum())   # own entry never did
    stuck = healthy[healthy["above_2"] > 0]          # an entry sat above 2
    stuck_share = 100 * stuck["above_2"] / stuck["steps"]
    quiet = healthy[healthy["above_1"] == 0]         # never crossed 1
    dipped = healthy[(healthy["above_1"] > 0)        # crossed 1, ended
                     & (healthy["final"] <= alert)]  # back under it
    steps_after = int(faulted["steps"].iloc[0])
    severities = sorted(faulted["severity"].unique())
    per_cell = faulted.groupby(["device", "kind", "severity"]).size()
    said = verdicts(faulted, healthy, alert)

    W, H = style.WIDTH, 6.30
    fig = style.figure(H, theme)
    used = style.header(
        fig, theme,
        "On the quadcopter, the health entries do not read as fault severities",
        "A working bias readout would sit on the diagonal: near 0 on a healthy "
        "flight, at the injected severity under a fault. At the robot's\n"
        "alert level, %d, an alert would fire on %d of %d healthy flights "
        "here, and the faulted device's own bias entry never crosses it on "
        "%d of %d bias faults."
        % (alert, fired, len(healthy), missed, len(bias_faults)))
    legend(fig, theme, [("bias entry", "bias"),
                        ("noise entry, blind to a noise fault by "
                         "construction (dh/dm = 0)", "noise"),
                        ("a working bias readout", "diag"),
                        ("robot's alert level", "level")],
           used + 0.16, gap=0.26)

    left, right, gap = 0.72, 0.74, 0.84
    pw = (W - left - right - 2 * gap) / 3
    ph = 2.36
    heads = used + 0.56
    top = heads + 0.78
    # The axes stop just past the last dot, so the end labels sit clear of
    # the grid and the alert rule instead of on them.
    xlim = (-0.5, severities[-1] + 0.14)
    ymax = max(5, math.ceil(data["final"].max()))

    for c, (device, name) in enumerate(DEVICES):
        x0 = left + c * (pw + gap)
        fig.text(x0 / W, 1 - heads / H, name, ha="left", va="top",
                 fontsize=11.5, fontweight="semibold", color=theme.ink)
        fig.text(x0 / W, 1 - (heads + 0.25) / H, said[device], ha="left",
                 va="top", fontsize=9, color=theme.ink2, linespacing=1.35)

        ax = fig.add_axes([x0 / W, (H - top - ph) / H, pw / W, ph / H])
        style.axes_style(ax, theme)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, ymax + 0.1)
        ax.set_yticks(range(0, ymax + 1))
        ax.set_xticks([0] + severities)
        ax.set_xticklabels(["healthy"] + ["%g" % s for s in severities])
        ax.set_xlabel("injected severity", fontsize=8.5, color=theme.muted,
                      labelpad=5)
        if c == 0:
            ax.set_ylabel("health entry, final value", fontsize=8.5,
                          color=theme.muted, labelpad=6)
        else:
            ax.tick_params(labelleft=False)                  # shared scale

        # The level the robot alarms at: above it, an alert; below, silence.
        ax.axhline(alert, color=theme.muted, linewidth=style.px(1), zorder=1)

        # The ideal: the bias entry reads the severity it was given.
        ax.plot([0, xlim[1]], [0, xlim[1]], color=theme.ink2,
                linewidth=style.px(1.2), zorder=2, solid_capstyle="butt")
        if c == 0:
            p0 = ax.transData.transform((0, 0))
            p1 = ax.transData.transform((1, 1))
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            ax.text(1.35, 1.52, "a working bias readout", rotation=angle,
                    rotation_mode="anchor", ha="left", va="bottom",
                    fontsize=7.5, color=theme.ink2)

        # Faulted flights: the matching entry at each severity, joined so
        # its slope can be read against the diagonal's. Both joins are full
        # orange, so line and key match; the noise join is a touch thinner.
        labels = []
        for kind, width in (("noise", 1.5), ("bias", 2)):
            rows = rows_of(faulted, device, kind)
            ax.plot(rows["severity"], rows["final"], color=theme.orange,
                    linewidth=style.px(width), zorder=3)
            mark(ax, kind, rows["severity"], rows["final"], theme)
            last = rows.iloc[-1]
            labels.append((last["severity"], last["final"],
                           "%s %.2f" % (kind, last["final"])))
        text_labels(ax, labels, theme, dx=13)

        # Healthy flights, spread a little in x so near-equal values show.
        # Only the dots that end above the high level are named; the rule is
        # the dot's own value, since above_2 on a healthy row may belong to
        # another entry.
        mine = healthy[healthy["device"] == device].sort_values("seed")
        offsets = (np.linspace(-JITTER, JITTER, len(mine)) if len(mine) > 1
                   else np.zeros(len(mine)))
        for dx, (_, row) in zip(offsets, mine.iterrows()):
            mark(ax, row["kind"], [dx], [row["final"]], theme)
            if row["final"] > high:
                ax.annotate("flight %d: %.2f" % (row["seed"], row["final"]),
                            (dx, row["final"]), xytext=(style.px(10), 0),
                            textcoords="offset points", ha="left",
                            va="center", fontsize=8.5, color=theme.ink,
                            annotation_clip=False)

    # Final values hide an entry that crossed the alert level and came back.
    if len(dipped) == 1:
        back = dipped.iloc[0]
        returned = ": flight %d ends at %.2f after %d unbroken steps above " \
                   "%d" % (back["seed"], back["final"], back["above_1"], alert)
    elif len(dipped) > 1:
        returned = " (%s end under %d after crossing it)" % (
            listing(dipped["seed"].astype(int)), alert)
    else:
        returned = ""
    fault_n = ("%s flight per device, fault mode and severity"
               % number_word(int(per_cell.max()))
               if per_cell.min() == per_cell.max() else
               "%d–%d flights per device, fault mode and severity"
               % (per_cell.min(), per_cell.max()))
    note = (
        "Quadcopter simulation, deployed configuration. Fault dots: the "
        "matching entry %d steps after a mid-flight onset, %s (the source "
        "script reuses one flight for both severities). Healthy dots: one "
        "per flight, the entry longest above %d (on %s none crossed it, and "
        "the csv lists %s). Values are final, so an entry can cross a level "
        "and return%s. On %s an entry stays above %d for %s%% of the flight "
        "unbroken. The robot's bias entries track severity "
        "(experiments/health_value.py); noise entries cannot see a pure "
        "noise fault on either vehicle (dh/dm = 0), and the "
        "covariance-matching residual covers those faults. The entries "
        "still improve accuracy (quad_sim/bakeoff.py); deploy/pi_main.py "
        "turns the level alert off for the quadcopter. Source: "
        "results/quad_health_readout.csv."
        % (steps_after, fault_n, alert, listing(quiet["seed"].astype(int)),
           "/".join(sorted(quiet["entry"].unique())), returned,
           listing(stuck["seed"].astype(int)), high, span(stuck_share)))
    style.footnote(fig, theme, textwrap.fill(note, 172), bottom=0.14)
    return fig


if __name__ == "__main__":
    style.render("health_readout", draw)

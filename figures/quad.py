"""
The quadcopter bake-off: calibration follows the fault type, but accuracy
does not -- health beats adaptive R on roll-and-pitch error even under noise,
the fault adaptive R is calibrated for.

Four arms, one device degraded at a time. Each line runs from a biased
device on the left, through the healthy flight in the middle, to a noisy one
on the right. The left column is roll-and-pitch RMS error; the right column
is NIS against its target of nine. Read across a row to see the two
questions part: under noise adaptive R is nearer the NIS target, yet health
has the lower error. Read down a column to compare devices: the NIS crossing
repeats whichever device is degraded, while adaptive R's error under noise
climbs only when the accelerometer is the noisy one, because roll and pitch
are observed mainly through it.

Reads results/quad_bakeoff.csv (accelerometer degraded) and
results/quad_bakeoff_mag.csv (magnetometer degraded), both written by
quad_sim/bakeoff.py. Nothing is recomputed except percentage changes, counts
of conditions, the severity from which the NIS crossing holds, and layered's
worst distance from the NIS target.

    python figures/quad.py
"""

import textwrap

import numpy as np
from matplotlib import patheffects

import style

CHANNELS = 9          # NIS target: 3 accelerometer + 3 gyro + 3 magnetometer

ANALYTIC = "analytic + best const R"
ADAPTIVE = "adaptive R (Mehra)"
HEALTH = "health-conditioned"
LAYERED = "layered"
DRAW_ORDER = [ANALYTIC, LAYERED, ADAPTIVE, HEALTH]      # story arms on top
LEFT_ARMS = [ADAPTIVE, HEALTH]      # named at the bias end: the story pair
LEGEND = [(ANALYTIC, "analytic, constant R"),
          (ADAPTIVE, "adaptive R (Mehra)"),
          (HEALTH, "health-conditioned"),
          (LAYERED, "layered")]

# (file, panel title, device in running text)
DEVICES = [("quad_bakeoff.csv", "Accelerometer degraded", "accelerometer"),
           ("quad_bakeoff_mag.csv", "Magnetometer degraded", "magnetometer")]

# (column, heading, candidate y ticks, value format)
METRICS = [("attitude_rmse_deg",
            "Accuracy — roll and pitch RMS error, degrees",
            [0.7, 1, 1.5, 2, 3], "%.2f"),
           ("nis", "Calibration — NIS, target %d" % CHANNELS,
            [9, 18, 36], "%.1f")]
HEADROOM = 1.12       # log-axis margin beyond the data, as a ratio

MODE_WORD = {"bias": "bias", "noise_inflation": "noise",
             "none": "the healthy flight"}


# ---------------------------------------------------------------- data

def ladder(frame):
    """x position of every condition: bias severities stepping left of
    healthy, noise severities stepping right, evenly spaced (ordinal)."""
    bias = sorted(frame.loc[frame["mode"] == "bias", "severity"].unique())
    noise = sorted(frame.loc[frame["mode"] == "noise_inflation",
                             "severity"].unique())
    pos = {("none", 0.0): 0}
    for i, s in enumerate(bias):
        pos[("bias", float(s))] = -(i + 1)
    for i, s in enumerate(noise):
        pos[("noise_inflation", float(s))] = i + 1
    return pos, bias, noise


def series(frame, arm, column, pos):
    part = frame[frame["arm"] == arm]
    x = np.array([pos[(m, float(s))]
                  for m, s in zip(part["mode"], part["severity"])])
    order = np.argsort(x)
    return x[order], part[column].values[order]


def value(frame, arm, mode, severity, column):
    row = frame[(frame["arm"] == arm) & (frame["mode"] == mode)
                & (frame["severity"] == severity)]
    return float(row[column].iloc[0])


def conditions(frame):
    """Every (mode, severity) the sweep ran, healthy included."""
    return sorted(set(zip(frame["mode"], frame["severity"].astype(float))))


def pct(change):
    """A percentage, with one decimal below 10% so small changes keep two
    significant figures."""
    return ("%.1f%%" if abs(change) < 0.1 else "%.0f%%") % (100 * change)


def nearer(frame, mode, severity):
    """Of health and adaptive, the arm whose NIS is closer to the target."""
    gap = {arm: abs(value(frame, arm, mode, severity, "nis") - CHANNELS)
           for arm in (HEALTH, ADAPTIVE)}
    return min(gap, key=gap.get)


def crossing_from(frame, severities):
    """The lowest severity from which, at that severity and every one above
    it, health is nearer the NIS target under bias and adaptive R under
    noise. None if the crossing fails even at the worst severity."""
    start = None
    for s in reversed(severities):
        if (nearer(frame, "bias", s), nearer(frame, "noise_inflation", s)) \
                != (HEALTH, ADAPTIVE):
            break
        start = s
    return start


# ---------------------------------------------------------------- marks

def side_labels(ax, items, theme, side="right", dx=7, min_gap=12, size=8.5):
    """Text beside endpoints, pushed apart vertically where it would collide.

    A two-sided version of style.end_labels that leaves the dots to the
    caller: ``items`` is (x, y, text); ``side`` is "right" or "left". A
    hairline leader joins text that had to move back to its point, and a
    surface-coloured halo keeps a rule or gridline from running through the
    glyphs.
    """
    fig = ax.figure
    scale = fig.dpi * fig.get_figwidth() / 900.0
    sign = 1 if side == "right" else -1
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
        tx_px = pts[i][0] + sign * dx * scale
        ty_px = placed[i] + shift
        tx, ty = inv.transform((tx_px, ty_px))
        if abs(ty_px - pts[i][1]) > 4 * scale:
            lx, _ = inv.transform((pts[i][0] + sign * 2 * scale, 0))
            ax.plot([lx, tx], [y, ty], color=theme.baseline,
                    linewidth=style.px(1), clip_on=False, zorder=4)
            tx, _ = inv.transform((tx_px + sign * 2 * scale, 0))
        ax.text(tx, ty, text, ha="left" if side == "right" else "right",
                va="center", fontsize=size, color=theme.ink, clip_on=False,
                zorder=6, path_effects=[patheffects.withStroke(
                    linewidth=style.px(4), foreground=theme.surface)])


def panel_title(ax, title, lines, theme):
    """The panel's name in ink and, beneath it, what it shows, one or two
    short lines in ink2."""
    ax.annotate("\n".join(lines), (0, 1), xycoords="axes fraction",
                xytext=(0, style.px(9)), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=theme.ink2,
                linespacing=1.4, annotation_clip=False)
    ax.annotate(title, (0, 1), xycoords="axes fraction",
                xytext=(0, style.px(29 + 16 * (len(lines) - 1))),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=theme.ink, fontweight="semibold",
                annotation_clip=False)


def text_width(fig, text, size):
    """Width in inches of ``text`` at ``size`` points in the current font."""
    t = fig.text(0, 0, text, fontsize=size)
    width = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return width


# ---------------------------------------------------------------- figure

def draw(theme):
    frames = [style.read(name) for name, _, _ in DEVICES]
    pos, bias, noise = ladder(frames[0])
    assert list(bias) == list(noise), (bias, noise)
    top_bias, top_noise = max(bias), max(noise)
    lo_x, hi_x = min(pos.values()), max(pos.values())

    # The headline is a claim about the data, so check it against the data.
    # Calibration follows the fault type: at the worst severity at least,
    # health's NIS is nearer the target under bias and adaptive's under
    # noise, on both devices.
    start = [crossing_from(f, list(bias)) for f in frames]
    assert all(s is not None for s in start), \
        "the NIS crossing fails at the worst severity: rewrite headline"

    # Accuracy does not: health has the lower error even under noise, the
    # fault adaptive is calibrated for -- at every noise severity, both
    # devices.
    def beats(f, mode, s):
        return (value(f, HEALTH, mode, s, "attitude_rmse_deg")
                < value(f, ADAPTIVE, mode, s, "attitude_rmse_deg"))
    assert all(beats(f, "noise_inflation", s) for f in frames for s in noise), \
        "adaptive now beats health on accuracy under noise: rewrite headline"
    wins = [sum(beats(f, m, s) for m, s in conditions(f)) for f in frames]
    counts = [len(conditions(f)) for f in frames]

    # Where layered, the deployed arm, has more error than the constant-R
    # baseline -- said in the footnote so the figure does not read as layered
    # winning everywhere. The healthy flight is shared by both files.
    losses, seen = [], set()
    for (_, _, device), f in zip(DEVICES, frames):
        for m, s in conditions(f):
            lay = value(f, LAYERED, m, s, "attitude_rmse_deg")
            ana = value(f, ANALYTIC, m, s, "attitude_rmse_deg")
            key = (m, s) if m == "none" else (device, m, s)
            if lay > ana and key not in seen:
                seen.add(key)
                where = (MODE_WORD[m] if m == "none" else
                         "%s %s %g" % (device, MODE_WORD[m], s))
                losses.append("%s (%.2f against %.2f degrees)"
                              % (where, lay, ana))

    W, H = style.WIDTH, 8.60
    fig = style.figure(H, theme)
    used = style.header(
        fig, theme,
        "Calibration follows the fault type, but health beats adaptive R on "
        "accuracy even under noise",
        "Each line runs from a biased device (left) through the healthy "
        "flight to a noisy one (right). Error: lower is better.\nNIS: %d is "
        "calibrated, higher is overconfident. Roll and pitch are observed "
        "mainly through the accelerometer." % CHANNELS)
    style.legend_row(
        fig, theme,
        [(label, style.arm_color(theme, arm)) for arm, label in LEGEND]
        + [("NIS target", theme.ink2, "rule")],
        used + 0.14, gap=0.26)

    left, right, gap = 0.62, 0.90, 1.16
    pw = (W - left - right - gap) / 2
    ph = 1.84
    heads = used + 0.62
    top0 = heads + 0.74
    row_gap = 1.24

    def axes(row, col):
        x = left + col * (pw + gap)
        y = H - (top0 + row * (ph + row_gap) + ph)
        return fig.add_axes([x / W, y / H, pw / W, ph / H])

    # Room on the left of every panel for the named bias-end labels, sized
    # from the widest one so all four panels share one x scale.
    widest = max(text_width(fig, "%s %s" % (style.SHORT[arm], fmt % value(
        f, arm, "bias", top_bias, column)), 8.5)
        for column, _, _, fmt in METRICS for f in frames for arm in LEFT_ARMS)
    pad_l, pad_r = widest + 0.16, 0.12                     # inches
    unit = (pw - pad_l - pad_r) / (hi_x - lo_x)            # inches per step
    xlim = (lo_x - pad_l / unit, hi_x + pad_r / unit)

    for c, (column, heading, ticks, fmt) in enumerate(METRICS):
        fig.text((left + c * (pw + gap)) / W, 1 - heads / H, heading,
                 ha="left", va="bottom", fontsize=11.5,
                 fontweight="semibold", color=theme.ink)

        # One y scale per column, from the data, so the rows compare.
        values = np.concatenate([f[column].values for f in frames])
        if column == "nis":
            values = np.append(values, CHANNELS)
        ylim = (values.min() / HEADROOM, values.max() * HEADROOM)
        yticks = [t for t in ticks if ylim[0] <= t <= ylim[1]]

        for r, ((_, device, _), frame) in enumerate(zip(DEVICES, frames)):
            ax = axes(r, c)
            style.axes_style(ax, theme)
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            ax.set_yticklabels(["%g" % t for t in yticks])
            ax.minorticks_off()
            ax.set_xlim(*xlim)
            ax.set_xticks(sorted(pos.values()))
            ax.axvline(0, color=theme.baseline, linewidth=style.px(1),
                       zorder=1)
            ax.set_xticklabels(["%g" % s for s in reversed(bias)]
                               + ["healthy"] + ["%g" % s for s in noise])
            for x, text, ha in [(-1, "←  bias severity", "right"),
                                (1, "noise severity  →", "left")]:
                ax.annotate(text, (x, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -style.px(29)),
                            textcoords="offset points", ha=ha, va="top",
                            fontsize=8.5, color=theme.muted)

            if column == "nis":
                style.rule(ax, CHANNELS, theme)

            for arm in DRAW_ORDER:
                colour = style.arm_color(theme, arm)
                x, y = series(frame, arm, column, pos)
                ax.plot(x, y, color=colour, linewidth=style.px(2), zorder=3)
                style.dot(ax, x, y, theme, colour, zorder=5)

            # Every arm named at the noise end; only the story pair at the
            # bias end, so the left edge does not read as a second axis.
            side_labels(ax, [(hi_x, v, "%s %s" % (style.SHORT[arm], fmt % v))
                             for arm in DRAW_ORDER for v in [value(
                                 frame, arm, "noise_inflation", top_noise,
                                 column)]], theme, side="right")
            side_labels(ax, [(lo_x, v, "%s %s" % (style.SHORT[arm], fmt % v))
                             for arm in LEFT_ARMS for v in [value(
                                 frame, arm, "bias", top_bias, column)]],
                        theme, side="left")

            if column == "attitude_rmse_deg":
                share = ("all %d" % counts[r] if wins[r] == counts[r]
                         else "%d of %d" % (wins[r], counts[r]))
                change = (value(frame, ADAPTIVE, "noise_inflation", top_noise,
                                column)
                          / value(frame, ADAPTIVE, "none", 0.0, column) - 1)
                lines = ["health %shas less error than adaptive R in %s "
                         "conditions" % ("again " if r else "", share),
                         "adaptive R’s error %s %s from healthy to noise %g"
                         % ("rises" if change > 0 else "falls",
                            pct(abs(change)), top_noise)]
            else:
                where = ("at every severity" if start[r] == min(bias) else
                         "from severity %g up" % start[r]
                         if start[r] < top_bias else
                         "at severity %g" % start[r])
                gap_9 = np.max(np.abs(
                    frame.loc[frame["arm"] == LAYERED, "nis"].values
                    / CHANNELS - 1))
                first = ("nearer %d %s: health under bias, adaptive R under "
                         "noise" % (CHANNELS, where) if r == 0 else
                         "the same crossing, here %s" % where)
                lines = [first, "layered stays within %.0f%% of %d in all %d "
                         "conditions" % (100 * gap_9, CHANNELS, counts[r])]
            panel_title(ax, device, lines, theme)

    if losses:
        trail = ("Layered, the deployed arm, has more error than the "
                 "analytic baseline %sunder %s."
                 % ("only " if len(losses) == 1 else "",
                    " and ".join(losses)))
    else:
        trail = ("Layered, the deployed arm, never has more error than the "
                 "analytic baseline.")
    note = ("Quadcopter simulation, attitude scored over roll and pitch (yaw "
            "wraps and is left out). Each dot is one arm under one condition, "
            "averaged over the sweep’s flights; the csv keeps no per-flight "
            "rows, so there are no bands. Severities are evenly spaced, not "
            "to scale; both y axes are logarithmic. Labels give every arm at "
            "noise %g on the right, health and adaptive R at bias %g on the "
            "left. %s Sources: results/quad_bakeoff.csv (accelerometer), "
            "results/quad_bakeoff_mag.csv (magnetometer)."
            % (top_noise, top_bias, trail))
    style.footnote(fig, theme, textwrap.fill(note, 158), bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("quad", draw)

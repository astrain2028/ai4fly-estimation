"""
A frozen sensor, and the three-line check that fixes it.

A stuck encoder repeats its last good reading while the wheel keeps turning.
That neither shifts the mean the way health expects nor widens the spread the
way covariance matching expects, so the learned filter barely moves the
error. A check that switches off a channel whose last few readings are all
identical recovers almost all of it, in front of either filter. The upper
panels show that with the 25-step window; the lower row shows why the window
is 25 steps and not 5.

Reads results/frozen.csv (5-step window) and results/frozen_w25.csv (25-step
window), both written by experiments/frozen.py. Nothing is recomputed except
ratios between rows.

    python figures/frozen.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

import style

FILES = {5: "frozen.csv", 25: "frozen_w25.csv"}
DEPLOY = 25                       # the window the upper panels show
SHORT = 5                         # the window it is compared against
CHECK = " + frozen detector"
# csv arm label -> the style module's arm name, for colour
ARMS = [("analytic", "analytic + best const R"), ("layered", "layered")]
CHANNELS = 3                      # robot: two encoders and a gyro


# ---------------------------------------------------------------- data

def load():
    frames = {w: style.read(name) for w, name in FILES.items()}
    # The bare arms do not depend on the window; say so if that ever breaks.
    for arm, _ in ARMS:
        a = frames[SHORT][frames[SHORT]["arm"] == arm]["speed"].values
        b = frames[DEPLOY][frames[DEPLOY]["arm"] == arm]["speed"].values
        if not np.allclose(a, b):
            print("warning: bare %s differs between windows" % arm)
    conditions = (frames[DEPLOY][["condition", "fraction"]]
                  .drop_duplicates().sort_values("fraction"))
    return frames, list(conditions.itertuples(index=False))


def value(frames, window, arm, condition, column="speed"):
    f = frames[window]
    row = f[(f["arm"] == arm) & (f["condition"] == condition)]
    return float(row[column].iloc[0])


def group_name(fraction):
    return "No fault" if fraction == 0 else "%.0f%% frozen" % (100 * fraction)


def signed(fraction):
    """A change as a percentage with a true minus sign: +11%, −18%."""
    text = "%+.0f%%" % (100 * fraction)
    return text.replace("-", "−")


def count(v):
    """A count as the csv means it: whole when large, one decimal when not."""
    return "%.0f" % v if v >= 10 else ("%.1f" % v).rstrip("0").rstrip(".")


# ---------------------------------------------------------------- marks

def ring(ax, x, y, theme, colour, size=9, zorder=6):
    """The hollow counterpart of style.dot: a filter without the check."""
    ax.plot([x], [y], linestyle="none", marker="o", markersize=style.px(size),
            markerfacecolor=theme.surface, markeredgecolor=colour,
            markeredgewidth=style.px(2), zorder=zorder, clip_on=False)


def filled(ax, x, y, theme, colour, size=9, zorder=7):
    ax.plot([x], [y], linestyle="none", marker="o", markersize=style.px(size),
            color=colour, markeredgecolor=theme.surface,
            markeredgewidth=style.px(2), zorder=zorder, clip_on=False)


def halo(theme):
    """A surface-coloured box behind text that sits across hairline grids."""
    return dict(boxstyle="square,pad=0.12", facecolor=theme.surface,
                edgecolor="none")


def side_labels(ax, items, theme, side="left", dx=8, min_gap=13, size=8.5):
    """Labels beside points, on either side, pushed apart where they collide.

    ``items`` is a list of (x, y, text). Like style.end_labels, but the text
    can sit to the left of the point (right-aligned) and no dot is drawn.
    """
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
    sign = -1 if side == "left" else 1
    inv = ax.transData.inverted()
    for i, (_, _, text) in enumerate(items):
        ty = placed[i] + shift
        tx = pts[i][0] + sign * dx * scale
        if abs(ty - pts[i][1]) > 4 * scale:
            a = inv.transform((pts[i][0] + sign * 4 * scale, pts[i][1]))
            b = inv.transform((tx, ty))
            ax.plot([a[0], b[0]], [a[1], b[1]], color=theme.baseline,
                    linewidth=style.px(1), clip_on=False, zorder=4)
            tx += sign * 2 * scale
        X, Y = inv.transform((tx, ty))
        ax.text(X, Y, text, ha="right" if side == "left" else "left",
                va="center", fontsize=size, color=theme.ink, clip_on=False)


def check_keys(fig, theme, x_in, y_in, gap=0.30):
    """Legend keys for the fill encoding, continuing a style.legend_row."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    renderer = fig.canvas.get_renderer()
    y = 1 - y_in / h
    for label, hollow in (("filter as it is", True),
                          ("same filter with the check", False)):
        fig.add_artist(plt.Line2D(
            [(x_in + 0.06) / w], [y], linestyle="none", marker="o",
            markersize=style.px(9),
            markerfacecolor=theme.surface if hollow else theme.ink2,
            markeredgecolor=theme.ink2 if hollow else theme.surface,
            markeredgewidth=style.px(2), transform=fig.transFigure))
        x_in += 0.18
        t = fig.text(x_in / w, y, label, ha="left", va="center", fontsize=9,
                     color=theme.ink2)
        x_in += t.get_window_extent(renderer).width / fig.dpi + gap


# ---------------------------------------------------------------- figure

def draw(theme):
    frames, conditions = load()
    faulted = [c for c in conditions if c.fraction > 0]
    colour = {arm: style.arm_color(theme, name) for arm, name in ARMS}

    def bare(arm, cond, col="speed"):
        return value(frames, DEPLOY, arm, cond, col)

    def checked(arm, cond, window=DEPLOY, col="speed"):
        return value(frames, window, arm + CHECK, cond, col)

    # The numbers the text quotes, all from the csv.
    cut = [1 - checked(a, c.condition) / bare(a, c.condition)
           for a, _ in ARMS for c in faulted]
    alone = [bare("layered", c.condition) / bare("analytic", c.condition) - 1
             for c in faulted]
    hits = [value(frames, w, a + CHECK, c.condition, "hit")
            for w in FILES for a, _ in ARMS for c in faulted]
    alarms = {w: np.mean([value(frames, w, a + CHECK, c.condition, "false")
                          for a, _ in ARMS for c in faulted]) for w in FILES}
    fewer = 1 - alarms[DEPLOY] / alarms[SHORT]
    lo, hi = 100 * min(cut), 100 * max(cut)
    span = ("%.0f%%" % lo if round(lo) == round(hi)
            else "%.0f–%.0f%%" % (lo, hi))

    # Vertical layout, in inches from the top, worked out before the figure
    # exists because the header places itself by the figure's height.
    pitch, group_gap = 1.0, 1.55                # rows, in data units
    rows = []                                   # (y, condition, arm)
    y = 0.0
    for cond in conditions:
        for arm, _ in ARMS:
            rows.append((y, cond, arm))
            y += pitch
        y += group_gap - pitch
    y_top, y_bot = -0.75, rows[-1][0] + 0.6
    row_in = 0.27                               # inches per data unit

    head = 0.28 + 0.30 + 0.20 * 2 + 0.06        # style.header, 2-line subtitle
    top = head + 0.72                           # upper panels
    ph = (y_bot - y_top) * row_in
    sec = top + ph + 0.66                       # lower section heading
    mini_top = sec + 1.14                       # lower panels
    mh = 0.98
    W, H = style.WIDTH, mini_top + mh + 1.12

    fig = style.figure(H, theme)
    used = style.header(
        fig, theme,
        "The learned filter cannot fix a frozen encoder; "
        "a three-line check cuts the error %s" % span,
        "A frozen reading is neither a shift, which health reads, nor a "
        "spread, which covariance matching reads. Left encoder frozen\nfor "
        "the final %s of the run; the check switches off any channel whose "
        "last %d readings are identical, in front of either filter."
        % (" or ".join("%.0f%%" % (100 * c.fraction) for c in faulted),
           DEPLOY))
    if abs(used - head) > 1e-9:
        print("warning: header height changed; the layout assumes %.2f in"
              % head)
    legend_y = used + 0.14
    style.legend_row(fig, theme, [(arm, colour[arm], "dot")
                                  for arm, _ in ARMS], legend_y)
    renderer = fig.canvas.get_renderer()
    end = fig.texts[-1].get_window_extent(renderer).x1 / fig.dpi
    # A wider gap: the fill keys are a second idea, not two more arms.
    check_keys(fig, theme, end + 0.56, legend_y)

    # ------------------------------------------------ upper: both panels
    left1, w1 = 1.72, 3.78
    left2, w2 = 6.12, 2.14
    ax1 = fig.add_axes([left1 / W, 1 - (top + ph) / H, w1 / W, ph / H])
    ax2 = fig.add_axes([left2 / W, 1 - (top + ph) / H, w2 / W, ph / H])

    for ax in (ax1, ax2):
        style.axes_style(ax, theme, grid="x")
        ax.set_xscale("log")
        ax.set_ylim(y_bot, y_top)
        ax.yaxis.set_major_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())

    ax1.set_xlim(0.0034, 0.3)
    ticks1 = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    ax1.xaxis.set_major_locator(FixedLocator(ticks1))
    ax1.xaxis.set_major_formatter(FixedFormatter(["%g" % t for t in ticks1]))
    ax1.set_title("Speed error, m/s", loc="left", fontsize=9.5,
                  color=theme.ink, pad=10)

    ax2.set_xlim(0.75, 650)
    ticks2 = [1, 3, 10, 30, 100, 300]
    ax2.xaxis.set_major_locator(FixedLocator(ticks2))
    ax2.xaxis.set_major_formatter(FixedFormatter(["%g" % t for t in ticks2]))
    ax2.set_title("NIS, the filter's own consistency check", loc="left",
                  fontsize=9.5, color=theme.ink, pad=10)
    for ax in (ax1, ax2):
        ax.set_xlabel("log scale", fontsize=8.5, color=theme.muted,
                      labelpad=4)

    # The consistency target, one per live channel.
    ax2.axvline(CHANNELS, color=theme.ink2, linewidth=style.px(1), zorder=1)
    ax2.text(CHANNELS, y_top, " target %d" % CHANNELS, ha="left",
             va="top", fontsize=7.5, color=theme.ink2)
    ax2.text(ax2.get_xlim()[1], y_top, "overconfident →", ha="right",
             va="top", fontsize=7.5, color=theme.muted, bbox=halo(theme))

    # Row and group labels, in the left margin.
    margin = blended_transform_factory(fig.transFigure, ax1.transData)
    for y, cond, arm in rows:
        ax1.text(-0.1, y, arm, transform=blended_transform_factory(
            ax1.transAxes, ax1.transData), ha="right", va="center",
            fontsize=8.5, color=theme.ink2)
    for cond in conditions:
        ys = [y for y, c, _ in rows if c == cond]
        fig_y = np.mean(ys)
        ax1.text(0.28 / W, fig_y, group_name(cond.fraction),
                 transform=margin, ha="left", va="center", fontsize=9,
                 fontweight="semibold", color=theme.ink)

    def label(ax, x, y, text, side=1):
        """A value beside its mark, on a surface halo so gridlines stop
        short of the digits."""
        ax.annotate(text, (x, y), xytext=(side * style.px(8), 0),
                    textcoords="offset points",
                    ha="left" if side > 0 else "right", va="center",
                    fontsize=8.5, color=theme.ink, bbox=halo(theme))

    for y, cond, arm in rows:
        c = colour[arm]
        name = cond.condition
        s0, s1 = bare(arm, name), checked(arm, name)
        n0, n1 = bare(arm, name, "nis"), checked(arm, name, col="nis")
        if cond.fraction == 0:
            # Without a fault the two coincide; one dot, and the value.
            for ax, v in ((ax1, s0), (ax2, n0)):
                filled(ax, v, y, theme, c)
            label(ax1, s0, y, "%.4f" % s0)
            continue
        for ax, v0, v1 in ((ax1, s0, s1), (ax2, n0, n1)):
            ax.plot([v1, v0], [y, y], color=c, linewidth=style.px(2),
                    solid_capstyle="butt", zorder=3)
            ring(ax, v0, y, theme, c)
            filled(ax, v1, y, theme, c)
        label(ax1, s0, y, "%.3f" % s0)
        label(ax1, s1, y, "%.4f" % s1, side=-1)
        label(ax2, n0, y, "%.0f" % n0 if n0 >= 10 else "%.1f" % n0)

    # The fault-free rows carry the check too; say that it changed nothing.
    healthy = conditions[0].condition
    same = all(abs(checked(a, healthy) - bare(a, healthy)) < 1e-12
               for a, _ in ARMS)
    if same:
        ys = [y for y, c, _ in rows if c == conditions[0]]
        ax1.text(0.021, np.mean(ys), "unchanged by the check,\n"
                 "to six figures",
                 ha="left", va="center", fontsize=8, color=theme.ink2,
                 linespacing=1.35, bbox=halo(theme))

    # ------------------------------------------------ lower: the window
    fig.text(0.28 / W, 1 - sec / H,
             "Why %d steps and not %d: the same %.0f%% detection with %.0f%% "
             "fewer false alarms" % (DEPLOY, SHORT, 100 * min(hits),
                                     100 * fewer),
             ha="left", va="top", fontsize=11.5, fontweight="semibold",
             color=theme.ink)
    fig.text(0.28 / W, 1 - (sec + 0.30) / H,
             "The encoders are quantised: a slow wheel repeats its tick count "
             "for a few steps, and a %d-step window takes that for a freeze.\n"
             "Waiting %d steps costs a little accuracy while frozen, because "
             "the check switches the channel off later."
             % (SHORT, DEPLOY),
             ha="left", va="top", fontsize=9, color=theme.ink2,
             linespacing=1.45)

    block, room_l, pw, gap = 2.64, 0.98, 1.02, 0.22
    xs = [0, 1]
    worst = faulted[-1].condition

    def mini(i, title):
        x0 = 0.28 + i * (block + gap)
        ax = fig.add_axes([(x0 + room_l) / W, 1 - (mini_top + mh) / H,
                           pw / W, mh / H])
        style.axes_style(ax, theme, grid=None)
        ax.set_xlim(-0.07, 1.07)
        ax.set_xticks(xs)
        ax.set_xticklabels(["%d steps" % SHORT, "%d steps" % DEPLOY])
        ax.yaxis.set_major_locator(NullLocator())
        fig.text(x0 / W, 1 - (mini_top - 0.14) / H, title, ha="left",
                 va="bottom", fontsize=9.5, fontweight="semibold",
                 color=theme.ink)
        return ax

    # (a) false alarms: a property of the check, not of either arm, so ink.
    ax = mini(0, "False alarms per run")
    fa = {w: [value(frames, w, "analytic" + CHECK, c.condition, "false")
              for c in faulted] for w in (SHORT, DEPLOY)}
    top_a = max(fa[SHORT]) * 1.12
    ax.set_ylim(0, top_a)
    a0, a1 = sorted(fa[SHORT]), sorted(fa[DEPLOY])
    ax.fill([0, 0, 1, 1], [a0[0], a0[1], a1[1], a1[0]],
            color=style.mix(theme.ink2, theme.surface, 0.82), linewidth=0,
            zorder=2)
    for x, (l, u) in zip(xs, (a0, a1)):
        ax.plot([x, x], [l, u], color=theme.ink2, linewidth=style.px(2),
                zorder=3)
        for v in (l, u):
            filled(ax, x, v, theme, theme.ink2, size=8)
    side_labels(ax, [(0, np.mean(a0), "%s–%s" % (count(a0[0]),
                                                      count(a0[1])))],
                theme, "left")
    side_labels(ax, [(1, np.mean(a1), "%s–%s" % (count(a1[0]),
                                                      count(a1[1])))],
                theme, "right")

    # (b) the cost of the check on a run with no fault at all.
    ax = mini(1, "Extra error on a fault-free run")
    cost = {arm: [checked(arm, healthy, w) / bare(arm, healthy) - 1
                  for w in (SHORT, DEPLOY)] for arm, _ in ARMS}
    every = [v for pair in cost.values() for v in pair]
    ax.set_ylim(min(0, min(every)), max(every) * 1.18)
    for arm, _ in ARMS:                     # the subject drawn last
        ax.plot(xs, cost[arm], color=colour[arm], linewidth=style.px(2),
                zorder=3)
        for x, v in zip(xs, cost[arm]):
            filled(ax, x, v, theme, colour[arm], size=8)
    side_labels(ax, [(0, cost[a][0], "%s  %s" % (a, signed(cost[a][0])))
                     for a, _ in ARMS], theme, "left")
    ends = [cost[a][1] for a, _ in ARMS]
    if max(abs(e) for e in ends) < 0.0005:
        side_labels(ax, [(1, 0, "0% for both")], theme, "right")
    else:
        side_labels(ax, [(1, cost[a][1], signed(cost[a][1]))
                         for a, _ in ARMS], theme, "right")

    # (c) what the longer wait costs while the encoder is frozen.
    ax = mini(2, "Speed error, %s, m/s" % group_name(faulted[-1].fraction))
    err = {arm: [checked(arm, worst, w) for w in (SHORT, DEPLOY)]
           for arm, _ in ARMS}
    ax.set_ylim(0, max(max(v) for v in err.values()) * 1.18)
    for arm, _ in ARMS:                     # the subject drawn last
        ax.plot(xs, err[arm], color=colour[arm], linewidth=style.px(2),
                zorder=3)
        for x, v in zip(xs, err[arm]):
            filled(ax, x, v, theme, colour[arm], size=8)
    side_labels(ax, [(0, err[a][0], "%s  %.4f" % (a, err[a][0]))
                     for a, _ in ARMS], theme, "left")
    side_labels(ax, [(1, err[a][1], "%.4f" % err[a][1]) for a, _ in ARMS],
                theme, "right")

    fractions = " and ".join("%.0f%%" % (100 * c.fraction) for c in faulted)
    style.footnote(
        fig, theme,
        "Ground-robot simulation, experiments/frozen.py; the encoder freezes "
        "at its last good reading. Frozen rows are scored over the frozen "
        "stretch, fault-free\nrows over the whole run. Every mark is a mean "
        "over the experiment's seeds; the csv has no per-seed rows, so there "
        "are no bands. On its own, layered\nchanges the frozen-stretch error "
        "by %s to %s against analytic. NIS target is %d, one per channel, "
        "and %d once the frozen channel is switched off.\nFalse alarms are "
        "healthy steps flagged in the %s frozen runs. Sources: "
        "results/frozen.csv (%d-step window), results/frozen_w25.csv "
        "(%d-step)."
        % (signed(min(alone)), signed(max(alone)), CHANNELS, CHANNELS - 1,
           fractions, SHORT, DEPLOY),
        bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("frozen", draw)

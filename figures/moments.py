"""
The thesis as a prediction: moment order names the mechanism that should win
each fault, including four fault modes no arm was trained on.

Reads results/moments.csv (experiments/moments.py): one row per arm, fault
mode and severity, already averaged over seeds. Each column is taken at the
top of that mode's severity ladder; each cell is the arm's speed error
divided by the analytic model's in the same column. The verdict rows repeat
the adjudication moments.py prints -- health against adaptive R, the two
single-mechanism arms -- so the figure and the table cannot disagree. The
notes also judge the lower rungs, where the verdict changes.

The footnote says what the top rung of each ladder means physically. Those
meanings differ by mode (a scale error is a gain, not a multiple of the
healthy spread), so they are computed by running robot/faults.py's own fault
functions on a probe signal rather than typed here.

    python figures/moments.py
"""

import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import style

ANALYTIC = "analytic + best const R"
ADAPTIVE = "adaptive R (Mehra)"
HEALTH = "health-conditioned"
ORDER = [ANALYTIC, ADAPTIVE, HEALTH, "combined", "layered"]
HEALTH_BASED = [HEALTH, "combined", "layered"]

# What each arm reads, in a phrase, beside its name.
ROLE = {ANALYTIC: "its own error, m/s",
        ADAPTIVE: "reads innovation size",
        HEALTH: "reads innovation direction",
        "combined": "health × a multiplier",
        "layered": "health + a residual"}

# The written-down prediction, copied from experiments/moments.py EXPECTED.
# "neither" has none, which is the point of including it.
EXPECTED = {"first": HEALTH, "second": ADAPTIVE}

MODES = ["none", "bias", "drift", "scale_error", "noise_inflation",
         "stuck", "dropout"]
NAME = {"none": "healthy", "bias": "bias", "drift": "drift",
        "scale_error": "scale error", "noise_inflation": "noise",
        "stuck": "stuck", "dropout": "dropout"}
GROUP = {"none": ("Reference", "no fault"),
         "first": ("First moment", "the reading shifts"),
         "second": ("Second", "it spreads"),
         "neither": ("Neither", "it freezes or drops out")}

# Colour bins on the ratio to analytic, symmetric in log: halved, a fifth,
# a twentieth either way. Seven bins for the seven diverging stops.
EDGES = np.array([0.5, 0.8, 0.95, 1 / 0.95, 1.25, 2.0])
TIE = 0.01          # cells within 1% of the column's best are ringed too


# ---------------------------------------------------------------- data

def verdict(rows, predicted):
    """Which single-mechanism arm has the lower error in ``rows``."""
    single = rows[rows["arm"].isin([HEALTH, ADAPTIVE])]
    if not predicted or single["arm"].nunique() < 2:
        return None
    return single.loc[single["speed_rmse"].idxmin(), "arm"]


def scoreboard(data):
    """Per mode: the top-of-ladder speed error of every arm present."""
    arms = [a for a in ORDER if a in set(data["arm"])]
    columns = []
    for mode in MODES:
        rows = data[data["mode"] == mode]
        if rows.empty:
            continue
        top = rows[rows["severity"] == rows["severity"].max()]
        low = rows[rows["severity"] == rows["severity"].min()]
        err = {a: float(top.loc[top["arm"] == a, "speed_rmse"].iloc[0])
               for a in arms if (top["arm"] == a).any()}
        moment = top["moment"].iloc[0]
        best = min(err.values())
        predicted = EXPECTED.get(moment)
        columns.append({
            "mode": mode, "moment": moment,
            "trained": bool(top["trained"].iloc[0]),
            "severity": float(top["severity"].iloc[0]),
            "low": float(low["severity"].iloc[0]),
            "err": err,
            "ratio": {a: e / err[ANALYTIC] for a, e in err.items()},
            "best": [a for a in arms if a in err and err[a] <= best * (1 + TIE)],
            "predicted": predicted,
            "actual": verdict(top, predicted),
            "actual_low": verdict(low, predicted)})
    return arms, columns


def join(words):
    return words[0] if len(words) == 1 else \
        ", ".join(words[:-1]) + " and " + words[-1]


def headline(columns):
    """The finding in words, assembled from the verdicts rather than typed."""
    tested = [c for c in columns if c["actual"]]
    hits = [c for c in tested if c["actual"] == c["predicted"]]
    misses = [c for c in tested if c not in hits]
    unseen_hit = [NAME[c["mode"]] for c in hits if not c["trained"]]
    unseen_miss = [NAME[c["mode"]] for c in misses if not c["trained"]]
    seen_miss = [NAME[c["mode"]] for c in misses if c["trained"]]
    if not misses:
        text = "Moment order called all %d faults right" % len(tested)
        return text + (", unseen %s included" % join(unseen_hit)
                       if unseen_hit else "")
    parts = []
    if unseen_hit:
        parts.append("unseen %s yes" % join(unseen_hit))
    if unseen_miss:
        parts.append("unseen %s no" % join(unseen_miss))
    if seen_miss:
        parts.append("trained %s no" % join(seen_miss))
    return ("Moment order called %d of %d faults right — %s"
            % (len(hits), len(tested), ", ".join(parts)))


def notes(columns):
    """What the grid adds beyond the verdict, as (lead, sentence) pairs built
    from the numbers rather than typed."""
    out = []
    by_mode = {c["mode"]: c for c in columns}

    for c in columns:
        if not c["actual"] or c["actual"] == c["predicted"]:
            continue
        r = c["ratio"]
        if "layered" in r and "layered" in c["best"]:
            pairs = [(style.SHORT[a], percent(r[a]))
                     for a in ("combined", ADAPTIVE) if a in r]
            others = (["%s manages %s" % pairs[0]]
                      + ["%s %s" % p for p in pairs[1:]]) if pairs else []
            out.append(("Where it missed",
                        "layered is best by far: on %s its error is %s "
                        "against analytic, where %s. It adds its residual "
                        "variance where combined multiplies one."
                        % (NAME[c["mode"]], percent(r["layered"]),
                           join(others) if others
                           else "no other arm comes close")))

    neither = [c for c in columns if c["moment"] == "neither"]
    if neither:
        said = []
        for c in neither:
            if c["best"] == [ANALYTIC]:
                said.append("nothing beats analytic on %s" % NAME[c["mode"]])
            else:
                said.append("%s is best on %s" % (
                    join([style.SHORT[a] for a in c["best"]]), NAME[c["mode"]]))
        out.append(("No call made",
                    "the frozen and missing faults follow no single "
                    "mechanism: %s." % "; ".join(said)))

    healthy = by_mode.get("none")
    paying = [a for a in HEALTH_BASED if healthy and a in healthy["ratio"]]
    if paying:
        cost = sorted(healthy["ratio"][a] for a in paying)
        span = (percent(cost[0]) if percent(cost[0]) == percent(cost[-1])
                else "%s to %s" % (percent(cost[0]), percent(cost[-1])))
        sentence = ("the health-based arms pay %s on a healthy run for their "
                    "extra states" % span)
        flipped = [c for c in columns if c["actual_low"] and c["actual"]
                   and c["actual_low"] != c["actual"]
                   and c["actual"] == c["predicted"]]
        if flipped:
            lows = sorted({c["low"] for c in flipped})
            sentence += (", enough to lose small faults: at the lowest rung "
                         "(severity %s), %s beats %s on %s"
                         % (" and ".join("%g" % s for s in lows),
                            style.SHORT[flipped[0]["actual_low"]],
                            style.SHORT[flipped[0]["actual"]],
                            join([NAME[c["mode"]] for c in flipped])))
        out.append(("Health's price", sentence + "."))
    return out


# ---------------------------------------------------------------- helpers

def luminance(hex_colour):
    c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def ink_on(fill):
    """Whichever of the two inks reads better on ``fill``."""
    def contrast(a, b):
        la, lb = sorted([luminance(a), luminance(b)], reverse=True)
        return (la + 0.05) / (lb + 0.05)
    return max([style.LIGHT.ink, style.DARK.ink],
               key=lambda ink: contrast(ink, fill))


def percent(ratio):
    change = 100 * (ratio - 1)
    if round(change) == 0:
        return "0%"
    return ("−" if change < 0 else "+") + "%.0f%%" % abs(change)


def box(ax, x, y, w, h, radius, **kw):
    """A rounded rectangle in the inch canvas (y grows downward)."""
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=%g" % radius,
                           **kw)
    ax.add_patch(patch)
    return patch


def tick(ax, cx, cy, colour, s=0.052):
    ax.plot([cx - s, cx - 0.3 * s, cx + s], [cy, cy + 0.7 * s, cy - 0.75 * s],
            color=colour, linewidth=style.px(2), solid_capstyle="round",
            solid_joinstyle="round")


def cross(ax, cx, cy, colour, s=0.042):
    for d in (1, -1):
        ax.plot([cx - s, cx + s], [cy - d * s, cy + d * s], color=colour,
                linewidth=style.px(2), solid_capstyle="round")


def width_of(fig, artist):
    return artist.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi


def wrap(text, width, fontsize):
    """Greedy word wrap to ``width`` inches, measured in the real font."""
    fig = plt.figure(figsize=(style.WIDTH, 1))
    probe = fig.text(0, 0, "", fontsize=fontsize)
    lines, line = [], ""
    for word in text.split(" "):          # a no-break space holds its pair
        trial = (line + " " + word).strip()
        probe.set_text(trial)
        if line and width_of(fig, probe) > width:
            # Never strand a sentence's first word at the end of a line.
            head, _, last = line.rpartition(" ")
            if head.endswith(".") and last[:1].isupper():
                lines.append(head)
                line = last + " " + word
            else:
                lines.append(line)
                line = word
        else:
            line = trial
    lines.append(line)
    plt.close(fig)
    return lines


def shrink(path, keep):
    """style._shrink, but with ``keep`` guaranteed in the palette. Median cut
    gives a small mark few pixels and so no palette entry of its own: an
    8 px arm dot came out in the nearest diverging red or blue."""
    try:
        from PIL import Image
    except ImportError:
        return
    image = Image.open(path).convert("RGB")
    free = 256 - len(keep)
    base = image.quantize(colors=free, method=Image.Quantize.MEDIANCUT,
                          dither=Image.Dither.NONE)
    palette = base.getpalette()[:3 * free]
    palette += [v for c in keep for v in
                (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))]
    carrier = Image.new("P", (1, 1))
    carrier.putpalette(palette)
    image.quantize(palette=carrier, dither=Image.Dither.NONE).save(
        path, optimize=True)


# ---------------------------------------------------------------- figure

def draw(theme):
    data = style.read("moments.csv")
    arms, columns = scoreboard(data)
    ratio_arms = [a for a in arms if a != ANALYTIC]

    W = style.WIDTH
    left = 0.28
    label_x = left + 0.17             # row names, after the arm dot
    grid_x = 2.02                     # where the cells begin
    right = W - 0.30
    group_gap = 0.20
    cell_gap = 0.11
    n_groups = len({c["moment"] for c in columns})
    pitch = ((right - grid_x - group_gap * (n_groups - 1) + cell_gap)
             / len(columns))          # so the last cell ends at ``right``
    cell_w = pitch - cell_gap
    ref_h = 0.36                      # the analytic row: text, no fill
    cell_h = 0.40
    row_pitch = 0.50
    note_size, note_lead = 8.5, 0.19

    wrapped = [(lead, wrap(body[0].upper() + body[1:], right - grid_x,
                           note_size)) for lead, body in notes(columns)]

    amount = max(c["severity"] for c in columns
                 if c["moment"] in ("first", "second"))
    fraction = max((c["severity"] for c in columns
                    if c["moment"] == "neither"), default=None)
    ladder = "Ladders top out at %g× the channel's healthy spread" % amount
    if fraction is not None:
        ladder += (", or for stuck and dropout at %.0f%% of the run frozen or "
                   "lost" % (100 * fraction))
    foot = wrap("Ground-robot simulation, fault on the left encoder, each "
                "cell averaged over the seeds of experiments/moments.py. "
                "%s. Held-out modes never appeared in training. Source: "
                "results/moments.csv." % ladder, right - left, 7.5)

    # Vertical plan in inches from the top, so the height follows the rows.
    groups_y = 1.86
    names_y = groups_y + 0.50
    tags_y = names_y + 0.22
    grid_y = tags_y + 0.24
    ratio_y = grid_y + ref_h + 0.10
    grid_end = ratio_y + len(ratio_arms) * row_pitch - (row_pitch - cell_h)
    pred_y = grid_end + 0.34
    out_y = pred_y + 0.42
    notes_y = out_y + 0.52
    notes_h = sum(len(lines) * note_lead + 0.10 for _, lines in wrapped)
    H = notes_y + notes_h + 0.08 + 0.155 * len(foot) + 0.12

    fig = style.figure(H, theme)
    ax = fig.add_axes([0, 0, 1, 1])        # an inch canvas, y downward
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    used = style.header(
        fig, theme, headline(columns),
        "Speed error at the top of each severity ladder, as a change against "
        "the analytic model in the same column (top row: its own error).\n"
        "The call, written before the run: first-moment faults go to health, "
        "second-moment to adaptive R, judged between those two alone.")

    # ---- key: the diverging scale, then the ring
    ky = used + 0.20
    sw, sh = 0.40, 0.15
    t = ax.text(left, ky, "better", ha="left", va="center", fontsize=8.5,
                color=theme.ink2)
    x = left + width_of(fig, t) + 0.08
    for i, colour in enumerate(theme.diverging):
        box(ax, x + i * sw + 0.012, ky - sh / 2, sw - 0.024, sh, 0.025,
            facecolor=colour, edgecolor="none")
    for i, edge in enumerate(EDGES):
        ax.text(x + (i + 1) * sw, ky + sh / 2 + 0.05, percent(edge),
                ha="center", va="top", fontsize=7.5, color=theme.muted)
    x += 7 * sw + 0.08
    t = ax.text(x, ky, "worse than analytic", ha="left", va="center",
                fontsize=8.5, color=theme.ink2)
    x += width_of(fig, t) + 0.42
    box(ax, x, ky - 0.09, 0.30, 0.18, 0.05, facecolor="none",
        edgecolor=theme.ink, linewidth=style.px(1.5))
    ax.text(x + 0.40, ky, "lowest error in the column (or within %d%% of it)"
            % round(100 * TIE), ha="left", va="center", fontsize=8.5,
            color=theme.ink2)

    # ---- column positions, grouped by moment
    xs, gx, previous = [], grid_x, None
    for c in columns:
        if previous is not None and c["moment"] != previous:
            gx += group_gap
        xs.append(gx)
        gx += pitch
        previous = c["moment"]

    # ---- group headers with a hairline beneath
    for moment in dict.fromkeys(c["moment"] for c in columns):
        members = [i for i, c in enumerate(columns) if c["moment"] == moment]
        x0 = xs[members[0]]
        x1 = xs[members[-1]] + cell_w
        title, sub = GROUP[moment]
        ax.text(x0, groups_y, title, ha="left", va="bottom", fontsize=10,
                fontweight="semibold", color=theme.ink)
        ax.text(x0, groups_y + 0.06, sub, ha="left", va="top", fontsize=8,
                color=theme.muted)
        ax.plot([x0, x1], [groups_y + 0.30] * 2, color=theme.baseline,
                linewidth=style.px(1), solid_capstyle="butt")

    # ---- column names and the trained / held-out tag
    for c, x in zip(columns, xs):
        cx = x + cell_w / 2
        ax.text(cx, names_y, NAME[c["mode"]], ha="center", va="center",
                fontsize=9.5, color=theme.ink)
        if c["mode"] == "none":
            ax.text(cx, tags_y, "severity 0", ha="center", va="center",
                    fontsize=7.5, color=theme.muted)
        elif c["trained"]:
            ax.text(cx, tags_y, "trained on", ha="center", va="center",
                    fontsize=7.5, color=theme.muted)
        else:
            ax.text(cx, tags_y, "held out", ha="center", va="center",
                    fontsize=7.5, color=theme.ink, fontweight="semibold",
                    bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.75",
                              facecolor=theme.wash, edgecolor="none"))

    # ---- row labels: the arm's own colour on a dot, the name in ink
    def arm_dot(x, y, arm):
        ax.plot([x], [y], marker="o", markersize=style.px(8),
                color=style.arm_color(theme, arm),
                markeredgecolor=theme.surface, markeredgewidth=style.px(2))

    def row_label(y_mid, name, role, arm=None):
        if arm:
            arm_dot(left + 0.05, y_mid - 0.07, arm)
        ax.text(label_x, y_mid - 0.07, name, ha="left", va="center",
                fontsize=9.5, color=theme.ink)
        ax.text(label_x, y_mid + 0.10, role, ha="left", va="center",
                fontsize=7.5, color=theme.muted)

    def ring(x, y, h):
        box(ax, x - 0.022, y - 0.022, cell_w + 0.044, h + 0.044, 0.06,
            facecolor="none", edgecolor=theme.ink, linewidth=style.px(1.5),
            zorder=4)

    # The analytic row: absolute error, the denominator of every cell below.
    row_label(grid_y + ref_h / 2, style.SHORT[ANALYTIC], ROLE[ANALYTIC],
              ANALYTIC)
    for c, x in zip(columns, xs):
        best = ANALYTIC in c["best"]
        ax.text(x + cell_w / 2, grid_y + ref_h / 2,
                "%.2g" % c["err"][ANALYTIC], ha="center", va="center",
                fontsize=9.5, color=theme.ink2,
                fontweight="semibold" if best else "normal")
        if best:
            ring(x, grid_y, ref_h)
    ax.plot([grid_x, right], [grid_y + ref_h + 0.05] * 2,
            color=theme.baseline, linewidth=style.px(1), solid_capstyle="butt")

    # The ratio rows.
    for r, arm in enumerate(ratio_arms):
        y = ratio_y + r * row_pitch
        row_label(y + cell_h / 2, style.SHORT[arm], ROLE[arm], arm)
        for c, x in zip(columns, xs):
            if arm not in c["ratio"]:
                continue
            ratio = c["ratio"][arm]
            fill = theme.diverging[int(np.searchsorted(EDGES, ratio))]
            box(ax, x, y, cell_w, cell_h, 0.04, facecolor=fill,
                edgecolor="none", zorder=2)
            best = arm in c["best"]
            ax.text(x + cell_w / 2, y + cell_h / 2, percent(ratio),
                    ha="center", va="center", fontsize=9.5,
                    color=ink_on(fill),
                    fontweight="semibold" if best else "normal", zorder=3)
            if best:
                ring(x, y, cell_h)

    # ---- the verdict: the call, then whether it held
    row_label(pred_y, "predicted", "written before the run")
    row_label(out_y, "result", "health vs adaptive R")
    for c, x in zip(columns, xs):
        cx = x + cell_w / 2
        if c["mode"] == "none":
            continue
        if not c["predicted"]:
            ax.text(cx, pred_y - 0.07, "no call", ha="center", va="center",
                    fontsize=9, color=theme.muted)
            continue
        t = ax.text(cx + 0.07, pred_y - 0.07, style.SHORT[c["predicted"]],
                    ha="center", va="center", fontsize=9, color=theme.ink)
        arm_dot(cx + 0.07 - width_of(fig, t) / 2 - 0.10, pred_y - 0.07,
                c["predicted"])
        if c["actual"] is None:
            continue
        held = c["actual"] == c["predicted"]
        t = ax.text(cx + 0.09, out_y - 0.07, "held" if held else "missed",
                    ha="center", va="center", fontsize=9.5, color=theme.ink,
                    fontweight="semibold")
        mark_x = cx + 0.09 - width_of(fig, t) / 2 - 0.12
        (tick if held else cross)(ax, mark_x, out_y - 0.07, theme.ink)
        if not held:
            ax.text(cx, out_y + 0.13, "%s won" % style.SHORT[c["actual"]],
                    ha="center", va="center", fontsize=7.5,
                    color=theme.ink2)

    # ---- what the grid adds, in sentences built from it
    ax.plot([left, right], [notes_y - 0.20] * 2, color=theme.grid,
            linewidth=style.px(1), solid_capstyle="butt")
    y = notes_y
    for lead, lines in wrapped:
        ax.text(label_x, y, lead, ha="left", va="top", fontsize=note_size,
                color=theme.ink, fontweight="semibold")
        ax.text(grid_x, y, "\n".join(lines), ha="left", va="top",
                fontsize=note_size, color=theme.ink2, linespacing=1.5)
        y += len(lines) * note_lead + 0.10

    style.footnote(fig, theme, "\n".join(foot), bottom=0.12)
    return fig


if __name__ == "__main__":
    written = style.render("moments", draw, quantize=False)
    for path, theme in zip(written, style.THEMES):
        shrink(path, [style.arm_color(theme, a) for a in ORDER]
               + list(theme.diverging) + [theme.ink, theme.ink2, theme.muted,
                                          theme.surface, theme.wash])

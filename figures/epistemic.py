"""
What layered's epistemic term is for: telling when the model is out of its
depth.

The term carries a novelty ratio -- how far the model's input sits from its
training data, as a multiple of the training average, 1 on an ordinary input.
Left, its mean over each run, one dot per seed, beside the range the healthy
runs span. Right, what the vehicle does with it in flight: how many of eight
flights the calibrated alert flagged, at the smoothing window it flies. That
window was picked on these same flights (only the level came from separate
calibration flights), so the footnote re-derives the pick from the csv and
gives the counts at the windows not flown.

Reads results/epistemic.csv and results/quad_epistemic.csv (the arm
'layered + epistemic'; experiments/epistemic.py, quad_sim/epistemic.py) and
results/novelty_detection.csv and results/novelty_threshold.csv (rows with
chosen == True; deploy/calibrate_novelty.py). Nothing is recomputed except
means, ranges, ratios and counts.

    python figures/epistemic.py
"""

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
from matplotlib.transforms import blended_transform_factory

import style

ARM = "layered + epistemic"
# Trained on first, then held out; drift and dropout beside each other, and
# scale error and stuck, so each pair's note can sit between its two rows.
TRAINED_MODES = ["bias", "noise_inflation"]
HELD_OUT_MODES = ["drift", "dropout", "scale_error", "stuck"]
NAME = {"bias": "bias", "noise_inflation": "noise", "drift": "drift",
        "scale_error": "scale error", "stuck": "stuck", "dropout": "dropout"}
QUAD_DEVICES = [("accel", "accelerometer faulted"),
                ("mag", "magnetometer faulted")]
NOVELTY = ["novelty_accel", "novelty_gyro", "novelty_mag"]
DEVICE_NAME = {"novelty_accel": "accelerometer", "novelty_gyro": "gyro",
               "novelty_mag": "magnetometer"}
FAULTED = {"accel": "accelerometer", "mag": "magnetometer"}
ENVELOPE = "aggressive x2"

ROW = 0.222           # inches per row
FOOT_WIDTH = style.WIDTH - 2 * 0.28    # inches a footnote line may run
TICKS = (0.5, 1, 2, 5, 10, 20, 50, 100)


# ---------------------------------------------------------------- data

def robot_rows(nov, det):
    """Healthy, then every condition the robot was run under, left encoder:
    the bias and noise ladders, then the held-out modes at their top rung.
    Only the top rungs were among the alert's test flights."""
    rob = nov[nov["arm"] == ARM]
    alert = det[det["vehicle"] == "robot"].set_index("condition")
    healthy = rob[rob["mode"] == "none"].set_index("seed")["novelty_mean"]
    rows = [dict(kind="row", label="healthy", tag="reference",
                 values=healthy, alert=alert.loc["healthy"])]
    rows.append(dict(kind="sub", label="left encoder faulted"))
    for mode in TRAINED_MODES + HELD_OUT_MODES:
        part = rob[rob["mode"] == mode]
        for severity in sorted(part["severity"].unique()):
            key = "left_encoder %s %.1f" % (mode, severity)
            a = alert.loc[key] if key in alert.index else None
            values = part[np.isclose(part["severity"], severity)].set_index(
                "seed")["novelty_mean"]
            rows.append(dict(kind="row",
                             label="%s %g" % (NAME[mode], severity),
                             tag="trained" if mode in TRAINED_MODES
                             else "untrained",
                             values=values, alert=a, mode=mode,
                             severity=severity))
    # The csv's own trained/untrained call must agree with the grouping.
    for r in rows[2:]:
        if r["alert"] is not None:
            assert r["alert"]["kind"] == r["tag"], r["label"]
    return rows


def quad_rows(nov, det):
    """Healthy; each device's faults at the top of their ladders; healthy
    sensors flown hard.

    A flight's novelty is the highest of its three devices' run means: the
    alert watches every device, so whichever is most unfamiliar is the one
    that would raise it.
    """
    q = nov[nov["arm"] == ARM].copy()
    q["novelty"] = q[NOVELTY].max(axis=1)
    alert = det[det["vehicle"] == "quad"].set_index("condition")
    healthy = q[(q["mode"] == "none") & (q["condition"] == "healthy")]
    # The healthy flights are flown once per faulted device; they are the
    # same flights, so they must agree.
    per_device = healthy.pivot(index="seed", columns="device",
                               values="novelty")
    assert np.allclose(per_device.min(axis=1), per_device.max(axis=1))
    rows = [dict(kind="row", label="healthy", tag="reference",
                 values=per_device.iloc[:, 0], alert=alert.loc["healthy"])]
    for device, heading in QUAD_DEVICES:
        rows.append(dict(kind="sub", label=heading))
        top = q[q["device"] == device].groupby("mode")["severity"].max()
        for mode in TRAINED_MODES + HELD_OUT_MODES:
            part = q[(q["device"] == device) & (q["mode"] == mode)
                     & np.isclose(q["severity"], top[mode])]
            a = alert.loc["%s %s %.1f" % (device, mode, top[mode])]
            tag = "trained" if mode in TRAINED_MODES else "untrained"
            assert a["kind"] == tag, (device, mode)
            rows.append(dict(kind="row",
                             label="%s %g" % (NAME[mode], top[mode]),
                             tag=tag, values=part.set_index("seed")["novelty"],
                             alert=a, device=device, mode=mode))
    hard = q[q["condition"] == ENVELOPE]
    rows.append(dict(kind="sub", label="healthy sensors, flown hard"))
    rows.append(dict(kind="row", label="at %g× the training rates"
                     % factor(ENVELOPE), tag=alert.loc[ENVELOPE, "kind"],
                     values=hard.set_index("seed")["novelty"],
                     alert=alert.loc[ENVELOPE], envelope=True))
    return rows, q


def factor(condition):
    """'aggressive x2' -> 2.0: body rates as a multiple of training's."""
    return float(condition.rsplit("x", 1)[1])


def place(blocks, sub=0.95, gap=0.9):
    """Row positions, top to bottom, in row units: a head per vehicle, then
    its rows, subheads a little shorter than a row."""
    y, out = 0.0, []
    for b, (name, rows) in enumerate(blocks):
        if b:
            y += gap
        head = y
        y += 1.05
        for r in rows:
            if r["kind"] == "sub":
                y += 0.15
                r["y"] = y
                y += sub
            else:
                r["y"] = y
                y += 1
        out.append((name, head, rows, y - 1))
    return out, y - 1


def groups(rows):
    """Runs of consecutive rows sharing a trained/held-out tag."""
    out, current = [], []
    for r in rows:
        tag = r.get("tag") if r["kind"] == "row" else None
        if current and (tag != current[0]["tag"]):
            out.append(current)
            current = []
        if tag in ("trained", "untrained"):
            current.append(r)
    if current:
        out.append(current)
    return out


def fmt(x):
    """A novelty value as printed: whole numbers from 10, else two decimals."""
    return "%.0f" % x if x >= 10 else "%.2f" % x


def count(n_above, n_runs):
    if n_above == n_runs:
        return "all %d" % n_runs
    if n_above == 0:
        return "none of %d" % n_runs
    return "%d of %d" % (n_above, n_runs)


def series(values, form):
    """'a, b and c'."""
    values = [form % v for v in values]
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def wrap(paragraphs, width=FOOT_WIDTH, size=7.5):
    """Footnote lines: each paragraph broken at spaces so no line runs past
    ``width`` inches at the footnote's size, measured, not guessed."""
    scratch = Figure(figsize=(style.WIDTH, 1), dpi=style.DPI)
    renderer = FigureCanvasAgg(scratch).get_renderer()
    probe = scratch.text(0, 0, "", fontsize=size)

    def fits(text):
        probe.set_text(text)
        return probe.get_window_extent(renderer).width / style.DPI <= width

    lines = []
    for paragraph in paragraphs:
        line = ""
        for word in paragraph.strip().split(" "):
            trial = word if not line else line + " " + word
            if line and not fits(trial):
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def window_choice(det_all, flown):
    """How the alert's smoothing window was picked, re-derived from the csv.

    deploy/calibrate_novelty.py flies the window that catches the most of the
    same held-out flights the figure shows, among windows with no false alarm
    on them, the shorter on a tie. Only the level was set on separate
    calibration flights. Returns the candidate windows, the vehicles on which
    another window tied, every window's false alarms, and the counts each
    tested condition gets at the windows not flown.
    """
    windows = sorted(det_all["time_constant"].unique())
    ties, others = {}, {}
    for vehicle, part in det_all.groupby("vehicle", sort=False):
        caught = part[part["kind"] != "false"].groupby(
            "time_constant")["alarmed"].sum()
        false = part[part["kind"] == "false"].groupby(
            "time_constant")["alarmed"].sum()
        usable = [t for t in windows if false.get(t, 0) == 0]
        best = max(usable, key=lambda t: (caught[t], -t))
        assert best == flown[vehicle], (vehicle, best, flown[vehicle])
        ties[vehicle] = [t for t in usable
                         if t != best and caught[t] == caught[best]]
        table = part.pivot_table(index="condition", columns="time_constant",
                                 values="alarmed")
        rest = [t for t in windows if t != best]
        order = part.drop_duplicates("condition")
        others[vehicle] = [
            (c, int(table.loc[c, rest].min()), int(table.loc[c, rest].max()))
            for c, kind in zip(order["condition"], order["kind"])
            if kind != "false" and table.loc[c].max() > 0]
    false_any = int(det_all.loc[det_all["kind"] == "false", "alarmed"].sum())
    return windows, ties, others, false_any


def condition_name(vehicle, condition):
    """A detection row's condition as the footnote names it."""
    if condition.startswith("aggressive x"):
        return "%g× rates" % factor(condition)
    device, mode = condition.split(" ")[:2]
    name = NAME[mode]
    return "%s %s" % (FAULTED[device], name) if vehicle == "quad" else name


# ---------------------------------------------------------------- figure

def draw(theme):
    rnov = style.read("epistemic.csv")
    qnov = style.read("quad_epistemic.csv")
    det_all = style.read("novelty_detection.csv")
    thr = style.read("novelty_threshold.csv")
    det = det_all[det_all["chosen"].astype(str) == "True"]
    thr = thr[thr["chosen"].astype(str) == "True"]
    window = thr.groupby("vehicle")["time_constant"].first()
    margin = float((thr["threshold"] / thr["healthy_peak"]).median())
    assert np.allclose(thr["threshold"] / thr["healthy_peak"], margin,
                       rtol=2e-3)
    flights = set(det["flights"])
    assert len(flights) == 1, "flight counts differ between conditions"
    n_flights = flights.pop()
    windows, ties, others, false_any = window_choice(det_all, window)

    robot = robot_rows(rnov, det)
    quad, q_all = quad_rows(qnov, det)
    blocks, last = place([("Ground robot", robot), ("Quadcopter", quad)])

    # ---- the numbers the words use
    r_rows = [r for r in robot if r["kind"] == "row"]
    q_rows = [r for r in quad if r["kind"] == "row"]
    r_healthy, q_healthy = r_rows[0]["values"], q_rows[0]["values"]
    r_lo, r_hi = r_healthy.min(), r_healthy.max()
    q_lo, q_hi = q_healthy.min(), q_healthy.max()

    def above(rows, ceiling):
        return (sum(int((r["values"] > ceiling).sum()) for r in rows),
                sum(len(r["values"]) for r in rows))

    q_faults = [r for r in q_rows[1:] if not r.get("envelope")]
    hard = [r for r in q_rows if r.get("envelope")][0]
    # Which flights the over-range faults are: the healthy flights already
    # differ, and the over-range ones are mostly the one highest healthy.
    top_seed = q_healthy.idxmax()
    over = [s for r in q_faults for s, v in r["values"].items() if v > q_hi]
    # Paired by flight: each fault flight against the same flight healthy,
    # pooled over both devices per mode; the mode that moves novelty most.
    paired = {}
    for r in q_faults:
        ratio = r["values"] / q_healthy.reindex(r["values"].index)
        paired.setdefault(r["mode"], []).append(ratio)
    paired = {m: pd.concat(v) for m, v in paired.items()}
    biggest = max(paired, key=lambda m: paired[m].median())
    big = paired[biggest]
    # Lower rungs of the quadcopter's bias and noise ladders, not drawn.
    qe = q_all[q_all["device"] != "none"]
    tops = qe.groupby(["device", "mode"])["severity"].transform("max")
    lower = qe[(qe["mode"].isin(TRAINED_MODES)) & (qe["severity"] < tops)]
    # Per device, healthy and hard flights: the story told device by device.
    q_h = q_all[(q_all["arm"] == ARM) & (q_all["condition"] == "healthy")
                & (q_all["device"] == q_all["device"].iloc[0])]
    q_x = q_all[(q_all["arm"] == ARM) & (q_all["condition"] == ENVELOPE)]
    dev_h = q_h[NOVELTY].mean()
    dev_x = q_x[NOVELTY].mean()

    # ---- footnote, built first: its line count sets the figure's height
    milder = sorted((c for c in set(det["condition"])
                     if c.startswith("aggressive x") and c != ENVELOPE),
                    key=factor)
    # Milder envelope runs are not drawn; when their count is the same at
    # every window they are said once here and left out of the spread below.
    rest = {(v, c): (lo, hi) for v in others for c, lo, hi in others[v]}
    milder_text, constant = [], set()
    for c in milder:
        row = det[det["condition"] == c].iloc[0]
        n = int(row["alarmed"])
        everywhere = rest.get((row["vehicle"], c), (0, 0)) == (n, n)
        if everywhere:
            constant.add((row["vehicle"], c))
        milder_text.append(
            "At %g× the training rates (not drawn) the alert caught %d of %d%s."
            % (factor(c), n, n_flights, " at every window" if everywhere
               else ""))
    milder_text = " ".join(milder_text)
    n_low_over = int((lower["novelty"] > q_hi).sum())

    # How the window was chosen, and what the bars read at the others.
    vehicle_name = {"robot": "the robot", "quad": "the quadcopter"}
    tied = [v for v in ("robot", "quad") if ties[v]]
    tie_text = (" (the shorter on a tie, as on %s)"
                % " and ".join(vehicle_name[v] for v in tied)) if tied else ""

    def spread(vehicle):
        # Non-breaking spaces keep each condition on one line with its count.
        return series(["%s %s" % (condition_name(vehicle, c)
                                       .replace(" ", " "),
                                       "%d" % lo if lo == hi
                                       else "%d–%d" % (lo, hi))
                       for c, lo, hi in others[vehicle]
                       if (vehicle, c) not in constant], "%s")

    false_text = ("No window raised a false alarm" if false_any == 0 else
                  "The windows raised %d false alarms between them"
                  % false_any)
    foot = wrap([
        "Simulation, %d seeds per row. Robot novelty: the highest channel's "
        "at each step, averaged over the run. Quadcopter: the highest of its "
        "three devices' run means; per device (%s), healthy flights average "
        "%s, and %s at %g× the rates. Quadcopter faults are shown at the top "
        "of their ladders; of the %d flights on lower rungs of bias and "
        "noise, %d %s above the healthy range."
        % (len(r_healthy), ", ".join(DEVICE_NAME[c] for c in NOVELTY),
           series(dev_h[NOVELTY].values, "%.2f"),
           series(dev_x[NOVELTY].values, "%.0f"), factor(ENVELOPE),
           len(lower), n_low_over, "is" if n_low_over == 1 else "are"),
        "The alert column replays the vehicle's own runtime on its own "
        "simulated flights of the same seeds. It smooths novelty over %g s on "
        "the robot and %g s on the quadcopter and fires above %.3g× the "
        "highest level separate healthy calibration flights reached. Only "
        "that level was set apart: each window is the one of %s s that "
        "caught the most of these same flights with no false alarm%s. At the "
        "other windows the robot's counts are %s; the quadcopter's, %s. %s. "
        "%s"
        % (window["robot"], window["quad"], margin, series(windows, "%g"),
           tie_text, spread("robot"), spread("quad"), false_text,
           milder_text),
        "Sources: results/epistemic.csv, quad_epistemic.csv, "
        "novelty_detection.csv, novelty_threshold.csv."])

    W = style.WIDTH
    left = 0.28
    name_x = 0.44
    dots_l, dots_r = 2.42, 6.02
    mean_x = 6.50                      # centre of the mean column
    bars_l, bars_r = 7.02, 8.30
    top_pad, bottom_pad = 0.62, 0.62   # row units above the first head / below
    ph = (last + top_pad + bottom_pad) * ROW

    foot_lines = len(foot)
    used_guess = 0.28 + 0.30 + 0.20 * 2 + 0.06     # header with two lines
    axes_top = used_guess + 0.30 + 0.56
    H = axes_top + ph + 0.52 + 0.155 * foot_lines + 0.12
    fig = style.figure(H, theme)

    used = style.header(
        fig, theme,
        "The model can tell when it is out of its depth: clearly on the "
        "robot, partly on the quadcopter",
        "Novelty: how far the model's input sits from its training data, as "
        "a multiple of the training average (1 = an ordinary input).\n"
        "Left: its mean over each run, one dot per seed. Right: how many of "
        "%d flights the vehicle's calibrated alert flagged." % n_flights)

    wash = theme.wash if theme.name == "light" else style.mix(
        theme.surface, theme.wash, 0.45)
    style.legend_row(fig, theme, [
        ("one run of layered, its mean novelty", theme.aqua, "dot"),
        ("range of the %d healthy runs" % len(r_healthy), wash, "band"),
        ("flights the alert flagged", theme.aqua, "band"),
    ], used + 0.14, gap=0.34)

    y_top, y_bot = -top_pad, last + bottom_pad
    bottom = H - axes_top - ph

    # ---- the novelty strip
    ax = fig.add_axes([dots_l / W, bottom / H, (dots_r - dots_l) / W, ph / H])
    style.axes_style(ax, theme, grid="x")
    lo = 0.42
    hi = 100
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([t for t in TICKS if lo <= t <= hi])
    ax.set_xticklabels(["%g×" % t for t in TICKS if lo <= t <= hi])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([])
    ax.set_xlabel("novelty, mean over the run, × the training average "
                  "(log scale)", fontsize=8.5, color=theme.muted, labelpad=6)

    # ---- the alert bars
    bx = fig.add_axes([bars_l / W, bottom / H, (bars_r - bars_l) / W, ph / H])
    style.axes_style(bx, theme, grid="x")
    bx.set_xlim(0, n_flights)
    bx.set_ylim(y_bot, y_top)
    bx.set_xticks(range(0, n_flights + 1, 2))
    bx.set_yticks([])
    bx.set_xlabel("flights flagged, of %d" % n_flights, fontsize=8.5,
                  color=theme.muted, labelpad=6)

    # Column titles.
    title_y = 1 - (axes_top - 0.24) / H
    fig.text(dots_l / W, title_y, "Novelty over each run", ha="left",
             va="bottom", fontsize=9.5, fontweight="semibold", color=theme.ink)
    fig.text(mean_x / W, title_y, "mean", ha="center", va="bottom",
             fontsize=8.5, color=theme.muted)
    fig.text(bars_l / W, title_y, "In-flight alert", ha="left", va="bottom",
             fontsize=9.5, fontweight="semibold", color=theme.ink)

    # 1 = the training average.
    ax.axvline(1, color=theme.ink2, linewidth=style.px(1), zorder=2)
    ax.annotate("training average", (1, y_top), xytext=(0, style.px(3)),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.5, color=theme.ink2, annotation_clip=False)

    fig_y = blended_transform_factory(fig.transFigure, ax.transData)
    for (name, head, rows, end), (hlo, hhi) in zip(
            blocks, [(r_lo, r_hi), (q_lo, q_hi)]):
        ax.text(left / W, head, name, transform=fig_y, ha="left",
                va="center", fontsize=10, fontweight="semibold",
                color=theme.ink)
        # The healthy runs' range, washed across the vehicle's block.
        first = rows[0]["y"] - 0.5
        ax.add_patch(Rectangle((hlo, first), hhi - hlo, end + 0.5 - first,
                               facecolor=wash, edgecolor="none", zorder=0))
        ax.text(np.sqrt(hlo * hhi), head + 0.12, "healthy %s–%s"
                % (fmt(hlo), fmt(hhi)), ha="center", va="center",
                fontsize=7.5, color=theme.ink2, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", linewidth=0,
                          facecolor=theme.surface))

        for r in rows:
            if r["kind"] == "sub":
                ax.text(name_x / W, r["y"], r["label"], transform=fig_y,
                        ha="left", va="center", fontsize=8, color=theme.muted,
                        style="italic")
                continue
            ax.text(name_x / W, r["y"], r["label"], transform=fig_y,
                    ha="left", va="center", fontsize=8.5, color=theme.ink2)
            values = np.sort(r["values"].values)
            style.dot(ax, values, np.full(len(values), r["y"]), theme,
                      theme.aqua, size=7.5, zorder=5)
            ax.text(mean_x / W, r["y"], fmt(values.mean()), transform=fig_y,
                    ha="center", va="center", fontsize=8.5,
                    color=theme.ink)

        # Trained on / held out, once per run of rows, with a bracket.
        for g in groups(rows):
            held = g[0]["tag"] == "untrained"
            y0, y1 = g[0]["y"] - 0.34, g[-1]["y"] + 0.34
            ax.plot([(dots_l - 0.10) / W] * 2, [y0, y1], transform=fig_y,
                    color=theme.baseline, linewidth=style.px(1),
                    solid_capstyle="butt", clip_on=False, zorder=1)
            ax.text((dots_l - 0.17) / W, (g[0]["y"] + g[-1]["y"]) / 2,
                    "held out" if held else "trained on", transform=fig_y,
                    ha="right", va="center", fontsize=7.5,
                    color=theme.ink if held else theme.muted,
                    fontweight="semibold" if held else "normal")

    # Bars last: hbar converts its radius through the final limits.
    for name, head, rows, end in blocks:
        for r in rows:
            if r["kind"] != "row":
                continue
            if r["alert"] is None:
                bx.annotate("not tested", (0, r["y"]), xytext=(style.px(6), 0),
                            textcoords="offset points", ha="left",
                            va="center", fontsize=7.5, color=theme.muted,
                            annotation_clip=False)
                continue
            n = int(r["alert"]["alarmed"])
            if n:
                style.hbar(bx, r["y"], n, theme.aqua, theme, thickness=11,
                           radius=3)
            bx.annotate("%d" % n, (n, r["y"]), xytext=(style.px(6), 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.5,
                        color=theme.ink if n else theme.muted,
                        fontweight="semibold" if n else "normal",
                        annotation_clip=False)

    # ---- what the strip shows, in words built from it
    note = dict(ha="left", va="center", fontsize=8, color=theme.ink2,
                linespacing=1.4, zorder=6)
    NOTE_X = 10.5

    def mid(rows):
        return np.mean([r["y"] for r in rows])

    by_mode = {}
    for r in r_rows[1:]:
        by_mode.setdefault(r["mode"], []).append(r)
    bias = by_mode["bias"]
    bias_means = [r["values"].mean() for r in bias]
    assert all(np.diff(bias_means) > 0)    # the note says it rises
    # Rung by rung, including those well inside training: novelty follows how
    # far the fault pushes health, not only whether it was held out.
    rung_over = [int((r["values"] > r_hi).sum()) for r in bias]
    rung_runs = {len(r["values"]) for r in bias}
    assert len(rung_runs) == 1
    ax.text(NOTE_X, mid(bias), "bias: rises with its size, even\ninside "
            "training: %s of %d\nruns above the healthy range;\n%g is the "
            "edge of training"
            % (series(rung_over, "%d"), rung_runs.pop(), bias[-1]["severity"]),
            **note)
    noise = by_mode["noise_inflation"]
    ax.text(NOTE_X, mid(noise), "noise: %s runs above\nthe healthy range"
            % count(*above(noise, r_hi)), **note)
    for pair in (("drift", "dropout"), ("scale_error", "stuck")):
        rows_ = [r for m in pair for r in by_mode[m]]
        ax.text(NOTE_X, mid(rows_), "%s: %s\nabove the healthy range"
                % (", ".join(NAME[m] for m in pair),
                   count(*above(rows_, r_hi))), **note)

    accel = [r for r in q_faults if r["device"] == "accel"]
    mag = [r for r in q_faults if r["device"] == "mag"]
    n_over, n_fault = above(q_faults, q_hi)
    ax.text(NOTE_X, mid(accel[2:4]) + 0.5,
            "faults, either device:\n%s above the healthy range,\n%d of them "
            "on the one flight\nalready highest when healthy"
            % (count(n_over, n_fault), over.count(top_seed)), **note)
    ax.text(NOTE_X, mid(mag[1:3]) + 0.5,
            "flight for flight, %s\nraises novelty on %s, by\na median "
            "×%.1f; healthy flights\nalready span %.1f-fold"
            % (NAME[biggest], count(int((big > 1).sum()), len(big)),
               big.median(), q_hi / q_lo), **note)
    sub = [r for r in quad if r["kind"] == "sub"][-1]
    ax.text(7.0, sub["y"] - 0.05,
            "%s above the healthy range:\nlowest %s, against %s"
            % (count(*above([hard], q_hi)), fmt(hard["values"].min()),
               fmt(q_hi)), **note)

    style.footnote(fig, theme, "\n".join(foot), bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("epistemic", draw)

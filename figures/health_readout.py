"""
The quadcopter's health entries, read correctly: quiet on healthy flights
once the filter starts from the true first state (heading included), and
still low ten seconds into a fault.

Top: the same eight healthy flights run twice, from the true first state (as
every other experiment starts) and from all zeros (heading 0, as the earlier
published readout did). Each dot is the highest value any of the six health
entries reaches over the flight. Ordered by how far the flight's true
starting heading is from 0, the zero-start artifact sits exactly where the
filter begins half a turn wrong.

Bottom: the matching entry ten seconds after a fault arrives mid-flight,
against the severity injected. A working readout would sit on the diagonal
once settled; ten seconds is not settled, which the ground robot's own trace
shows (results/trace.csv), so the bias dots cannot say slow from wrong.
Noise entries are blind to a pure noise fault on either vehicle (dh/dm = 0);
the covariance-matching residual is what covers those, which the quadcopter
bakeoff shows through NIS.

Reads results/quad_health_readout.csv (quad_sim/health_readout.py); for the
robot comparison, results/trace.csv (experiments/trace.py); for the noise
claim, results/quad_bakeoff.csv and results/quad_bakeoff_mag.csv
(quad_sim/bakeoff.py). The starting headings are not in the csv; they are
recomputed from the flights themselves (quad_sim/trajectories.py,
deterministic by seed).

    python figures/health_readout.py
"""

import math
import sys
import textwrap

import numpy as np

import style

sys.path.insert(0, str(style.ROOT / "quad_sim"))
import trajectories                           # noqa: E402  (headings only)

DEVICES = [("accel", "Accelerometer"), ("gyro", "Gyroscope"),
           ("mag", "Magnetometer")]
QUIET = 2.0          # the level no truth-started healthy flight may pass
ROBOT_ALERT = 1.0    # deploy/pi_main.py HEALTH_ALERT["ground_robot"]
CHANNELS = 9         # NIS target: 3 accelerometer + 3 gyro + 3 magnetometer
BAKEOFFS = [("quad_bakeoff.csv", "accelerometer"),          # quad_sim/bakeoff.py
            ("quad_bakeoff_mag.csv", "magnetometer")]
MINUS = "−"


# ---------------------------------------------------------------- data

def load():
    """The csv, with each row's device, entry kind, severity and seed."""
    data = style.read("quad_health_readout.csv")
    data["kind"] = data["entry"].str.split("_").str[0]          # bias | noise
    data["device"] = data["entry"].str.split("_").str[1]
    data["healthy"] = data["condition"].str.startswith("healthy")
    words = data["condition"].str.split()
    data["severity"] = np.where(data["healthy"], 0.0,
                                words.str[-1].astype(float))
    data["seed"] = np.where(data["healthy"], words.str[-1].astype(float),
                            np.nan)
    data["mode"] = np.where(data["healthy"], "healthy", words.str[1])
    return data


def heading(seed):
    """A healthy flight's true starting heading, degrees, as the source
    script prints it: the first row of the flight's truth, yaw column."""
    flight = trajectories.random_run(int(seed))
    return float(np.degrees(trajectories.truth_matrix(flight)[0, 2]))


def robot_reference(after):
    """The ground robot's median bias entry ``after`` seconds past a
    mid-run severity-3 bias onset, and at the end of the run, from
    results/trace.csv. None if that experiment has not been run."""
    path = style.RESULTS / "trace.csv"
    if not path.exists():
        return None
    import pandas as pd
    trace = pd.read_csv(path)
    run = trace[(trace["arm"] == "layered") & (trace["scenario"] == "bias")]
    onset = float(run.loc[run["severity"] > 0, "t"].min())
    severity = float(run["severity"].max())
    table = run.pivot(index="t", columns="seed", values="bias_left")
    median = table.median(axis=1)
    at = float(median.iloc[np.argmin(np.abs(median.index.values
                                            - (onset + after)))])
    end_t = float(median.index.max())
    return dict(onset=onset, severity=severity, at=at,
                end=float(median.iloc[-1]), end_after=end_t - onset,
                seeds=table.shape[1])


def noise_cover():
    """Mean NIS under a noise fault in the quadcopter bakeoff, layered
    against health-conditioned, over whichever degraded devices have been
    run. None if neither has."""
    import pandas as pd
    frames, devices = [], []
    for name, device in BAKEOFFS:
        path = style.RESULTS / name
        if path.exists():
            frames.append(pd.read_csv(path))
            devices.append(device)
    if not frames:
        return None
    runs = pd.concat(frames)
    noise = runs[runs["mode"] == "noise_inflation"]
    layered = noise.loc[noise["arm"] == "layered", "nis"]
    health = noise.loc[noise["arm"] == "health-conditioned", "nis"]
    return dict(devices=devices, files=[n for n, d in BAKEOFFS
                                        if d in devices],
                severity=float(noise["severity"].max()),
                layered=(float(layered.min()), float(layered.max())),
                health=(float(health.min()), float(health.max())))


def span(lo, hi, fmt="%.2f"):
    a, b = fmt % lo, fmt % hi
    return a if a == b else "%s–%s" % (a, b)


def signed(deg):
    return ("+%.0f°" % deg) if round(deg) > 0 else (
        "%s%.0f°" % (MINUS, abs(deg)) if round(deg) < 0 else "0°")


# ---------------------------------------------------------------- marks
# style.py draws filled dots only. A noise entry is a hollow ring and a
# zero-start peak a tint of the health colour, so those marks, a text-only
# version of end_labels and a legend that can show them all live here.

def filled(ax, x, y, theme, colour, zorder=6):
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(10),
            color=colour, markeredgecolor=theme.surface,
            markeredgewidth=style.px(2), zorder=zorder, clip_on=False)


def hollow(ax, x, y, theme, colour):
    # A surface disc underneath clears lines from the centre and gives the
    # ring the same 2 px halo a filled dot has.
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(13),
            color=theme.surface, markeredgewidth=0, zorder=5, clip_on=False)
    ax.plot(x, y, linestyle="none", marker="o", markersize=style.px(7.5),
            markerfacecolor=theme.surface, markeredgecolor=colour,
            markeredgewidth=style.px(2), zorder=6, clip_on=False)


def text_labels(ax, items, theme, dx=10, min_gap=14, size=8.5, ha="left"):
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
    sign = 1 if ha == "left" else -1
    for i, (x, y, text) in enumerate(items):
        tx, ty = inv.transform((pts[i][0] + sign * dx * scale,
                                placed[i] + shift))
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=size,
                color=theme.ink, clip_on=False)


def legend(fig, theme, items, y_in, left=0.28, gap=0.30):
    """style.legend_row with the keys this figure needs: a filled dot in
    any colour, a hollow ring, a diagonal and a level rule."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    renderer = fig.canvas.get_renderer()
    x, y = left, 1 - y_in / H
    for label, kind, colour in items:
        if kind in ("dot", "ring"):
            key = fig.add_axes([x / W, y - 0.1 / H, 0.2 / W, 0.2 / H])
            key.set_axis_off()
            key.set_xlim(-1, 1)
            key.set_ylim(-1, 1)
            (filled if kind == "dot" else hollow)(key, [0], [0], theme,
                                                  colour)
            x += 0.22
        elif kind == "diag":
            fig.add_artist(style.plt.Line2D(
                [x / W, (x + 0.18) / W], [y - 0.08 / H, y + 0.08 / H],
                color=colour, linewidth=style.px(1.2),
                transform=fig.transFigure))
            x += 0.28
        else:                                                   # a rule
            fig.add_artist(style.plt.Line2D(
                [x / W, (x + 0.24) / W], [y, y], color=colour,
                linewidth=style.px(1), transform=fig.transFigure))
            x += 0.32
        text = fig.text(x / W, y, label, ha="left", va="center", fontsize=9,
                        color=theme.ink2)
        x += text.get_window_extent(renderer).width / fig.dpi + gap


def section(fig, theme, x_in, y_in, title):
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.text(x_in / W, 1 - y_in / H, title, ha="left", va="top",
             fontsize=11.5, fontweight="semibold", color=theme.ink)


# ---------------------------------------------------------------- figure

def draw(theme):
    data = load()
    healthy = data[data["healthy"]]
    truth = healthy[healthy["start"] == "truth"].set_index("seed")
    zero = healthy[healthy["start"] == "zero"].set_index("seed")
    faulted = data[~data["healthy"] & (data["start"] == "truth")]
    # The zero start is the superseded run: the health colour, receded
    # toward the surface. Less on dark, where the same mix goes muddy.
    tint = style.mix(theme.orange, theme.surface,
                     0.5 if theme.name == "light" else 0.34)

    # ---- the numbers the text states, all from the csv
    dt = trajectories.DT
    seeds = sorted(truth.index.astype(int))
    heads = {s: heading(s) for s in seeds}
    order = sorted(seeds, key=lambda s: abs(heads[s]))
    flight_s = truth["steps"].iloc[0] * dt
    truth_max = truth["peak"].max()
    truth_worst = int(truth["peak"].idxmax())
    assert truth_max < QUIET, "a truth-started healthy flight passes 2"
    stuck = zero[zero["above_2"] > 0]                  # held an entry above 2
    stuck_seeds = sorted(stuck.index.astype(int))
    stuck_share = 100 * stuck["above_2"] / stuck["steps"]
    stuck_far = min(abs(heads[s]) for s in stuck_seeds)
    others_far = max([abs(heads[s]) for s in seeds if s not in stuck_seeds]
                     or [0])
    assert stuck_far > others_far, "the stuck flights are not the far ones"
    crossed = truth[truth["above_1"] > 0]              # crossed the robot level
    after_s = faulted["steps"].iloc[0] * dt
    bias3 = faulted[(faulted["kind"] == "bias")
                    & (faulted["severity"] == faulted["severity"].max())]
    noise = faulted[faulted["kind"] == "noise"]
    top_severity = faulted["severity"].max()
    severities = sorted(faulted["severity"].unique())
    robot = robot_reference(after_s)

    near = (min(abs(heads[s]) for s in stuck_seeds),
            max(abs(heads[s]) for s in stuck_seeds))
    assert near[0] >= 150, "the stuck flights are not near 180 degrees"
    count = {1: "one", 2: "two", 3: "three", 4: "four"}.get(
        len(stuck), "%d" % len(stuck))

    # ---- the words, so the layout can be sized to them
    title = ("Started from the true first state, the quadcopter's health "
             "entries stay under %g when healthy" % QUIET)
    subtitle = (
        "The earlier readout started the filter from zeros, at heading 0, "
        "and %s healthy flights read as faulty: the %s heading near 180°.\n"
        "Under a real fault the entries read low: %.0f s after a severity-%g "
        "bias arrives mid-flight, the matching entry reads %s, not %g."
        % (count, count, after_s, top_severity,
           span(bias3["final"].min(), bias3["final"].max()), top_severity))
    if robot is not None:
        fault_note = (
            "Once settled, a working readout would sit on the diagonal. "
            "%.0f s is not settled: on the ground robot, the same severity-%g "
            "onset leaves the bias entry at %.2f after %.0f s and %.2f after "
            "%.0f s (median of %d runs). The bias dots cannot tell slow from "
            "wrong."
            % (after_s, robot["severity"], robot["at"], after_s, robot["end"],
               robot["end_after"], robot["seeds"]))
    else:
        fault_note = ("Once settled, a working readout would sit on the "
                      "diagonal; %.0f s after a mid-flight onset is not "
                      "settled, so the bias dots cannot tell slow from "
                      "wrong."
                      % after_s)
    fault_note = textwrap.fill(fault_note, 128)

    # Every healthy flight has the same length, so the step count is said
    # once, on the first flight named.
    crossed_txt = ", ".join(
        ("flight %d for %d unbroken steps of %d" if i == 0
         else "flight %d for %d")
        % ((s, r["above_1"], r["steps"]) if i == 0 else (s, r["above_1"]))
        for i, (s, r) in enumerate(crossed.sort_index().iterrows()))
    assert truth["steps"].nunique() == 1
    crossed_n = {1: "One flight", 2: "Two flights", 3: "Three flights"}.get(
        len(crossed), "%d flights" % len(crossed))
    sources = ["results/quad_health_readout.csv"]
    if robot is not None:
        sources.append("results/trace.csv")
    cover = noise_cover()
    if cover is not None:
        # Only say the residual covers noise faults if the bakeoff agrees.
        assert cover["layered"][1] < cover["health"][1], \
            "layered's NIS under noise is not below health's"
        cover_txt = (
            ". The covariance-matching residual covers those: under %s noise "
            "up to severity %g, layered's mean NIS stays at %s against a "
            "target of %d, while health-conditioned's reaches %.1f"
            % (" or ".join(cover["devices"]), cover["severity"],
               span(*cover["layered"], fmt="%.1f"), CHANNELS,
               cover["health"][1]))
        sources += ["results/" + name for name in cover["files"]]
    else:
        cover_txt = ""
    foot = (
        "Quadcopter simulation, deployed configuration (deploy/runtime.py), "
        "%.0f Hz. Healthy: %d flights of %.0f s; a dot is the highest value "
        "any of the six entries reaches over the flight. %s started from the "
        "truth do cross %g, the ground robot's alert level (%s), and "
        "deploy/pi_main.py leaves the quadcopter's level alert off. Faults: "
        "one flight per device, reused for both severities; noise entries "
        "peak at %.2f at most under a noise fault%s. Headings from "
        "quad_sim/trajectories.py. Sources: %s."
        % (1 / dt, len(seeds), flight_s, crossed_n, ROBOT_ALERT, crossed_txt,
           noise["peak"].max(), cover_txt, ", ".join(sources)))
    foot = textwrap.fill(foot, 176)

    # ---- layout, inches from the top
    used = 0.28 + 0.30 + 0.20 * (subtitle.count("\n") + 1) + 0.06
    left, right = 0.78, 1.30
    head_a = used + 0.14
    key_a = head_a + 0.44
    top_a, ph_a = key_a + 0.26, 2.2
    head_b = top_a + ph_a + 0.84
    note_b = head_b + 0.32
    key_b = note_b + 0.20 * (fault_note.count("\n") + 1) + 0.24
    top_b, ph_b = key_b + 0.56, 1.85
    W = style.WIDTH
    H = top_b + ph_b + 0.52 + 0.146 * (foot.count("\n") + 1) + 0.20

    fig = style.figure(H, theme)
    assert abs(style.header(fig, theme, title, subtitle) - used) < 1e-9

    # ================================================= A: healthy flights
    section(fig, theme, left, head_a,
            "Healthy flights — the same %d flights, started two ways"
            % len(seeds))
    legend(fig, theme, [
        ("started from the true first state, as every other experiment",
         "dot", theme.orange),
        ("started from zeros, heading 0: the earlier readout", "dot", tint),
    ], key_a, left=left, gap=0.34)

    pw_a = W - left - right
    ax = fig.add_axes([left / W, (H - top_a - ph_a) / H, pw_a / W,
                       ph_a / H])
    style.axes_style(ax, theme)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ymax = math.ceil(zero["peak"].max())
    ax.set_ylim(0, ymax + 0.15)
    ax.set_yticks(range(0, ymax + 1))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["%s\nflight %d" % (signed(heads[s]), s)
                        for s in order], linespacing=1.45)
    ax.set_xlabel("each flight's true starting heading, ordered by its "
                  "distance from 0° — where a zero start puts it",
                  fontsize=8.5, color=theme.muted, labelpad=6)
    ax.set_ylabel("highest health entry", fontsize=8.5, color=theme.muted,
                  labelpad=6)

    lift = (0, style.px(3))                   # clear the rule's own line
    ax.axhline(QUIET, color=theme.ink2, linewidth=style.px(1), zorder=1)
    ax.annotate("no flight started from the truth passes %g" % QUIET,
                (-0.44, QUIET), xytext=lift, textcoords="offset points",
                ha="left", va="bottom", fontsize=7.5, color=theme.ink2)
    ax.axhline(ROBOT_ALERT, color=theme.muted, linewidth=style.px(1),
               zorder=1)
    ax.annotate("%g, the ground robot's alert level" % ROBOT_ALERT,
                (-0.44, ROBOT_ALERT), xytext=lift,
                textcoords="offset points", ha="left", va="bottom",
                fontsize=7.5, color=theme.muted)

    # Each flight is a short slope: the zero start on the left, the true
    # start on the right, so flights whose two runs agree still show both.
    off = 0.14
    for i, s in enumerate(order):
        z, t = zero.loc[s, "peak"], truth.loc[s, "peak"]
        ax.plot([i - off, i + off], [z, t], color=theme.baseline,
                linewidth=style.px(2), zorder=2)
        filled(ax, [i - off], [z], theme, tint, zorder=5)
        filled(ax, [i + off], [t], theme, theme.orange, zorder=6)
        if s in stuck_seeds:
            device, kind = zero.loc[s, "entry"].split("_")[::-1]
            ax.annotate("%s %s %.2f" % (device, kind, z), (i - off, z),
                        xytext=(style.px(10), 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5,
                        color=theme.ink, annotation_clip=False)
        if s == truth_worst:
            ax.annotate("%.2f" % t, (i + off, t), xytext=(style.px(10), 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.5, color=theme.ink, annotation_clip=False)

    ax.text(-0.44, ymax, "Started at heading 0, the %s flights on the right "
            "begin\n%s° from the truth, and an entry then stays above %g\n"
            "for %s%% of the flight, unbroken."
            % (count, span(near[0], near[1], "%.0f"), QUIET,
               span(stuck_share.min(), stuck_share.max(), "%.0f")),
            ha="left", va="top", fontsize=8.5, color=theme.ink2,
            linespacing=1.4)

    # ================================================= B: faults
    section(fig, theme, left, head_b,
            "Faults — the matching entry %.0f s after a mid-flight onset"
            % after_s)
    fig.text(left / W, 1 - note_b / H, fault_note, ha="left", va="top",
             fontsize=9, color=theme.ink2, linespacing=1.45)
    legend(fig, theme, [
        ("bias entry", "dot", theme.orange),
        ("noise entry, blind to a noise fault by construction (dh/dm = 0)",
         "ring", theme.orange),
        ("a working, settled readout", "diag", theme.ink2),
    ], key_b, left=left, gap=0.34)

    gap_b = 0.92
    pw_b = (W - left - right - 2 * gap_b) / 3
    xlim = (severities[0] - 0.42, severities[-1] + 0.07)
    ylim = (0, severities[-1] + 0.2)
    for c, (device, name) in enumerate(DEVICES):
        x0 = left + c * (pw_b + gap_b)
        ax = fig.add_axes([x0 / W, (H - top_b - ph_b) / H, pw_b / W,
                           ph_b / H])
        style.axes_style(ax, theme)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_yticks(range(0, int(severities[-1]) + 1))
        ax.set_xticks(severities)
        ax.set_xticklabels(["%g" % s for s in severities])
        ax.set_title(name, loc="left", fontsize=9.5, color=theme.ink, pad=8)
        ax.set_xlabel("injected severity", fontsize=8.5, color=theme.muted,
                      labelpad=4)
        if c == 0:
            ax.set_ylabel("entry, %.0f s after onset" % after_s,
                          fontsize=8.5, color=theme.muted, labelpad=6)
        else:
            ax.tick_params(labelleft=False)                  # shared scale

        ax.plot(xlim, xlim, color=theme.ink2, linewidth=style.px(1.2),
                zorder=2, solid_capstyle="butt")
        if c == 0:
            p0 = ax.transData.transform((2, 2))
            p1 = ax.transData.transform((3, 3))
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            ax.text(xlim[0] + 0.06, xlim[0] + 0.20,
                    "a working, settled readout", rotation=angle,
                    rotation_mode="anchor", ha="left", va="bottom",
                    fontsize=7.5, color=theme.ink2)

        ends, starts = [], []
        for kind, width in (("noise", 1.5), ("bias", 2)):
            rows = faulted[(faulted["device"] == device)
                           & (faulted["kind"] == kind)].sort_values(
                               "severity")
            ax.plot(rows["severity"], rows["final"], color=theme.orange,
                    linewidth=style.px(width), zorder=3)
            (filled if kind == "bias" else hollow)(
                ax, rows["severity"], rows["final"], theme, theme.orange)
            first, last = rows.iloc[0], rows.iloc[-1]
            ends.append((last["severity"], last["final"],
                         "%s %.2f" % (kind, last["final"])))
            # A noise entry usually reads the same at both severities, so it
            # gets one label; both ends are labelled whenever they differ.
            if kind == "bias" or ("%.2f" % first["final"]
                                  != "%.2f" % last["final"]):
                starts.append((first["severity"], first["final"],
                               "%.2f" % first["final"]))
        text_labels(ax, ends, theme, dx=12)
        text_labels(ax, starts, theme, dx=12, ha="right")

    style.footnote(fig, theme, foot, bottom=0.14)
    return fig


if __name__ == "__main__":
    style.render("health_readout", draw)

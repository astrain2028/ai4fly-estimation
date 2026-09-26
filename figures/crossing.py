"""
The complementarity result on the ground robot: which arm is best depends on
which moment of the reading the fault moves. A health estimate follows a
shift and cannot see a spread; covariance matching sees a spread and only
partly absorbs a shift; layered carries both and holds up under each.

Reads results/bakeoff.csv and results/bakeoff_layered.csv
(experiments/bakeoff.py): 20-s runs, fault on the left encoder from the
start, seeds 2000-2007. Nothing is recomputed except the medians and
quartiles across seeds and the per-seed head-to-head counts.

Every sentence on the figure is checked against the data when it is drawn.
If the headline's premises stop holding, a warning is printed and a plain
headline that claims nothing is used instead.

    python figures/crossing.py
"""

import sys

import numpy as np
import pandas as pd

import style

ANALYTIC = "analytic + best const R"
ADAPTIVE = "adaptive R (Mehra)"
HEALTH = "health-conditioned"
LAYERED = "layered"
ARMS = [ANALYTIC, ADAPTIVE, HEALTH, LAYERED]      # drawing order, back to front

COLUMNS = [("bias", "Bias — the reading shifts"),
           ("noise_inflation", "Noise — the reading spreads")]

NIS_TARGET = 3            # experiments/common.py NIS_DOF: two encoders, one gyro
MM = 1000.0               # speed error is drawn in mm/s; the csv holds m/s
NIS_TICKS = [1, 2, 3, 5, 10, 20]

HEADLINE = ("Health beats adaptive R on large shifts, loses on every spread; "
            "layered holds up in both")
PLAIN = "Speed error and consistency of four arms under a shift and a spread"


def warn(message):
    print("  crossing: WARNING -- %s" % message, file=sys.stderr)


def load():
    data = pd.concat([style.read("bakeoff.csv"),
                      style.read("bakeoff_layered.csv")], ignore_index=True)
    return data[data["arm"].isin(ARMS)]


def ladder(data, arm, mode, column, scale=1.0):
    """One arm down one severity ladder, healthy first: the median across
    seeds at each severity and the middle-50% band."""
    part = data[(data["arm"] == arm) & data["mode"].isin(["none", mode])]
    g = part.groupby("severity")[column]
    s = np.array(sorted(part["severity"].unique()))
    return (s, g.median().loc[s].values * scale,
            g.quantile(0.25).loc[s].values * scale,
            g.quantile(0.75).loc[s].values * scale)


def med(data, arm, mode, column, scale=1.0):
    """{severity: median across seeds} for one arm on one ladder."""
    s, mid, _, _ = ladder(data, arm, mode, column, scale)
    return dict(zip(s, mid))


def wins(data, arm, rival, mode, severity):
    """Seeds on which ``arm`` has lower speed error than ``rival``."""
    part = data[(data["mode"] == mode) & (data["severity"] == severity)]
    p = part.pivot_table(index="seed", columns="arm", values="speed_rmse")
    return int((p[arm] < p[rival]).sum())


def combined_gap(frame):
    """Largest per-seed relative difference between combined and health on
    any bias condition, for the footnote."""
    b = frame[(frame["mode"] == "bias")
              & frame["arm"].isin(["combined", HEALTH])]
    p = b.pivot_table(index=["severity", "seed"], columns="arm",
                      values="speed_rmse")
    return float((p["combined"] / p[HEALTH] - 1).abs().max())


def ahead_from(data, mode, arms, rival):
    """The fault severities from which every arm in ``arms`` has a lower
    median speed error than ``rival`` at that severity and every larger one."""
    r = med(data, rival, mode, "speed_rmse")
    m = [med(data, a, mode, "speed_rmse") for a in arms]
    faulted = [s for s in sorted(r) if s > 0]
    return [s for s in faulted
            if all(x[t] < r[t] for x in m for t in faulted if t >= s)]


def headline(data):
    """HEADLINE if the data supports every clause of it, else PLAIN."""
    seeds = data["seed"].nunique()
    faulted = sorted(s for s in data["severity"].unique() if s > 0)
    checks = []

    # "Health beats adaptive R on large shifts": on the median at every
    # severity from some point up, not at the smallest, and on most seeds.
    large = ahead_from(data, "bias", [HEALTH], ADAPTIVE)
    checks.append((bool(large) and large[0] > faulted[0],
                   "health does not beat adaptive R on the large biases only"))
    checks.append((all(wins(data, HEALTH, ADAPTIVE, "bias", s) > seeds / 2
                       for s in large),
                   "health beats adaptive R on half the seeds or fewer at a "
                   "large bias"))

    # "loses on every spread": the median, at every noise severity.
    he = med(data, HEALTH, "noise_inflation", "speed_rmse")
    ad = med(data, ADAPTIVE, "noise_inflation", "speed_rmse")
    checks.append((all(he[s] > ad[s] for s in faulted),
                   "health does not lose to adaptive R at every noise "
                   "severity"))

    # "layered holds up in both": within the NIS target throughout, and the
    # smallest rise in speed error from healthy to the largest fault.
    for mode, _ in COLUMNS:
        nis = med(data, LAYERED, mode, "nis")
        checks.append((max(nis.values()) <= NIS_TARGET,
                       "layered's NIS exceeds the target under %s" % mode))
        rise = {a: (lambda m: m[max(m)] - m[0])(med(data, a, mode,
                                                    "speed_rmse"))
                for a in ARMS}
        checks.append((min(rise, key=rise.get) == LAYERED,
                       "layered's error does not rise least under %s" % mode))

    failed = [why for ok, why in checks if not ok]
    for why in failed:
        warn("headline premise fails: " + why)
    return PLAIN if failed else HEADLINE


def mm_gap(x):
    """A gap in mm/s as the note prints it: whole millimetres once it is at
    least one, so it agrees with the difference of the one-decimal end
    labels either way it rounds."""
    x = abs(x)
    return "%.0f" % x if x >= 1 else "%.1f" % x


def notes(data):
    """The one sentence each panel is about, with its numbers read from the
    medians and the per-seed rows. Each claim is checked, and the wording
    falls back to what the data does say if it stops holding."""
    out = {}
    seeds = data["seed"].nunique()

    # Bias, accuracy: the crossing. The learned arms start behind -- the
    # analytic model is exact when healthy -- then pass adaptive R.
    he = med(data, HEALTH, "bias", "speed_rmse", MM)
    la = med(data, LAYERED, "bias", "speed_rmse", MM)
    an = med(data, ANALYTIC, "bias", "speed_rmse", MM)
    top = max(he)
    ahead = ahead_from(data, "bias", [HEALTH, LAYERED], ADAPTIVE)
    learned = sorted({"%.1f" % he[0], "%.1f" % la[0]})
    text = "Healthy: learned arms %s, analytic %.1f." % (
        "–".join(learned), an[0])
    if ahead:
        text += ("\nFrom severity %g on, both beat adaptive R;\nat %g, health "
                 "on %d of %d seeds, layered on %d."
                 % (ahead[0], top, wins(data, HEALTH, ADAPTIVE, "bias", top),
                    seeds, wins(data, LAYERED, ADAPTIVE, "bias", top)))
    else:
        text += "\nThey never both pass adaptive R."
        warn("the learned arms do not both pass adaptive R under bias")
    out["bias", 0] = text

    # Noise, accuracy: covariance matching stays in front throughout.
    ad = med(data, ADAPTIVE, "noise_inflation", "speed_rmse", MM)
    others = {a: med(data, a, "noise_inflation", "speed_rmse", MM)
              for a in (ANALYTIC, HEALTH, LAYERED)}
    top = max(ad)
    faulted = [s for s in sorted(ad) if s > 0]
    lines = []
    lowest = all(ad[s] < o[s] for o in others.values() for s in ad)
    if lowest:
        lines.append("Adaptive R is lowest at every severity.")
    else:
        warn("adaptive R is not lowest at every noise severity")
    beaten = [wins(data, ADAPTIVE, HEALTH, "noise_inflation", s)
              for s in faulted]
    subject = "It" if lowest else "Adaptive R"
    if min(beaten) == seeds:
        lines.append("%s beats health on all %d seeds each time."
                     % (subject, seeds))
    else:
        lines.append("%s beats health on %d to %d of %d seeds."
                     % (subject, min(beaten), max(beaten), seeds))
        warn("adaptive R does not beat health on every seed under noise")
    # From the unrounded medians, in whole mm/s: the end labels round each
    # median to 0.1, and their difference can differ from the true gap in
    # the first decimal.
    g_he = others[HEALTH][top] - ad[top]
    g_la = others[LAYERED][top] - ad[top]
    if g_he > 0 and g_la > 0:
        lines.append("At %g, health trails it by %s mm/s, layered by %s."
                     % (top, mm_gap(g_he), mm_gap(g_la)))
    else:
        lines.append("At %g, health is %s mm/s %s adaptive R, layered %s %s."
                     % (top, mm_gap(g_he), "above" if g_he > 0 else "below",
                        mm_gap(g_la), "above" if g_la > 0 else "below"))
    out["noise_inflation", 0] = "\n".join(lines)

    # Bias, calibration: the health arms follow the shift and stay cautious.
    he = med(data, HEALTH, "bias", "nis")
    la = med(data, LAYERED, "bias", "nis")
    ad = med(data, ADAPTIVE, "bias", "nis")
    under = max(max(he.values()), max(la.values())) < NIS_TARGET
    over = ad[max(ad)] > NIS_TARGET
    if under and over:
        out["bias", 1] = ("Health and layered stay under the\ntarget; "
                          "adaptive R ends above it.")
    else:
        warn("bias NIS note dropped: health/layered under target %s, "
             "adaptive above %s" % (under, over))

    # Noise, calibration: health is blind to a spread.
    he = med(data, HEALTH, "noise_inflation", "nis")
    la = med(data, LAYERED, "noise_inflation", "nis")
    if he[max(he)] > NIS_TARGET and max(la.values()) < NIS_TARGET:
        out["noise_inflation", 1] = ("Health cannot see a spread, so it turns"
                                     "\noverconfident; layered stays under %d."
                                     % NIS_TARGET)
    else:
        warn("noise NIS note dropped: health not above target or layered "
             "not under it")
    return out


def hidden_until(ax, lines, tol=2.0):
    """Index of the last severity up to which the health and layered lines
    and bands lie within ``tol`` displayed pixels of each other -- where the
    aqua line, drawn last, covers the orange. 0 if they part at once."""
    fig = ax.figure
    scale = fig.dpi * fig.get_figwidth() / 900.0
    s = lines[HEALTH][0]
    k = 0
    for i in range(len(s)):
        gaps = [abs(ax.transData.transform((s[i], lines[HEALTH][j][i]))[1]
                    - ax.transData.transform((s[i], lines[LAYERED][j][i]))[1])
                for j in (1, 2, 3)]
        if max(gaps) >= tol * scale:
            break
        k = i
    return k


def draw(theme):
    everything = style.read("bakeoff.csv")
    data = load()
    ladder_x = sorted(data["severity"].unique())
    faulted = [s for s in ladder_x if s > 0]
    top = ladder_x[-1]
    seeds = data["seed"].nunique()
    said = notes(data)
    others = sorted(set(everything["arm"]) - set(ARMS) - {"combined"})

    W, H = style.WIDTH, 7.50
    fig = style.figure(H, theme)
    used = style.header(
        fig, theme, headline(data),
        "One left-encoder fault, present for the whole run, at severity %s "
        "or %g (multiples of the healthy noise), beside healthy runs.\nTop: "
        "speed error, lower is better. Bottom: NIS, the filter's "
        "self-consistency check; above the target of %d it is overconfident, "
        "below it cautious."
        % (", ".join("%g" % s for s in faulted[:-1]), faulted[-1],
           NIS_TARGET))
    style.legend_row(fig, theme, [
        (style.SHORT[ANALYTIC] + ", constant R",
         style.arm_color(theme, ANALYTIC)),
        (style.SHORT[ADAPTIVE] + ", covariance matching",
         style.arm_color(theme, ADAPTIVE)),
        (style.SHORT[HEALTH] + ", sensor health states",
         style.arm_color(theme, HEALTH)),
        (LAYERED + ", health + covariance matching",
         style.arm_color(theme, LAYERED)),
    ], used + 0.14, gap=0.22)

    left, right, gap = 0.72, 1.12, 1.06
    pw = (W - left - right - gap) / 2
    ph = 1.78
    heads = used + 0.66
    top0 = heads + 0.46
    row_gap = 0.66

    def axes(row, col):
        x = left + col * (pw + gap)
        y = H - (top0 + row * (ph + row_gap) + ph)
        return fig.add_axes([x / W, y / H, pw / W, ph / H])

    rows = [("Speed error, RMSE", "mm/s", "speed_rmse", MM),
            ("Consistency, mean NIS (log scale)", "NIS", "nis", 1.0)]
    rmse_top = max(ladder(data, a, m, "speed_rmse", MM)[3].max()
                   for a in ARMS for m, _ in COLUMNS)
    nis_top = max(ladder(data, a, m, "nis")[3].max()
                  for a in ARMS for m, _ in COLUMNS)

    for c, (mode, heading) in enumerate(COLUMNS):
        fig.text((left + c * (pw + gap)) / W, 1 - heads / H, heading,
                 ha="left", va="bottom", fontsize=11.5,
                 fontweight="semibold", color=theme.ink)

        for r, (title, unit, column, scale) in enumerate(rows):
            ax = axes(r, c)
            style.axes_style(ax, theme)
            ax.set_xlim(-0.06, top)
            ax.set_xticks(ladder_x)
            ax.set_xticklabels(["healthy" if s == 0 else "%g" % s
                                for s in ladder_x])
            ax.set_title(title, loc="left", fontsize=9.5, color=theme.ink,
                         pad=6)
            if r == 0:
                ax.set_ylim(0, np.ceil(rmse_top / 5) * 5)
                ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, 10))
                ax.tick_params(labelbottom=False)
            else:
                ax.set_yscale("log")
                ax.set_ylim(1, nis_top * 1.25)
                ax.set_yticks(NIS_TICKS)
                ax.set_yticklabels(["%d" % t for t in NIS_TICKS])
                ax.minorticks_off()
                ax.set_xlabel("fault severity", fontsize=8.5,
                              color=theme.muted, labelpad=4)
                ax.axhline(NIS_TARGET, color=theme.ink2,
                           linewidth=style.px(1), zorder=1)
                # Below the rule, where the healthy lines leave room: the
                # fixed arms sit on the target at severity 0.
                ax.annotate("target %d" % NIS_TARGET, (0.06, NIS_TARGET),
                            xytext=(0, -style.px(4)),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=7.5, color=theme.ink2)
            if c == 0:
                ax.set_ylabel(unit, fontsize=8.5, color=theme.muted,
                              labelpad=6)
            else:
                ax.tick_params(labelleft=False)     # shared scale per row

            lines = {arm: ladder(data, arm, mode, column, scale)
                     for arm in ARMS}
            # On speed error, layered runs on top of health from healthy
            # until the fault grows: the orange line is hidden and the two
            # bands would mix into a tint that is no arm's. Draw health's band
            # only where the two part, and say where it is hidden. On the log
            # NIS row they part within the first step, and orange shows.
            hid = hidden_until(ax, lines) if r == 0 else 0

            ends = []
            for arm in ARMS:
                colour = style.arm_color(theme, arm)
                context = arm == ANALYTIC
                s, mid, lo, hi = lines[arm]
                k = hid if arm == HEALTH else 0
                ax.fill_between(s[k:], lo[k:], hi[k:], color=colour,
                                alpha=0.10 if context else 0.14,
                                linewidth=0, zorder=2)
                ax.plot(s, mid, color=colour, zorder=3,
                        linewidth=style.px(1.6 if context else 2.0))
                ends.append((s[-1], mid[-1],
                             "%s %.1f" % (style.SHORT[arm], mid[-1]), colour))
            style.end_labels(ax, ends, theme)

            if hid > 0:
                # Below every line: at the left the lowest arms sit near
                # 5 mm/s and the axis starts at 0.
                ax.text(0.06, ax.get_ylim()[1] * 0.08,
                        "Up to %g, health is hidden under layered."
                        % lines[HEALTH][0][hid],
                        ha="left", va="center", fontsize=7.5,
                        color=theme.ink2)

            note = said.get((mode, r))
            if note:
                # Upper left is empty in every panel: the faults push the
                # lines up and to the right.
                y = (ax.get_ylim()[1] * 0.96 if r == 0
                     else ax.get_ylim()[1] * 0.82)
                ax.text(0.06, y, note, ha="left", va="top", fontsize=8,
                        color=theme.ink2, linespacing=1.4)

    style.footnote(
        fig, theme,
        "Ground-robot simulation: fault on the left encoder from the start of "
        "each 20-s run. Lines are the median of %d seeds; bands, the middle "
        "50%%. Healthy, the analytic\nmodel is exact (the simulator draws "
        "readings from its equations), so the learned arms' higher error there "
        "is the cost of a learned model. Combined matches health\non every bias "
        "condition (per-seed speed error within %.0f%%) and is superseded by "
        "layered; it and the %d other arms are in the csv. Source: "
        "results/bakeoff.csv, results/bakeoff_layered.csv."
        % (seeds, np.ceil(100 * combined_gap(everything)), len(others)),
        bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("crossing", draw)

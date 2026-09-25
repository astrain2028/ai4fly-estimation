"""
The mechanism, watched: a fault arrives mid-run and each half of layered
responds only to the kind it can see.

Reads results/trace.csv (experiments/trace.py). Nothing is recomputed except
the rolling error and the medians across seeds.

    python figures/trace.py
"""

import numpy as np

import style

ONSET = 8.0
SEVERITY = 3.0
HEALTHY_SIGMA = 0.1805            # left encoder, robot/faults.py REFERENCE
INJECTED_VAR = (SEVERITY * HEALTHY_SIGMA) ** 2
ROLL = 20                         # samples at 10 Hz: a two-second window
LATE = 12.0                       # seconds at the end scored in the footnote

COLUMNS = [("bias", "Bias — the reading shifts"),
           ("noise_inflation", "Noise — the reading spreads")]


def spread(frame, column):
    """Median across seeds at each time, and the middle-50% band."""
    table = frame.pivot(index="t", columns="seed", values=column)
    return (table.index.values, table.median(axis=1).values,
            table.quantile(0.25, axis=1).values,
            table.quantile(0.75, axis=1).values)


def rolling_rms(frame):
    """Per-seed rolling RMS of speed error, then median across seeds."""
    table = frame.pivot(index="t", columns="seed", values="speed_error")
    rms = (table ** 2).rolling(ROLL, min_periods=1).mean() ** 0.5
    return (table.index.values, rms.median(axis=1).values,
            rms.quantile(0.25, axis=1).values, rms.quantile(0.75, axis=1).values)


def late_rms(frame, t_end):
    """Each seed's RMS speed error over the last LATE seconds."""
    part = frame[frame["t"] >= t_end - LATE]
    return part.groupby("seed")["speed_error"].apply(
        lambda e: float(np.sqrt(np.mean(e ** 2))))


def line(ax, t, mid, lo, hi, colour, band=0.14, width=2.0):
    ax.fill_between(t, lo, hi, color=colour, alpha=band, linewidth=0)
    ax.plot(t, mid, color=colour, linewidth=style.px(width))


def draw(theme):
    data = style.read("trace.csv")
    layered = data[data["arm"] == "layered"]
    analytic = data[data["arm"] == "analytic + best const R"]
    t_end = data["t"].max()
    noise_entry = style.mix(theme.orange, theme.surface, 0.5)

    W, H = style.WIDTH, 8.57
    fig = style.figure(H, theme)
    used = style.header(
        fig, theme,
        "Each fault is caught by the mechanism that can see it",
        "Left encoder, healthy until t = 8 s, then severity 3. A bias moves "
        "the reading, so the health estimate climbs\nto meet it. Extra noise "
        "leaves the mean where it was: health cannot see it, and the "
        "residual takes it instead.")
    style.legend_row(fig, theme, [
        ("health, bias entry", theme.orange),
        ("health, noise entry", noise_entry),
        ("residual R_unm", theme.blue),
        ("layered", theme.aqua),
        ("analytic, constant R", theme.muted),
    ], used + 0.14, gap=0.26)

    left, right, gap = 0.78, 1.22, 1.02
    pw = (W - left - right - gap) / 2
    ph = 1.36
    heads = used + 0.62
    top0 = heads + 0.44
    row_gap = 0.64

    def axes(row, col):
        x = left + col * (pw + gap)
        y = H - (top0 + row * (ph + row_gap) + ph)
        return fig.add_axes([x / W, y / H, pw / W, ph / H])

    rows = [("Health estimate", "severity", (0, 3.4)),
            ("Unmodelled variance, R_unm", "(rad/s)²", (0, 0.34)),
            ("Speed error, 2-s rolling RMS", "m/s", (0, 0.045))]

    for c, (mode, heading) in enumerate(COLUMNS):
        L = layered[layered["scenario"] == mode]
        A = analytic[analytic["scenario"] == mode]
        fig.text((left + c * (pw + gap)) / W, 1 - heads / H, heading,
                 ha="left", va="bottom", fontsize=11.5,
                 fontweight="semibold", color=theme.ink)

        for r, (title, unit, ylim) in enumerate(rows):
            ax = axes(r, c)
            style.axes_style(ax, theme)
            ax.set_xlim(0, t_end)
            ax.set_ylim(*ylim)
            ax.axvline(ONSET, color=theme.baseline, linewidth=style.px(1),
                       zorder=1)
            ax.set_title(title, loc="left", fontsize=9.5, color=theme.ink,
                         pad=6)
            if c == 0:
                ax.set_ylabel(unit, fontsize=8.5, color=theme.muted,
                              labelpad=6)
            else:
                ax.tick_params(labelleft=False)     # shared scale per row
            if r < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time, s", fontsize=8.5, color=theme.muted,
                              labelpad=4)

            if r == 0:
                # The injected fault in the entry's own units: under bias the
                # bias entry should reach 3, under noise the noise entry.
                t = np.array([0, ONSET, ONSET, t_end])
                ax.plot(t, [0, 0, SEVERITY, SEVERITY], color=theme.ink2,
                        linewidth=style.px(1), zorder=1)
                ax.text(ONSET + 0.6, SEVERITY, "injected severity",
                        ha="left", va="bottom", fontsize=7.5,
                        color=theme.ink2)
                tn, mn, lo, hi = spread(L, "noise_left")
                line(ax, tn, mn, lo, hi, noise_entry, band=0.10)
                tb, mb, lo, hi = spread(L, "bias_left")
                line(ax, tb, mb, lo, hi, theme.orange)
                style.end_labels(ax, [
                    (tb[-1], mb[-1], "bias %.2f" % mb[-1], theme.orange),
                    (tn[-1], mn[-1], "noise %.2f" % mn[-1], noise_entry)],
                    theme)

            elif r == 1:
                tt, mid, lo, hi = spread(L, "r_unm_left")
                line(ax, tt, mid, lo, hi, theme.blue)
                style.end_labels(ax, [(tt[-1], mid[-1], "%.3f" % mid[-1],
                                       theme.blue)], theme)
                if mode == "noise_inflation":
                    ax.axhline(INJECTED_VAR, color=theme.ink2,
                               linewidth=style.px(1), zorder=1)
                    ax.text(ONSET + 0.6, INJECTED_VAR,
                            "injected, (3σ)² = %.3f" % INJECTED_VAR,
                            ha="left", va="bottom", fontsize=7.5,
                            color=theme.ink2)
                    late = L[L["t"] >= t_end - 10]
                    share = late["r_unm_left"].median() / INJECTED_VAR
                    note = ("recovers %.0f%% of the injected\nvariance, "
                            "with no retraining" % (100 * share))
                else:
                    note = ("stays near zero: health\nalready explains "
                            "the shift")
                ax.text(t_end, 0.075, note, ha="right", va="bottom",
                        fontsize=8, color=theme.ink2, linespacing=1.35)

            else:
                ta, ma, lo, hi = rolling_rms(A)
                line(ax, ta, ma, lo, hi, theme.muted, band=0.10, width=1.6)
                tl, ml, lo, hi = rolling_rms(L)
                line(ax, tl, ml, lo, hi, theme.aqua)
                style.end_labels(ax, [
                    (ta[-1], ma[-1], "analytic %.3f" % ma[-1], theme.muted),
                    (tl[-1], ml[-1], "layered %.3f" % ml[-1], theme.aqua)],
                    theme)

    # The bias column's accuracy, stated per seed because the median hides
    # the spread that matters: most seeds gain, one loses badly.
    bias_L = late_rms(layered[layered["scenario"] == "bias"], t_end)
    bias_A = late_rms(analytic[analytic["scenario"] == "bias"], t_end)
    wins = int((bias_L < bias_A).sum())
    worst = bias_L.idxmax()
    pooled_L = float(np.sqrt((bias_L ** 2).mean()))
    pooled_A = float(np.sqrt((bias_A ** 2).mean()))
    style.footnote(
        fig, theme,
        "Ground-robot simulation. Lines are the median of %d seeds; bands, "
        "the middle 50%%. Under the mid-run bias, layered beats the analytic "
        "model on\n%d of %d seeds over the last %.0f s, but seed %d, whose "
        "health estimate stalls near 1, loses badly (%.3f against %.3f m/s), "
        "and the pooled error is\n%.3f against %.3f. A fault present from "
        "the start (experiments/bakeoff.py) does not show this. Source: "
        "results/trace.csv."
        % (len(bias_L), wins, len(bias_L), LATE, worst, bias_L[worst],
           bias_A[worst], pooled_L, pooled_A),
        bottom=0.12)
    return fig


if __name__ == "__main__":
    style.render("trace", draw)

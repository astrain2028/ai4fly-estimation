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
NEUTRAL = 3         # the diverging stop for "within 5% of analytic"
TIE = 0.01          # cells within 1% of the column's best are ringed too
FAR = 0.5           # "by far": under half of every other arm's error
EDGE = 0.02         # a lower-rung win by less than this is an edge

CHANNEL = "left_encoder"        # where experiments/moments.py puts the fault


# ---------------------------------------------------------------- data

def verdict(err, predicted):
    """Which single-mechanism arm has the lower error, judged as moments.py
    judges it: idxmin over its rows, where adaptive R comes first, so a tie
    goes to adaptive R."""
    if not predicted or HEALTH not in err or ADAPTIVE not in err:
        return None
    return HEALTH if err[HEALTH] < err[ADAPTIVE] else ADAPTIVE


def scoreboard(data):
    """Per mode: every arm's speed error on every rung of the ladder, and the
    verdict on each rung. The grid shows the top rung."""
    arms = [a for a in ORDER if a in set(data["arm"])]
    columns = []
    for mode in MODES:
        rows = data[data["mode"] == mode]
        if rows.empty:
            continue
        rungs = sorted(float(s) for s in rows["severity"].unique())
        ladder = []
        for severity in rungs:
            at = rows[rows["severity"] == severity]
            ladder.append({a: float(at.loc[at["arm"] == a, "speed_rmse"]
                                    .iloc[0])
                           for a in arms if (at["arm"] == a).any()})
        err = ladder[-1]
        moment = rows["moment"].iloc[0]
        best = min(err.values())
        predicted = EXPECTED.get(moment)
        calls = [verdict(e, predicted) for e in ladder]
        columns.append({
            "mode": mode, "moment": moment,
            "trained": bool(rows["trained"].iloc[0]),
            "rungs": rungs, "severity": rungs[-1],
            "ladder": ladder, "err": err,
            "ratio": {a: e / err[ANALYTIC] for a, e in err.items()},
            "best": [a for a in arms if a in err and err[a] <= best * (1 + TIE)],
            "predicted": predicted,
            "calls": calls, "actual": calls[-1]})
    return arms, columns


def join(words):
    return words[0] if len(words) == 1 else \
        ", ".join(words[:-1]) + " and " + words[-1]


def tested(columns):
    return [c for c in columns if c["actual"]]


def headline(columns):
    """The finding in words, assembled from the verdicts rather than typed."""
    judged = tested(columns)
    hits = [c for c in judged if c["actual"] == c["predicted"]]
    misses = [c for c in judged if c not in hits]
    held_hit = [NAME[c["mode"]] for c in hits if not c["trained"]]
    held_miss = [NAME[c["mode"]] for c in misses if not c["trained"]]
    trained_miss = [NAME[c["mode"]] for c in misses if c["trained"]]
    # The grid is the top rung. If a lower rung judges differently, say so.
    steady = all(len(set(c["calls"])) == 1 for c in judged)
    where = "" if steady else " at full severity"
    if not misses:
        text = "Moment order got all %d calls right%s" % (len(judged), where)
        return text + (", held-out %s included" % join(held_hit)
                       if held_hit else "")
    parts = []
    if held_hit:
        parts.append("held-out %s yes" % join(held_hit))
    if held_miss:
        parts.append("held-out %s no" % join(held_miss))
    if trained_miss:
        parts.append("trained-on %s no" % join(trained_miss))
    return ("Moment order got %d of %d calls right%s — %s"
            % (len(hits), len(judged), where, ", ".join(parts)))


def notes(columns):
    """What the grid adds beyond the verdict, as (lead, sentence) pairs built
    from the numbers rather than typed."""
    out = [("Where it missed", missed(c)) for c in tested(columns)
           if c["actual"] != c["predicted"]]
    neither = [c for c in columns if c["moment"] == "neither"]
    if neither:
        out.append(("No call made", no_call(neither)))
    price = health_price(columns)
    if price:
        out.append(("Health's price", price))
    return out


def effects(pairs, noun="the error"):
    """[(name, ratio)] -> ['health cut the error by 11%', 'adaptive R by 41%'].
    The verb is spelt out only where it changes; join() makes the sentence."""
    out, last = [], None
    for name, ratio in pairs:
        change = percent(ratio)
        verb = ("left" if change == "0%" else
                "cut" if ratio < 1 else "raised")
        if verb == "left":
            out.append("%s left %s unchanged" % (name, noun))
        elif verb == last:
            out.append("%s by %s" % (name, change[1:]))
        else:
            out.append("%s %s %s by %s" % (name, verb, noun, change[1:]))
        last = verb
    return [s.strip() for s in out]


def missed(c):
    """A miss: the adjudicated pair first, then the arms carrying both
    mechanisms, then whether the top rung flatters the best of them."""
    r, name = c["ratio"], NAME[c["mode"]]
    pair = [c["predicted"], c["actual"]]
    text = ("On %s, %s, so %s won."
            % (name, join(effects([(style.SHORT[a], r[a]) for a in pair])),
               style.SHORT[c["actual"]]))
    both = sorted((a for a in ("layered", "combined") if a in r), key=r.get)
    if not both:
        return text
    lead = both[0]
    others = [e for a, e in c["err"].items() if a != lead]
    far = lead in c["best"] and all(c["err"][lead] < FAR * e for e in others)
    said = effects([(style.SHORT[a], r[a]) for a in both], noun="it")
    if far:
        said[0] += ", the column's lowest by far" + ("," if len(said) > 1
                                                     else "")
    text += " Of the %s with both mechanisms, %s." % (
        "arm" if len(both) == 1 else "%s arms" % NUMBER.get(len(both),
                                                            len(both)),
        join(said))
    # A ladder that is not monotone makes the top rung a lucky draw or an
    # unlucky one; say so, with the rung below for comparison.
    steps = [rung[lead] for rung in c["ladder"] if lead in rung]
    if len(steps) > 1 and any(b < a for a, b in zip(steps, steps[1:])):
        below = c["ladder"][-2]
        text += (" %s's own error is not monotone up the ladder (%s m/s):"
                 " at severity %g it %s, not %s."
                 % (style.SHORT[lead].capitalize(),
                    ", ".join("%#.2g" % s for s in steps), c["rungs"][-2],
                    effects([("", below[lead] / below[ANALYTIC])])[0],
                    percent(r[lead])[1:]))
    return text


NUMBER = {2: "two", 3: "three", 4: "four"}


def no_call(neither):
    said = []
    for c in neither:
        r = c["ratio"]
        if c["best"] == [ANALYTIC]:
            near = min((a for a in r if a != ANALYTIC), key=r.get)
            said.append("nothing beats analytic on %s (closest: %s, %s)"
                        % (NAME[c["mode"]], style.SHORT[near], percent(r[near])))
        else:
            said.append("%s is best on %s (%s)" % (
                join([style.SHORT[a] for a in c["best"]]), NAME[c["mode"]],
                join([percent(r[a]) for a in c["best"]])))
    if len(neither) == 1:
        return said[0][0].upper() + said[0][1:] + "."
    winners = {tuple(c["best"]) for c in neither}
    everyone = "both" if len(neither) == 2 else "all %d" % len(neither)
    opener = ("No one arm is best on %s" % everyone if len(winners) > 1
              else "One arm is best on %s" % everyone)
    return "%s: %s." % (opener, "; ".join(said))


def health_price(columns):
    """The healthy-run cost of the health states, then the lower rungs, where
    the verdict can change."""
    healthy = next((c for c in columns if c["mode"] == "none"), None)
    paying = [a for a in HEALTH_BASED if healthy and a in healthy["ratio"]]
    if not paying:
        return None
    cost = sorted(healthy["ratio"][a] for a in paying)
    span = (percent(cost[0]) if percent(cost[0]) == percent(cost[-1])
            else "%s to %s" % (percent(cost[0]), percent(cost[-1])))
    text = ("The health-based arms pay %s on a healthy run for their extra "
            "states." % span)

    judged = tested(columns)
    if not judged:
        return text
    depth = min(len(c["calls"]) for c in judged)
    top_hits = sum(c["actual"] == c["predicted"] for c in judged)
    for i in range(depth - 2, -1, -1):          # from the rung below the top
        sevs = sorted({c["rungs"][i] for c in judged})
        at = "severity %s" % "/".join("%g" % s for s in sevs)
        hits = sum(c["calls"][i] == c["predicted"] for c in judged)
        flips = [c for c in judged if c["calls"][i] != c["actual"]]
        if not flips:
            text += (" At %s the calls come out as at the top, %d of %d."
                     % (at, hits, len(judged)))
            continue
        clauses = []
        for winner in dict.fromkeys(c["calls"][i] for c in flips):
            won = [c for c in flips if c["calls"][i] == winner]
            loser = HEALTH if winner == ADAPTIVE else ADAPTIVE
            margins = [c["ladder"][i][loser] / c["ladder"][i][winner] - 1
                       for c in won]
            modes = join([NAME[c["mode"]] for c in won])
            if max(margins) < EDGE:
                clause = "%s edges %s on %s by %s" % (
                    style.SHORT[winner], style.SHORT[loser], modes,
                    join(["%.1f%%" % (100 * m) for m in margins]))
            else:
                clause = "%s beats %s on %s" % (
                    style.SHORT[winner], style.SHORT[loser], modes)
            # Where adaptive R wins, is its error with the fault still under
            # health's with none? Then the healthy-run price alone decided it.
            if (winner == ADAPTIVE and HEALTH in healthy["err"]
                    and all(c["ladder"][i][ADAPTIVE] < healthy["err"][HEALTH]
                            for c in won)):
                clause += (", its error with the fault still below health's "
                           "with none")
            clauses.append(clause)
        text += " At %s the calls come out %d of %d%s: %s." % (
            at, hits, len(judged),
            ", but not the same ones" if hits == top_hits else "",
            "; ".join(clauses))
    return text


def ladder_words(columns):
    """What the top rung of each ladder means, found by running
    robot/faults.py's own fault functions on a probe signal."""
    spec = importlib.util.spec_from_file_location(
        "faults", style.ROOT / "robot" / "faults.py")
    faults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(faults)
    spread = faults.REFERENCE[CHANNEL]
    top = {c["mode"]: c["severity"] for c in columns}
    n = 101
    zeros, ones, ramp = np.zeros(n), np.ones(n), np.arange(float(n))
    bits = []
    # bias adds severity x spread; noise draws with that as its sigma.
    amount = [m for m in ("bias", "noise_inflation") if m in top]
    if amount:
        shift = faults.bias(zeros, top[amount[0]], spread, None, None)[0]
        same = len({top[m] for m in amount}) == 1
        bits.append("%s at %g× the encoder's healthy spread (%.2f "
                    "rad/s)" % (join([NAME[m] for m in amount]) if same
                                else NAME[amount[0]], shift / spread, spread))
    if "drift" in top:
        grown = faults.drift(zeros, top["drift"], 1.0, None, None)
        bits.append("drift growing from %g to %g× by the end of the run"
                    % (grown[0], grown[-1]))
    if "scale_error" in top:
        gain = faults.scale_error(ones, top["scale_error"], spread,
                                  None, None)[0]
        bits.append("scale error reading %.0f%% %s, so its error grows with "
                    "speed"
                    % (100 * abs(gain - 1), "high" if gain > 1 else "low"))
    if "stuck" in top:
        held = faults.stuck(ramp, top["stuck"], spread, None, None)
        bits.append("stuck frozen for the last %.0f%% of the run"
                    % (100 * np.mean(held != ramp)))
    if "dropout" in top:
        bits.append("dropout losing each reading with probability %g"
                    % top["dropout"])
    return ("Top rungs: " + "; ".join(bits) + ".") if bits else ""


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


def outline(theme, stop):
    """Cell outline: none, except a hairline on the neutral stop, which on
    the light surface is too close to the page (about 1.07:1) to read as a
    cell by its fill alone."""
    if stop == NEUTRAL:
        return dict(edgecolor=theme.grid, linewidth=style.px(1))
    return dict(edgecolor="none")


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

    foot = wrap("Ground-robot simulation, fault on the left encoder. %s "
                "Cells are seed means from experiments/moments.py with no "
                "per-seed spread kept, so arms a few percent apart are not "
                "resolved. Held-out modes never appeared in "
                "training. Source: results/moments.csv."
                % ladder_words(columns), right - left, 7.5)

    # Vertical plan in inches from the top, so the height follows the rows.
    groups_y = 1.86
    names_y = groups_y + 0.50
    tags_y = names_y + 0.22
    grid_y = tags_y + 0.24
    rule_y = grid_y + ref_h + 0.075   # midway between the two rows' rings
    ratio_y = grid_y + ref_h + 0.15
    grid_end = ratio_y + len(ratio_arms) * row_pitch - (row_pitch - cell_h)
    pred_y = grid_end + 0.34
    out_y = pred_y + 0.42
    notes_y = out_y + 0.52
    notes_h = sum(len(lines) * note_lead + 0.10 for _, lines in wrapped)
    H = notes_y + notes_h + 0.155 * len(foot) + 0.12

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
            facecolor=colour, **outline(theme, i))
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
    ax.plot([grid_x, right], [rule_y] * 2,
            color=theme.baseline, linewidth=style.px(1), solid_capstyle="butt")

    # The ratio rows.
    for r, arm in enumerate(ratio_arms):
        y = ratio_y + r * row_pitch
        row_label(y + cell_h / 2, style.SHORT[arm], ROLE[arm], arm)
        for c, x in zip(columns, xs):
            if arm not in c["ratio"]:
                continue
            ratio = c["ratio"][arm]
            stop = int(np.searchsorted(EDGES, ratio))
            fill = theme.diverging[stop]
            box(ax, x, y, cell_w, cell_h, 0.04, facecolor=fill, zorder=2,
                **outline(theme, stop))
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
                                          theme.surface, theme.wash,
                                          theme.grid, theme.baseline])

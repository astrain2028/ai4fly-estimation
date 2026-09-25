"""
One visual language for every figure in the repository.

Each figure module defines ``draw(theme)`` returning a matplotlib figure and
calls ``render(name, draw)``, which draws it twice -- once on the light
surface, once on the dark -- and writes both to figures/img/. The README
embeds them with a ``<picture>`` element, so GitHub shows whichever matches
the reader's theme.

Colour
------

Colour means the same thing in every figure, and it follows the arm, never
its rank:

    analytic + best const R    gray      the baseline, context rather than subject
    adaptive R (Mehra)         blue      reads innovation magnitude
    health-conditioned         orange    reads innovation direction
    layered                    aqua      both, added
    combined                   yellow    both, multiplied

Blue and orange are also the two mechanisms wherever a figure shows a
mechanism rather than an arm -- the covariance-matching residual is blue,
a health estimate is orange -- because that is what those arms are.

The four hues are the first four slots of a palette validated for colour
vision deficiency, in that order, in both modes (worst adjacent protan
delta-E 9.1 light, 8.4 dark; normal-vision 22.9 and 19.8). Aqua and yellow sit
below 3:1 on the light surface, so every figure labels its lines directly
rather than relying on the legend alone.

Size
----

Figures are 9 inches wide and meant to be seen at about 900 pixels, which is
the width of a README column. At that scale one displayed pixel is 0.72
points, and every length below is written in displayed pixels through
``px`` so the specs read as what they look like: 2 px lines, 8 px markers
with a 2 px surface ring, 1 px hairline grids.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT = HERE / "img"

WIDTH = 9.0            # inches
DPI = 220


def px(n):
    """Displayed pixels to points, for a 9-inch figure shown 900 px wide."""
    return 0.72 * n


class Theme:
    def __init__(self, name, **roles):
        self.name = name
        self.__dict__.update(roles)


LIGHT = Theme(
    "light",
    surface="#fcfcfb", page="#f9f9f7",
    ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7", wash="#f0efec",
    blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
    context="#c3c2b7",
    good="#006300", critical="#d03b3b",
    diverging=["#104281", "#2a78d6", "#86b6ef", "#f0efec",
               "#f2a3a2", "#e34948", "#a32826"],
)

DARK = Theme(
    "dark",
    surface="#1a1a19", page="#0d0d0d",
    ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", baseline="#383835", wash="#383835",
    blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
    context="#52514e",
    good="#0ca30c", critical="#e66767",
    diverging=["#86b6ef", "#3987e5", "#1c5cab", "#383835",
               "#a83a3a", "#e66767", "#f2a3a2"],
)

THEMES = [LIGHT, DARK]

# The arms by the label the csv files use.
ARM_ROLE = {
    "analytic + best const R": "muted",
    "adaptive R (Mehra)": "blue",
    "health-conditioned": "orange",
    "layered": "aqua",
    "combined": "yellow",
}

SHORT = {
    "analytic + best const R": "analytic",
    "adaptive R (Mehra)": "adaptive R",
    "health-conditioned": "health",
    "layered": "layered",
    "combined": "combined",
}


def arm_color(theme, arm):
    """An arm's colour; anything not named above is context gray."""
    return getattr(theme, ARM_ROLE.get(arm, "context"))


# ---------------------------------------------------------------- setup

FONT = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]


def setup(theme):
    """Global rcParams for one theme. Called by render before each draw."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 9,
        "text.color": theme.ink,
        "axes.facecolor": theme.surface,
        "figure.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "axes.edgecolor": theme.baseline,
        "axes.labelcolor": theme.ink2,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": theme.ink,
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelcolor": theme.ink2,
        "ytick.labelcolor": theme.ink2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": px(2),
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "mathtext.default": "regular",
    })


def mix(a, b, t):
    """Hex colour ``t`` of the way from ``a`` to ``b``. For a lighter tint of a
    series colour, mix it toward the surface rather than lowering alpha, so
    the legend key and the line are exactly the same colour."""
    a = np.array([int(a[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(b[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = np.round(a + (b - a) * t).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def axes_style(ax, theme, grid="y", baseline=True):
    """Recessive chrome: hairline solid grid, one baseline, no box.

    The axes background is switched off -- the figure is already the surface
    colour -- so labels that run past one panel into its neighbour are not
    painted over by the neighbour's background.
    """
    ax.patch.set_visible(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(baseline)
    ax.spines["bottom"].set_color(theme.baseline)
    ax.spines["bottom"].set_linewidth(px(1))
    ax.tick_params(length=0, pad=5)
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=theme.grid, linewidth=px(1), linestyle="-")


# ---------------------------------------------------------------- figure

def figure(height, theme):
    """A 9-inch-wide figure on the theme's surface."""
    fig = plt.figure(figsize=(WIDTH, height))
    fig.patch.set_facecolor(theme.surface)
    return fig


def header(fig, theme, title, subtitle=None, top=0.28, left=0.28):
    """Left-aligned headline and a one-line explanation beneath it.

    ``top`` and ``left`` are inches from the figure's top-left corner. The
    headline states the finding, not the axes; the subtitle says how to read.
    Returns the inches consumed, for placing whatever comes next.
    """
    h = fig.get_figheight()
    w = fig.get_figwidth()
    fig.text(left / w, 1 - top / h, title, ha="left", va="top",
             fontsize=13.5, fontweight="semibold", color=theme.ink)
    used = top + 0.30
    if subtitle:
        fig.text(left / w, 1 - used / h, subtitle, ha="left", va="top",
                 fontsize=9.5, color=theme.ink2, linespacing=1.45)
        used += 0.20 * (subtitle.count("\n") + 1) + 0.06
    return used


def legend_row(fig, theme, items, y_in, left=0.28, gap=0.30):
    """A horizontal key: short line in the series colour, label in ink.

    ``items`` is a list of (label, colour) or (label, colour, kind) where kind
    is "line" (default), "dot", "band" or "rule". ``y_in`` is inches from the
    top. Text never wears the series colour; the mark beside it carries it.
    """
    h = fig.get_figheight()
    w = fig.get_figwidth()
    x = left
    renderer = fig.canvas.get_renderer()
    for item in items:
        label, colour = item[0], item[1]
        kind = item[2] if len(item) > 2 else "line"
        y = 1 - y_in / h
        if kind == "line":
            fig.add_artist(plt.Line2D([x / w, (x + 0.24) / w], [y, y],
                                      color=colour, linewidth=px(2.5),
                                      solid_capstyle="round",
                                      transform=fig.transFigure))
            x += 0.32
        elif kind == "rule":
            fig.add_artist(plt.Line2D([x / w, (x + 0.24) / w], [y, y],
                                      color=colour, linewidth=px(1.2),
                                      transform=fig.transFigure))
            x += 0.32
        elif kind == "dot":
            fig.add_artist(plt.Line2D([(x + 0.06) / w], [y], marker="o",
                                      markersize=px(9), color=colour,
                                      markeredgecolor=theme.surface,
                                      markeredgewidth=px(2),
                                      transform=fig.transFigure))
            x += 0.18
        elif kind == "band":
            fig.add_artist(plt.Rectangle((x / w, y - 0.055 / h), 0.24 / w,
                                         0.11 / h, color=colour, alpha=1.0,
                                         linewidth=0,
                                         transform=fig.transFigure))
            x += 0.32
        text = fig.text(x / w, y, label, ha="left", va="center",
                        fontsize=9, color=theme.ink2)
        width = text.get_window_extent(renderer).width / fig.dpi
        x += width + gap


def footnote(fig, theme, text, bottom=0.14, left=0.28):
    """Source and method, muted, bottom-left."""
    fig.text(left / fig.get_figwidth(), bottom / fig.get_figheight(), text,
             ha="left", va="bottom", fontsize=7.5, color=theme.muted,
             linespacing=1.4)


def end_label(ax, x, y, text, theme, colour, dx=6, dy=0, dot=True,
              ha="left", size=8.5, weight="normal"):
    """A value or name at the end of a line, with the endpoint marked.

    The dot is 8 px with a 2 px surface ring so it stays legible where lines
    cross; the text is ink, offset ``dx`` displayed pixels to the right.
    """
    if dot:
        ax.plot([x], [y], marker="o", markersize=px(8), color=colour,
                markeredgecolor=theme.surface, markeredgewidth=px(2),
                zorder=5, clip_on=False)
    ax.annotate(text, (x, y), xytext=(px(dx), px(dy)),
                textcoords="offset points", ha=ha, va="center",
                fontsize=size, color=theme.ink, fontweight=weight,
                annotation_clip=False)


def end_labels(ax, items, theme, dx=7, min_gap=13, size=8.5):
    """Several end labels on one axes, pushed apart where they would collide.

    ``items`` is a list of (x, y, text, colour). Each endpoint gets its dot at
    the true value; the text moves vertically only as far as it must to clear
    its neighbours by ``min_gap`` displayed pixels, and a hairline leader
    connects it back when it has moved more than a few pixels.
    """
    fig = ax.figure
    scale = fig.dpi * fig.get_figwidth() / 900.0
    pts = [ax.transData.transform((x, y)) for x, y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: pts[i][1])
    placed = {}
    last = -np.inf
    for i in order:
        y = max(pts[i][1], last + min_gap * scale)
        placed[i] = y
        last = y
    # Centre the adjusted block on the original block so the nudge is shared.
    shift = (np.mean([pts[i][1] for i in order])
             - np.mean([placed[i] for i in order]))
    inv = ax.transData.inverted()
    for i, (x, y, text, colour) in enumerate(items):
        ax.plot([x], [y], marker="o", markersize=px(8), color=colour,
                markeredgecolor=theme.surface, markeredgewidth=px(2),
                zorder=5, clip_on=False)
        tx_px = pts[i][0] + dx * scale
        ty_px = placed[i] + shift
        tx, ty = inv.transform((tx_px, ty_px))
        if abs(ty_px - pts[i][1]) > 4 * scale:
            lx, _ = inv.transform((pts[i][0] + 2 * scale, 0))
            ax.plot([lx, tx], [y, ty], color=theme.baseline,
                    linewidth=px(1), clip_on=False, zorder=4)
            tx, _ = inv.transform((tx_px + 2 * scale, 0))
        ax.text(tx, ty, text, ha="left", va="center", fontsize=size,
                color=theme.ink, clip_on=False)


def dot(ax, x, y, theme, colour, size=8, zorder=5):
    """A marker with its surface ring."""
    ax.plot(x, y, linestyle="none", marker="o", markersize=px(size),
            color=colour, markeredgecolor=theme.surface,
            markeredgewidth=px(2), zorder=zorder)


def band(ax, lo, hi, theme, label=None, label_x=None, axis="y"):
    """A reference band -- the region a calibrated filter should sit in."""
    if axis == "y":
        ax.axhspan(lo, hi, color=theme.wash, linewidth=0, zorder=0)
    else:
        ax.axvspan(lo, hi, color=theme.wash, linewidth=0, zorder=0)
    if label:
        x = ax.get_xlim()[0] if label_x is None else label_x
        ax.text(x, hi, " " + label, ha="left", va="bottom", fontsize=7.5,
                color=theme.muted)


def rule(ax, value, theme, label=None, axis="y", label_pos=None, ha="left"):
    """A thin reference line in ink2 -- a target or an injected value."""
    if axis == "y":
        ax.axhline(value, color=theme.ink2, linewidth=px(1), zorder=1)
        if label:
            x = ax.get_xlim()[0] if label_pos is None else label_pos
            ax.text(x, value, label, ha=ha, va="bottom", fontsize=7.5,
                    color=theme.ink2)
    else:
        ax.axvline(value, color=theme.ink2, linewidth=px(1), zorder=1)
        if label:
            y = ax.get_ylim()[1] if label_pos is None else label_pos
            ax.text(value, y, " " + label, ha=ha, va="top", fontsize=7.5,
                    color=theme.ink2)


def hbar(ax, y, value, colour, theme, thickness=18, radius=4, start=0.0):
    """A horizontal bar with a rounded data end and a square baseline.

    Call after the axes limits are final: the corner radius is set in
    displayed pixels and converted to data units through the current
    transform, so the rounding is circular whatever the aspect.
    """
    fig = ax.figure
    to_px = ax.transData.transform
    x0p, y0p = to_px((start, y))
    x1p, _ = to_px((value, y))
    scale = fig.dpi * fig.get_figwidth() / 900.0    # displayed px to canvas px
    half = thickness / 2 * scale
    r = min(radius * scale, abs(x1p - x0p) / 2, half)
    sign = 1 if x1p >= x0p else -1
    k = 0.5523 * r
    top, bot = y0p + half, y0p - half
    xe = x1p - sign * r
    verts = [(x0p, bot), (xe, bot),
             (xe + sign * k, bot), (x1p, bot + r - k), (x1p, bot + r),
             (x1p, top - r),
             (x1p, top - r + k), (xe + sign * k, top), (xe, top),
             (x0p, top), (x0p, bot)]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    inv = ax.transData.inverted()
    data = inv.transform(np.array(verts))
    patch = PathPatch(MplPath(data, codes), facecolor=colour,
                      edgecolor="none", zorder=3)
    ax.add_patch(patch)
    return patch


# ---------------------------------------------------------------- output

def render(name, draw, quantize=True):
    """Draw ``name`` in both themes and write figures/img/name-{light,dark}.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for theme in THEMES:
        setup(theme)
        fig = draw(theme)
        path = OUT / ("%s-%s.png" % (name, theme.name))
        fig.savefig(path, dpi=DPI, facecolor=theme.surface)
        plt.close(fig)
        if quantize:
            _shrink(path)
        written.append(path)
        print("  wrote %s" % path.relative_to(ROOT))
    return written


def _shrink(path):
    """Palette-quantise a PNG. Flat charts lose nothing visible at 256 colours
    and the file drops to about a third, which matters for a repository that
    carries two copies of every figure."""
    try:
        from PIL import Image
    except ImportError:
        return
    image = Image.open(path).convert("RGB")
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                   dither=Image.Dither.NONE).save(path, optimize=True)


def read(name):
    """A results csv, or exit cleanly if the experiment has not been run."""
    import pandas as pd
    path = RESULTS / name
    if not path.exists():
        print("No %s -- run the experiment that writes it first."
              % path.relative_to(ROOT))
        sys.exit(0)
    return pd.read_csv(path)

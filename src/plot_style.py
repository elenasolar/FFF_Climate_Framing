"""Shared matplotlib/seaborn styling for publication-ready figures.

Usage (from a notebook in notebooks/):

    import sys
    from pathlib import Path
    sys.path.append(str(Path("..").resolve()))

    from src import plot_style
    plot_style.set_style()
"""

import colorsys
import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Figure sizes (inches), matched to common single/double-column journal widths.
FIGSIZE_SINGLE = (3.5, 2.8)   # single-column figure, few categories (e.g. bar/count plots)
FIGSIZE_TALL = (3.5, 4.5)     # single-column, many categories (e.g. horizontal bar lists)
FIGSIZE_WIDE = (7.0, 3.2)     # full-width figure (e.g. time series, side-by-side panels)
FIGSIZE_HERO = (13.0, 6.0)    # presentation-scale figure (e.g. the frame-intensity-over-time plots)

# Font sizes (pt) -- the FINAL on-page size at _REFERENCE_WIDTH (see below), close to but
# slightly smaller than typical body text, so figure text doesn't compete with the prose.
# Any figure narrower or wider than _REFERENCE_WIDTH needs scaled_fonts() (below) to land on
# these same final sizes once actually placed in the report -- these numbers are NOT what a
# HERO-sized figure should look like un-scaled at its own native 13in canvas.
FONT_SIZE_BASE = 9
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 9
FONT_SIZE_TICK = 8
FONT_SIZE_LEGEND = 8
FONT_SIZE_ANNOTATION = 8

# Qualitative palette for distinct categories (content type, account type, ...).
# Sequential palette for ordered/continuous values (e.g. histograms).
PALETTE_QUALITATIVE = sns.color_palette("Paired")
PALETTE_SEQUENTIAL = "mako"

# Separate palette for concept LINE plots (frame intensity over time etc.) -- "Paired"'s
# light/dark pairing puts two similarly-light colors next to each other when cycling through
# more than ~2 categories (e.g. axis A's 1st and 3rd concepts both land on a "light" slot),
# which is hard to tell apart as thin lines even though it reads fine as bars. "colorblind" is
# built for exactly this: distinguishable by both hue AND lightness, not just hue.
PALETTE_LINES = sns.color_palette("colorblind")

# Event-marker styles, keyed by src.key_events.KeyEvent.type -- only types listed here get
# drawn; add_event_lines() skips any event whose type isn't a key. Color is the primary
# distinguishing cue (dashed-vs-dotted line PATTERNS turned out hard to tell apart at a
# glance for thin vertical lines, even with only two of them) -- three well-separated hues,
# all solid, rather than leaning on pattern.
EVENT_STYLES = {
    "global_strike": {"color": "#c0392b", "linestyle": "-"},  # red
    "eu_election":   {"color": "#2980b9", "linestyle": "-"},  # blue
    "bt_election":   {"color": "#8e44ad", "linestyle": "-"},  # purple
}

# Display labels for event types -- "eu"/"bt" don't title-case sensibly via a generic
# str.replace("_", " ").title() (would read "Eu Election"/"Bt Election").
EVENT_TYPE_LABELS = {
    "global_strike": "Global Strike",
    "strike": "Strike",
    "campaign": "Campaign",
    "congress": "Congress",
    "election": "Election",
    "eu_election": "EU Election",
    "bt_election": "Bundestag Election",
    "COP": "COP",
}

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def set_style():
    """Apply the project-wide publication style. Call once per notebook session."""
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE_QUALITATIVE,
        rc={
            "figure.figsize": FIGSIZE_WIDE,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.title_fontsize": FONT_SIZE_LEGEND,
            "figure.titlesize": FONT_SIZE_TITLE,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.6,
        },
    )


def annotate_bars(ax, fmt="%.0f", **kwargs):
    """Add value labels to a bar/count plot, sized consistently with the rest of the figure."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=FONT_SIZE_ANNOTATION, padding=2, **kwargs)


def add_event_lines(ax, events, styles=None, linewidth=1.4, alpha=0.85, zorder=2):
    """Vertical lines marking specific dates (e.g. src.key_events.KEY_EVENTS), styled by
    event `type` via `styles` (dict: type -> {"color", "linestyle"}; defaults to
    EVENT_STYLES). Only events whose type is a key in `styles` are drawn -- pass the default
    for a curated plot showing just global_strike/eu_election/bt_election, or pass every
    type you care about for a fuller view.

    Deliberately unlabeled on the plot itself -- with many events, inline text labels would
    overlap and clutter the figure; print the event list as a small reference table in the
    notebook instead, and let the reader cross-reference by date.

    Returns the (type, style) pairs actually drawn, in first-seen order, so the caller can
    build a matching legend (see notebooks/05_llm_scoring_analysis.ipynb).

    Default `zorder=2` sits above shaded axvspan backgrounds (zorder=0) but below plotted
    data lines (zorder=3) -- fine for a line plot. For a filled/stacked plot (e.g.
    ax.stackplot), the fill is opaque and would hide lines drawn before it -- call this
    AFTER the fill instead, with a zorder higher than the fill's.
    """
    import pandas as pd

    styles = EVENT_STYLES if styles is None else styles
    drawn = []
    seen_types = set()
    for event in events:
        style = styles.get(event.type)
        if style is None:
            continue
        ax.axvline(pd.Timestamp(event.date, tz="UTC"), color=style["color"], linestyle=style["linestyle"],
                   linewidth=linewidth, alpha=alpha, zorder=zorder)
        if event.type not in seen_types:
            seen_types.add(event.type)
            drawn.append((event.type, style))
    return drawn


def _intensify_color(rgb, saturation=0.65, lightness=0.32):
    """Boost saturation and lower lightness of an RGB color while keeping its hue -- mirrors
    scripts/build_dashboard.py's JS `intensifyColor()`, so window-name labels use the same
    "same hue, more saturated" convention as the interactive dashboard."""
    h, _l, _s = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, lightness, saturation)


def add_window_labels(ax, resolved_windows, colors, y=0, y_stagger=0.06, fontsize=9):
    """Short labels (the text before the first ":" in each window's description) centered
    horizontally in the window and near the bottom of the axes, colored as a more saturated
    version of the window's own shading color -- same convention as the interactive
    dashboard. `resolved_windows` is `crisis_windows.resolve_windows()`'s output; `colors`
    must be the same per-window color list passed to the `axvspan()` shading calls.

    Deliberately NOT matched to FONT_SIZE_BASE/scaled_fonts()'s body-text convention, unlike
    title/legend/axis text: a window label must fit inside its own often-narrow date range
    (e.g. a one-year-wide crisis window on an 8-year x-axis), so sizing it to "match body
    text" makes it wider than the window for any short window, overlapping its neighbors --
    it's space-constrained like the correlation matrix's heatmap cell annotations, not a
    "should read like prose" element. If calling from inside scaled_fonts(...), pass an
    explicitly small scaled value (e.g. `fontsize=4.5 * scale`) rather than a value anywhere
    near FONT_SIZE_BASE, and check the rendered figure for overlap on the narrowest window.

    Every other label is lifted by `y_stagger` (same axes-fraction units as `y`) so two
    short, back-to-back windows don't collide horizontally even at a fixed font size --
    cheaper than shrinking the font further, and windows are date-ordered so alternating by
    index means adjacent windows are always on different rows.
    """
    text_kwargs = {} if fontsize is None else {"fontsize": fontsize}
    for i, ((_name, start, end, description), color) in enumerate(zip(resolved_windows, colors)):
        mid = start + (end - start) / 2
        label_y = y if i % 2 == 0 else y + y_stagger
        ax.text(mid, label_y, description.split(":")[0], transform=ax.get_xaxis_transform(),
                 ha="center", va="bottom", fontweight="bold",
                 color=_intensify_color(color), zorder=2, **text_kwargs)


# The REAL single-column text width of the report, measured directly from the actual
# Overleaf/LaTeX document via `\the\columnwidth` (sn-jnl.cls, single-column layout) rather
# than hand-computed from the .cls geometry settings -- 372.0pt, converted to inches using
# TeX's point (72.27pt = 1in, NOT the 72pt "big point"/PostScript point).
# Font sizes are an ABSOLUTE physical size (points), not relative to the figure. Every figure
# gets inserted into the report at this same final width regardless of its own native
# figsize, so a figure saved at a WIDER figsize (e.g. FIGSIZE_HERO's 13in) gets shrunk MORE to
# reach _REFERENCE_WIDTH, and the "same" point size ends up reading smaller on the page than
# it did on a narrower figure's canvas. scaled_fonts() below corrects for that, by scaling
# every figure's source font size proportionally to how much bigger its own canvas is than
# _REFERENCE_WIDTH, so after the report's own shrink-to-fit they all land back on the same
# FONT_SIZE_* value.
_REFERENCE_WIDTH = 372.0 / 72.27


@contextlib.contextmanager
def scaled_fonts(figsize):
    """Temporarily scale every font-size rcParam by `figsize[0] / _REFERENCE_WIDTH`, so that
    once this figure is shrunk (or grown) to fit the report's actual _REFERENCE_WIDTH column,
    its text lands on the FONT_SIZE_* values above -- regardless of this figure's own native
    figsize. Use for literally every figure meant for the report (including FIGSIZE_HERO ones
    -- FIGSIZE_HERO is no longer the reference size, it's simply another figsize that needs
    scaling like any other):

        with plot_style.scaled_fonts(plot_style.FIGSIZE_HERO):
            fig, ax = plt.subplots(figsize=plot_style.FIGSIZE_HERO)
            ax.set_title(...)   # picks up the scaled size when called INSIDE this block
            ...

    Every place that sets text (title/labels/ticks/legend -- including any *inside*
    seaborn/matplotlib helper calls, like sns.barplot's internal tick labels) must run
    inside the `with` block, since a Text object's fontsize is resolved from rcParams at
    creation time and doesn't update later. It's fine to call `plot_style.savefig()` /
    `plt.show()` before or after exiting the block -- the figure's artists already have
    their fontsize baked in by then.

    A wide figure's text will look SMALLER inline in this notebook than a narrow one's, even
    though both end up the same final size once placed in the report at _REFERENCE_WIDTH --
    that's expected, since inline display shows each figure at its own native inches, not at
    the report's actual final size. Don't "fix" a small-looking inline title by passing an
    explicit `fontsize=` -- that bypasses the scaling and reintroduces the original mismatch.

    Yields the computed `scale` factor, for the rare element that isn't driven by an rcParam
    the block above sets (e.g. seaborn heatmap's `annot_kws={"size": ...}`) -- multiply its
    own base size by this to keep it consistent with everything else in the figure:

        with plot_style.scaled_fonts((12, 10)) as scale:
            sns.heatmap(..., annot_kws={"size": 6 * scale})
    """
    scale = figsize[0] / _REFERENCE_WIDTH
    with plt.rc_context({
        "font.size": FONT_SIZE_BASE * scale,
        "axes.titlesize": FONT_SIZE_TITLE * scale,
        "axes.labelsize": FONT_SIZE_LABEL * scale,
        "xtick.labelsize": FONT_SIZE_TICK * scale,
        "ytick.labelsize": FONT_SIZE_TICK * scale,
        "legend.fontsize": FONT_SIZE_LEGEND * scale,
        "legend.title_fontsize": FONT_SIZE_LEGEND * scale,
        "figure.titlesize": FONT_SIZE_TITLE * scale,
    }):
        yield scale


def savefig(fig, name, subdir=None):
    """Save a figure as PDF (vector, for LaTeX/print). PDF only, deliberately -- no PNG."""
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")

"""Chart styling: a validated palette and a recessive matplotlib theme.

Colour is assigned by the job it does, not by taste:

* **sequential** (magnitude, such as a missingness percentage) is one hue
  running light to dark,
* **diverging** (polarity, such as a correlation or an EPA either side of zero)
  is two opposed hues with a neutral grey midpoint, never a hue at the middle,
* **categorical** (identity) draws slots in a fixed order, never cycled.

The categorical slots below were checked with a colour-vision validator: the
three used here clear the all-pairs separation floors (worst CVD delta-E 9.2,
worst normal-vision delta-E 24.0). Aqua sits below 3:1 against the surface, so
anything drawn in it carries a direct label rather than relying on colour alone.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e4e3df"

# Fixed categorical order. Slots are assigned in this order and never cycled.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")

# One hue, light to dark, for continuous magnitude.
SEQUENTIAL_STEPS = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#184f95",
    "#0d366b",
)

# Two opposed poles with a neutral grey midpoint, so "no relationship" reads as
# nothing rather than as a colour.
DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "gridiron_sequential", list(SEQUENTIAL_STEPS)
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "gridiron_diverging", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
)

FIGURE_DPI = 150


def apply_theme() -> None:
    """Install the project's chart defaults on matplotlib.

    Grid and axis lines are deliberately recessive: solid and pale, never
    dashed, so they sit behind the data instead of competing with it.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 10.5,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "text.color": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )

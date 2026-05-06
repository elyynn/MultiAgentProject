"""Shared style module for v2 figures.

Defines:
- Color palette for the six experiment arms (color-blind safe).
- Color palette for the three candidate types (Low/Medium/High).
- Two matplotlib rcParam presets: ``paper`` and ``slides``.

All figure functions in ``figures/make_figures.py`` import from here so colors
and labels remain consistent across every figure.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

# Okabe-Ito color-blind safe palette (https://jfly.uni-koeln.de/color/).
#  - black           #000000
#  - orange          #E69F00
#  - sky blue        #56B4E9
#  - bluish green    #009E73
#  - yellow          #F0E442
#  - blue            #0072B2
#  - vermillion      #D55E00
#  - reddish purple  #CC79A7

# Arm colors. ``default`` is neutral dark gray to read as the baseline; the
# five ablations get distinct hues.
ARM_COLORS = {
    "default":        "#333333",
    "no_detection":   "#D55E00",   # vermillion - large effect arm, stands out
    "no_spillover":   "#0072B2",   # blue
    "null_ai":        "#009E73",   # green - "AI off" floor
    "sens_detect_lo": "#E69F00",   # orange
    "sens_detect_hi": "#56B4E9",   # sky blue
}

ARM_LABELS = {
    "default":        "default",
    "no_detection":   "no_detection",
    "no_spillover":   "no_spillover",
    "null_ai":        "null_ai",
    "sens_detect_lo": "sens_detect_lo",
    "sens_detect_hi": "sens_detect_hi",
}

# Display order for cross-arm plots (default first, then ablations).
ARM_ORDER = [
    "default",
    "null_ai",
    "no_detection",
    "no_spillover",
    "sens_detect_lo",
    "sens_detect_hi",
]

# Candidate-type palette (Low/Medium/High).  Distinct from arm colors so
# per-type lines never visually collide with an arm's color.
TYPE_COLORS = {
    0: "#0072B2",   # Low    - blue
    1: "#E69F00",   # Medium - orange
    2: "#CC79A7",   # High   - reddish purple
}
TYPE_LABELS = {0: "Low", 1: "Medium", 2: "High"}
TYPE_ORDER = [0, 1, 2]

# Overall (aggregate-across-types) line color, used in F1, F2 alongside per-type lines.
OVERALL_COLOR = "#333333"

# Offer category colors for stacked bars (F9).
OFFER_COLORS = {
    0: "#BBBBBB",   # reject - light gray
    1: "#56B4E9",   # low offer - sky blue
    2: "#D55E00",   # high offer - vermillion
}
OFFER_LABELS = {0: "reject", 1: "low offer", 2: "high offer"}


# ---------------------------------------------------------------------------
# Figure presets
# ---------------------------------------------------------------------------

PAPER_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
    "savefig.transparent": False,
}

SLIDES_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
}

# Default panel sizes (per matplotlib axes panel; gridspecs scale up).
PANEL_SIZE = {
    "paper":  (5.5, 3.5),
    "slides": (10.0, 6.0),
}


def apply_variant(variant: str) -> None:
    """Reset rcParams and apply the chosen variant preset."""
    mpl.rcdefaults()
    if variant == "paper":
        mpl.rcParams.update(PAPER_RC)
    elif variant == "slides":
        mpl.rcParams.update(SLIDES_RC)
    else:
        raise ValueError(f"unknown variant: {variant!r}")


def panel_figsize(variant: str, ncols: int = 1, nrows: int = 1,
                  width_scale: float = 1.0, height_scale: float = 1.0) -> tuple:
    """Return a figsize sized for ``ncols x nrows`` panels."""
    w, h = PANEL_SIZE[variant]
    return (w * ncols * width_scale, h * nrows * height_scale)


def ci95(arr, axis: int = 0):
    """Return (mean, lower, upper) where bands are 1.96 * sem.

    ``arr`` shape: ``(n_seeds, n_epochs)`` (or any shape with ``axis`` indexing seeds).
    """
    import numpy as np
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[axis]
    mean = arr.mean(axis=axis)
    if n <= 1:
        return mean, mean.copy(), mean.copy()
    sem = arr.std(axis=axis, ddof=1) / np.sqrt(n)
    half = 1.96 * sem
    return mean, mean - half, mean + half


def add_ci_line(ax, x, arr, color: str, label: str, alpha_band: float = 0.2,
                linestyle: str = "-"):
    """Plot mean line and CI band on ``ax``."""
    mean, lo, hi = ci95(arr, axis=0)
    line, = ax.plot(x, mean, color=color, label=label, linestyle=linestyle)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha_band, linewidth=0)
    return line


def savefig_both(fig, paper_path_no_ext: str | None = None,
                 slides_path: str | None = None) -> None:
    """Save fig as PDF+PNG (paper) or PNG (slides) depending on supplied paths."""
    if paper_path_no_ext is not None:
        fig.savefig(paper_path_no_ext + ".pdf")
        fig.savefig(paper_path_no_ext + ".png")
    if slides_path is not None:
        fig.savefig(slides_path)
    plt.close(fig)

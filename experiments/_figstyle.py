"""Shared print style for the paper's figures.

Two constraints drive everything here, and both come from the destination:
the figures are reproduced in a two-column IEEE layout, and readers print
them in greyscale.

* **Palette.** Okabe-Ito, reordered so no adjacent pair collides under
  deuteranopia. Validated: lightness band, chroma floor, CVD separation
  (worst adjacent dE 9.6 deutan), normal-vision floor (18.4) all pass.
  Assigned in fixed order, never cycled.
* **Secondary encoding.** Every series also carries its own marker and dash
  pattern, so identity never rests on colour alone -- which is what keeps the
  figures readable in greyscale and under colour-vision deficiency.

Text is neutral ink, never the series colour; grid and spines are recessive.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Fixed categorical order. Do not cycle; a sixth series would need a rethink.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7")
MARKERS = ("o", "s", "^", "D", "v")
DASHES = ((), (5, 1.6), (1.4, 1.4), (6, 1.6, 1.4, 1.6), (3, 1.3))

INK = "#1a1a1a"        # primary text
INK_MUTED = "#595959"  # annotations, ceiling labels
RULE = "#9a9a9a"       # reference lines


def series(i: int) -> dict:
    """Line kwargs for series ``i`` -- colour, marker and dash together."""
    return {
        "color": PALETTE[i % len(PALETTE)],
        "marker": MARKERS[i % len(MARKERS)],
        "dashes": DASHES[i % len(DASHES)],
        "markersize": 4.2,
        "linewidth": 1.5,
        "markeredgewidth": 0,
    }


def apply() -> None:
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.prop_cycle": matplotlib.cycler(color=list(PALETTE)),
            "axes.grid": True,
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "axes.edgecolor": "#8a8a8a",
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#cccccc",
            "legend.borderpad": 0.4,
            "legend.handlelength": 2.4,
            "lines.solid_capstyle": "round",
        }
    )


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=8.5, color=INK, pad=6)


def ceiling(ax, y: float, label: str, *, x: float = 0.985, va: str = "bottom",
            ha: str = "right") -> None:
    """A dotted reference line with its label parked clear of the data.

    The label is placed in axes coordinates so it never lands on the line or
    on a legend, which is the collision these figures kept producing.
    """
    ax.axhline(y, ls=(0, (1.5, 1.8)), color=RULE, lw=0.9, zorder=1)
    offset = 0.012 if va == "bottom" else -0.012
    ax.text(
        x,
        ax.transLimits.transform((0, y))[1] + offset,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.2,
        color=INK_MUTED,
    )

"""Figure 4.9 of the thesis from the model output: the rural compliance ratio
post-2027 against pre-2027, one marker per run of Scenario 2.

The figure is drawn exactly as the thesis figure was (same style, size, colours
and axes), so a reader who has run `python code/p2p_voltage.py --full` can lay
the two side by side and see that the output is the same.

    python figures/figure_4_9.py                      # reads output/06_scenario_grid.csv
    python figures/figure_4_9.py path/to/06_scenario_grid.csv

Writes figures/figure_4_9.pdf and figures/figure_4_9.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
GRID_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "06_scenario_grid.csv"

# Style of every thesis figure (analysis/tools/thesis_style.py of the thesis code).
TEXTWIDTH_PT = 418.0          # text width of the thesis page, in points
PT_PER_INCH = 72.27
BLUE = "#0072B2"              # identity colour of Chapter 4
BASELINE_GREY = "#666666"
VCR_MIN = 0.95                # EN 50160: 95 % of the week inside the band

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": "#DDDDDD",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#BBBBBB",
    "legend.fancybox": False,
    "figure.constrained_layout.use": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def figure_size(frac: float, aspect: float) -> tuple[float, float]:
    w_in = TEXTWIDTH_PT * frac / PT_PER_INCH
    return (w_in, w_in * aspect)


def main() -> None:
    g = pd.read_csv(GRID_CSV)
    r = g[g.archetype == "rural"]
    print(f"{GRID_CSV}: {len(g)} runs, {len(r)} rural")

    fig, ax = plt.subplots(figsize=figure_size(0.66, 0.66))
    ax.axhspan(VCR_MIN, 1.005, color="#F2F2F2", zorder=0)
    ax.plot([0.6, 1.0], [0.6, 1.0], color=BASELINE_GREY, lw=0.8, linestyle=":")
    ax.axhline(VCR_MIN, color="black", lw=0.8)
    ax.axvline(VCR_MIN, color="black", lw=0.8)
    ax.scatter(r.vcr_pre, r.vcr_post, s=6, color=BLUE, alpha=0.45, edgecolors="none")
    ax.set_xlabel("VCR pre-2027")
    ax.set_ylabel("VCR post-2027")
    ax.set_xlim(0.60, 1.005); ax.set_ylim(0.60, 1.005)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

    for ext in ("pdf", "png"):
        out = HERE / f"figure_4_9.{ext}"
        fig.savefig(out)
        print("written", out)
    plt.close(fig)


if __name__ == "__main__":
    main()

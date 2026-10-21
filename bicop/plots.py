"""Figure 1 of the paper: the Pólya bounds zeta^(r) by level, from data/polya_levels.csv (no solver).

Run from code/:  python plots.py
Drawn at printed size (text width about 4.6 in) with the paper's fonts; included without rescaling.
"""

import csv
import logging
import math
import os
import subprocess
from collections import defaultdict

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED = "#0b0b0b", "#52514e"
SERIES = [("matching-3", r"$M_3$, and $2+2$ with one edge", "#2a78d6", "o"),
          ("matching-4", r"$M_4$", "#eb6834", "s"),
          ("crown-4", r"$K_{4,4}$ minus a perfect matching", "#1baf7a", "^")]


def paper_style():
    """Latin Modern text, Computer Modern math, 8 pt, fonts embedded as Type 42."""
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    try:
        path = subprocess.run(["kpsewhich", "lmroman10-regular.otf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        path = ""
    if path:
        fm = matplotlib.font_manager.fontManager
        fm.addfont(path)
        # every optical size of Latin Modern registers as "Latin Modern Roman"; keep only the 10 pt one
        fm.ttflist = [f for f in fm.ttflist if f.name != "Latin Modern Roman" or f.fname == path]
        plt.rcParams["font.family"] = "Latin Modern Roman"
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6})


def polya_levels():
    """zeta^(r) / (sqrt(g)/2) against r; levels where the program is infeasible are not drawn."""
    data = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "polya_levels.csv"))):
        data[r["graph"]].append((int(r["r"]), float(r["zeta"]) / float(r["half_sqrt_g"])))
    fig, ax = plt.subplots(figsize=(4.6, 2.3), constrained_layout=True)
    ax.axhline(1, color=MUTED, lw=0.8, ls="--")
    ax.text(-0.2, 1.06, r"value $\frac{1}{2}\sqrt{g}$", va="bottom", ha="left", fontsize=6.5, color=MUTED)
    for key, label, color, marker in SERIES:
        pts = sorted((r, v) for r, v in data[key] if math.isfinite(v))
        ax.plot([r for r, _ in pts], [v for _, v in pts], color=color, lw=1.4, marker=marker, ms=4.5,
                mec="white", mew=0.6, label=label)
    ax.set_xticks(range(8))
    ax.set_xlim(-0.3, 7.3)
    ax.set_ylim(0.8, 3.3)
    ax.set_xlabel("level $r$")
    ax.set_ylabel(r"$\zeta^{(r)}\,/\,\frac{1}{2}\sqrt{g}$")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "polya_levels.pdf"))


if __name__ == "__main__":
    paper_style()
    polya_levels()

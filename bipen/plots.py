"""The numbers of the figure of the paper, read from a data folder and written as CSV files to a figures folder; the
paper draws them with pgfplots. Column names use letters only. Run with: bipen plots
"""

import csv
import math
import os
from collections import defaultdict

GRAPHS = {"lpv-example-1": "oneedge", "matching-3": "matchingthree", "matching-4": "matchingfour", "crown-4": "crownfour"}


def polya_levels(data, figures):
    """zeta^(r) / (sqrt(g)/2) by level r, one column per graph; nan where the program is infeasible."""
    levels = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(data, "polya_levels.csv"))):
        ratio = float(r["zeta"]) / float(r["half_sqrt_g"])
        levels[int(r["r"])][GRAPHS[r["graph"]]] = ratio if math.isfinite(ratio) else math.nan
    with open(os.path.join(figures, "polya.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", *GRAPHS.values()])
        w.writeheader()
        w.writerows({"level": r, **levels[r]} for r in sorted(levels))


def build(data, figures):
    os.makedirs(figures, exist_ok=True)
    polya_levels(data, figures)

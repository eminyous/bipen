"""The numbers of the figure of the paper, read from the data folder and written as CSV files to the figures folder,
which the paper draws with pgfplots. Column names use letters only.
Run with: bipen plots
"""

import csv
import math
from collections import defaultdict

from bipen.data import Column, File, Graph, read_csv

CURVES = {
    Graph.LPV_EXAMPLE: "oneedge",
    Graph.MATCHING_3: "matchingthree",
    Graph.MATCHING_4: "matchingfour",
    Graph.CROWN_4: "crownfour",
}  # data graph: column of the figure


def polya_figure(data, figures):
    """Figure 1: zeta^(r) / (sqrt(g)/2) by level r, one column per graph, nan where the program is infeasible."""
    ratios = defaultdict(dict)
    for row in read_csv(data / File.POLYA_BOUNDS):
        ratio = float(row[Column.ZETA]) / float(row[Column.HALF_SQRT_G])
        ratios[int(row[Column.LEVEL])][CURVES[row[Column.GRAPH]]] = ratio if math.isfinite(ratio) else math.nan
    with (figures / "polya.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", *CURVES.values()])
        writer.writeheader()
        writer.writerows({"level": level, **ratios[level]} for level in sorted(ratios))


def build(data, figures):
    figures.mkdir(parents=True, exist_ok=True)
    polya_figure(data, figures)

"""The Pólya bounds zeta^(r) by level on the graphs of Figure 1.

Levels 0 <= r <= 7, on the graph of LPV, Example 1, on 3K_2 and 4K_2, and on the crown on 4 + 4 vertices.
Run with: bipen run polya/levels
"""

import warnings

import numpy as np

from bipen.data import Column, File, Graph, write_csv
from bipen.exact import alpha_star, biclique_number
from bipen.graphs import crown, matching
from bipen.polya import zeta

warnings.filterwarnings("ignore")
MAX_LEVEL = 7


def figure_graphs():
    yield Graph.LPV_EXAMPLE, np.array([[1.0, 0.0], [0.0, 0.0]])
    yield Graph.MATCHING_3, matching(3)
    yield Graph.MATCHING_4, matching(4)
    yield Graph.CROWN_4, crown(4)


def main(data):
    def rows():
        for name, U in figure_graphs():
            half_sqrt_g, a = 0.5 * np.sqrt(biclique_number(U)), alpha_star(U)
            for r in range(MAX_LEVEL + 1):
                value = zeta(U, r)
                print(name, r, value, flush=True)
                yield {
                    Column.GRAPH: name,
                    Column.HALF_SQRT_G: half_sqrt_g,
                    Column.ALPHA_STAR: a,
                    Column.LEVEL: r,
                    Column.ZETA: value,
                }

    write_csv(data / File.POLYA_LEVELS, rows())

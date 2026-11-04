"""h_1, h_1^D and rho^(1) on the structured graphs and the first random graphs (Table 2).

Also g, h and sqrt(g)/2. Graphs: the graph of LPV, Example 1; the matchings sK_2, 2 <= s <= 12; the crowns,
5 <= s <= 12; and ten random graphs for each balanced size 6+6 to 20+20 and each p in {0.3, 0.5, 0.7}. rho^(1) is
computed up to RHO_MAX_N vertices only, as its cost grows like n^3.
Run with: bipen run bounds/structured
"""

import time
import warnings

import numpy as np

from bipen.data import Column, Family, File, write_csv
from bipen.exact import biclique_number, biclique_ratio
from bipen.graphs import bounds_graph, crown, matching
from bipen.sdp import h1, rho1

warnings.filterwarnings("ignore")
RHO_MAX_N = 30


def bounds_row(family, U, p="", s=""):
    """The bounds on one graph, as a row of the files STRUCTURED_BOUNDS and RANDOM_BOUNDS."""
    start = time.time()
    m1, m2 = U.shape
    g = biclique_number(U)
    return {
        Column.FAMILY: family,
        Column.M1: m1,
        Column.M2: m2,
        Column.P: p,
        Column.SEED: s,
        Column.EDGES: int(U.sum()),
        Column.G: g,
        Column.H: biclique_ratio(U),
        Column.HALF_SQRT_G: 0.5 * np.sqrt(g),
        Column.H1: h1(U),
        Column.H1_DNN: h1(U, dnn=True),
        Column.RHO1: rho1(U) if m1 + m2 <= RHO_MAX_N else "",
        Column.SECONDS: round(time.time() - start, 2),
    }


def graphs():
    yield Family.LPV_EXAMPLE, np.array([[1.0, 0.0], [0.0, 0.0]]), "", ""
    for s in range(2, 13):
        yield Family.MATCHING, matching(s), "", ""
    for s in range(5, 13):
        yield Family.CROWN, crown(s), "", ""
    for m in [6, 8, 10, 12, 15, 20]:
        for p in [0.3, 0.5, 0.7]:
            for s in range(10):
                yield Family.RANDOM, bounds_graph(m, m, p, s), p, s


def main(data):
    def rows():
        for family, U, p, s in graphs():
            row = bounds_row(family, U, p, s)
            print(family, U.shape, p, s, row[Column.SECONDS], flush=True)
            yield row

    write_csv(data / File.STRUCTURED_BOUNDS, rows())

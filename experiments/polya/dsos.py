"""The DSOS bounds zeta~^(r) by level.

Levels 0 <= r <= 2, on the graphs of polya/levels and on N_2.
Run with: bipen run polya/dsos
"""

import warnings

import numpy as np

from bipen.data import Column, File, Graph, write_csv
from bipen.dsos import zeta_tilde
from bipen.exact import biclique_number
from bipen.graphs import near_biclique
from experiments.polya.levels import figure_graphs

warnings.filterwarnings("ignore")
MAX_LEVEL = 2


def graphs():
    yield from figure_graphs()
    yield Graph.NEAR_BICLIQUE_2, near_biclique(2)


def main(data):
    def rows():
        for name, U in graphs():
            half_sqrt_g = 0.5 * np.sqrt(biclique_number(U))
            for r in range(MAX_LEVEL + 1):
                value = zeta_tilde(U, r)
                print(name, r, value, flush=True)
                yield {Column.GRAPH: name, Column.HALF_SQRT_G: half_sqrt_g, Column.LEVEL: r, Column.ZETA_TILDE: value}

    write_csv(data / File.POLYA_DSOS, rows())

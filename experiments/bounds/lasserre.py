"""rho^(1) against the Lasserre bound g_2, values and times (Section 6).

Graphs: the first ten random graphs of Table 1 of each balanced size up to 15+15 and each p in {0.3, 0.5, 0.7}; with
--size and --seeds, that many graphs of that size only, in a file of its own.
Run with: bipen run bounds/lasserre --workers N [--size 20 --seeds 3]
"""

import time
import warnings
from multiprocessing import Pool

import numpy as np

from bipen.data import Column, File, write_csv
from bipen.exact import biclique_number
from bipen.graphs import bounds_graph
from bipen.sdp import lasserre_g, rho1

warnings.filterwarnings("ignore")
SIZES = [6, 8, 10, 12, 15]
PS = [0.3, 0.5, 0.7]


def timed(f, *args):
    start = time.time()
    value = f(*args)
    return value, round(time.time() - start, 2)


def job(graph):
    m, p, s = graph
    U = bounds_graph(m, m, p, s)
    rho, rho_seconds = timed(rho1, U)
    g2, lasserre_seconds = timed(lasserre_g, U)
    return {
        Column.M1: m,
        Column.M2: m,
        Column.P: p,
        Column.INDEX: s,
        Column.HALF_SQRT_G: 0.5 * np.sqrt(biclique_number(U)),
        Column.RHO1: rho,
        Column.HALF_SQRT_G2: 0.5 * np.sqrt(g2),
        Column.SECONDS_RHO1: rho_seconds,
        Column.SECONDS_LASSERRE: lasserre_seconds,
    }


def main(data, workers=2, size=None, seeds=10):
    sizes, name = (SIZES, File.BOUNDS_LASSERRE) if size is None else ([size], File.bounds_lasserre_size(size))
    todo = sorted([(m, p, s) for m in sizes for p in PS for s in range(seeds)], key=lambda graph: -graph[0])

    def rows(pool):
        for row in pool.imap_unordered(job, todo):
            print(*row.values(), flush=True)
            yield row

    with Pool(workers) as pool:
        write_csv(data / name, rows(pool))
    print("finished", flush=True)

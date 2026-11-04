"""The smallest valid penalty gamma* against the thresholds (Table 3).

With the thresholds gamma_1 = (1 + sqrt g)/2 and gamma_2 = (1 + max(m1, m2))/2, the threshold of Gillis and
Glineur after the change of parameter 2 gamma - 1.

Graphs: 100 random graphs for each size 2+3 to 5+5 and each p in {0.2, 0.4, 0.6, 0.8}, without the empty and the
complete ones, and the graphs N_k, 2 <= k <= 5.
Run with: bipen run penalty/smallest --workers N
"""

from multiprocessing import Pool

import numpy as np

from bipen.data import Column, Family, File, write_csv
from bipen.exact import biclique_number
from bipen.graphs import near_biclique, penalty_graph
from bipen.penalty import smallest_penalty

SIZES = [(2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
PS = [0.2, 0.4, 0.6, 0.8]
GRAPHS = 100


def penalty_row(family, U, p="", s=""):
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
        Column.GAMMA_STAR: smallest_penalty(U),
        Column.GAMMA1: (1 + np.sqrt(g)) / 2,
        Column.GAMMA2: (1 + max(m1, m2)) / 2,
    }


def job(graph):
    m1, m2, p, s = graph
    U = penalty_graph(m1, m2, p, s)
    if U.sum() in (0, m1 * m2):
        return None
    return penalty_row(Family.RANDOM, U, p, s)


def main(data, workers=6):
    todo = [(m1, m2, p, s) for m1, m2 in SIZES for p in PS for s in range(GRAPHS)]
    with Pool(workers) as pool:
        rows = [row for row in pool.map(job, todo, chunksize=8) if row is not None]
    rows += [penalty_row(Family.NEAR_BICLIQUE, near_biclique(k)) for k in range(2, 6)]
    write_csv(data / File.SMALLEST_PENALTY, rows)
    print(len(rows), "rows written")

"""The first level at which the Pólya bound is finite, against alpha* - 1 (Theorem 5.2).

Graphs: 15 random graphs for each size 2+2, 2+3, 3+3 and 3+4, with an edge probability drawn in [0.2, 0.8], without
the complete ones.
Run with: bipen run polya/finiteness
"""

import warnings

import numpy as np

from bipen.data import Column, File, write_csv
from bipen.exact import alpha_star
from bipen.polya import zeta

warnings.filterwarnings("ignore")


def first_finite_level(U, max_level):
    for r in range(max_level + 1):
        if np.isfinite(zeta(U, r)):
            return r
    return None


def main(data):
    rng = np.random.default_rng(8)
    rows = []
    for m1, m2 in [(2, 2), (2, 3), (3, 3), (3, 4)]:
        for _ in range(15):
            U = (rng.random((m1, m2)) < rng.uniform(0.2, 0.8)).astype(float)
            if U.sum() == m1 * m2:
                continue
            a = alpha_star(U)
            rows.append(
                {
                    Column.M1: m1,
                    Column.M2: m2,
                    Column.EDGES: int(U.sum()),
                    Column.ALPHA_STAR: a,
                    Column.FIRST_FINITE_LEVEL: first_finite_level(U, a),
                }
            )
    write_csv(data / File.POLYA_FINITENESS, rows)
    exceptions = sum(row[Column.FIRST_FINITE_LEVEL] != row[Column.ALPHA_STAR] - 1 for row in rows)
    print("graphs where the first finite level is not alpha* - 1:", exceptions)

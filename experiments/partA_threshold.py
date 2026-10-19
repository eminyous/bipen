"""Smallest penalty d* with max{u^T (C - d A_G) u : u >= 0, |u| = 1} = sqrt(g)/2.

Compares d* with the bound (1 + sqrt g)/2 and with (1 + max(|V1|,|V2|))/2, the
Gillis-Glineur threshold after the change of parameter d_GG = 2d - 1, on 100 random graphs for each size and each p
(empty and complete graphs are skipped) and on the graphs N_k. Run with: python -m experiments.partA_threshold
"""

import csv
import sys
from multiprocessing import Pool

import numpy as np

from bicop.brute import parameters
from bicop.graphs import cost_matrix, full_adjacency, random_bipartite
from bicop.penalty import max_on_nonneg_sphere


def threshold(B, tol=1e-7):
    """Return (d*, g) with d* the smallest d giving the value sqrt(g)/2 (bisection)."""
    m, k = B.shape
    C, A = cost_matrix(m, k), full_adjacency(B)
    g = parameters(B)["g"]
    target = 0.5 * np.sqrt(g)
    ok = lambda d: max_on_nonneg_sphere(C - d * A) <= target + 1e-9
    lo, hi = 0.0, 1.0
    while not ok(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi, g


def near_biclique(a):
    """V1 = A + {i}, V2 = B, |A| = |B| = a, the only edge joins i to one vertex of B."""
    B = np.zeros((a + 1, a))
    B[a, 0] = 1
    return B


SIZES = [(2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
PS = [0.2, 0.4, 0.6, 0.8]
SEEDS = 100


def job(args):
    m, k, p, s = args
    B = random_bipartite(m, k, p, np.random.default_rng(100000 * k + 1000 * m + 100 * int(10 * p) + s))
    if B.sum() == 0 or B.sum() == m * k:
        return None
    d, g = threshold(B)
    return {"family": "random", "m": m, "k": k, "p": p, "seed": s, "edges": int(B.sum()), "g": g, "d_star": d,
            "bound_sqrt_g": (1 + np.sqrt(g)) / 2, "bound_gg": (1 + max(m, k)) / 2}


def main(out, workers):
    todo = [(m, k, p, s) for m, k in SIZES for p in PS for s in range(SEEDS)]
    with Pool(workers) as pool:
        rows = [r for r in pool.map(job, todo, chunksize=8) if r is not None]
    for a in range(2, 6):
        B = near_biclique(a)
        d, g = threshold(B)
        rows.append({"family": "near-biclique", "m": a + 1, "k": a, "p": "", "seed": "", "edges": 1, "g": g, "d_star": d,
                     "bound_sqrt_g": (1 + np.sqrt(g)) / 2, "bound_gg": (1 + a + 1) / 2})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/partA_threshold.csv", int(sys.argv[2]) if len(sys.argv) > 2 else 6)

"""Smallest penalty d* with max{u^T (C - d A_G) u : u >= 0, |u| = 1} = sqrt(g)/2.

Compares d* with the bound (1 + sqrt g)/2 and with (1 + max(|V1|,|V2|))/2, the
Gillis-Glineur threshold after the change of parameter d_GG = 2d - 1.
"""

import csv
import sys

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


def main(out):
    rng = np.random.default_rng(2026)
    rows = []
    for m, k in [(2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]:
        for p in [0.2, 0.4, 0.6, 0.8]:
            for _ in range(8):
                B = random_bipartite(m, k, p, rng)
                if B.sum() == 0 or B.sum() == m * k:
                    continue
                d, g = threshold(B)
                rows.append({"family": "random", "m": m, "k": k, "p": p, "edges": int(B.sum()), "g": g, "d_star": d,
                             "bound_sqrt_g": (1 + np.sqrt(g)) / 2, "bound_gg": (1 + max(m, k)) / 2})
    for a in range(2, 6):
        B = near_biclique(a)
        d, g = threshold(B)
        rows.append({"family": "near-biclique", "m": a + 1, "k": a, "p": "", "edges": 1, "g": g, "d_star": d,
                     "bound_sqrt_g": (1 + np.sqrt(g)) / 2, "bound_gg": (1 + a + 1) / 2})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    ratio = max(r["d_star"] / r["bound_sqrt_g"] for r in rows)
    viol = [r for r in rows if r["d_star"] > r["bound_sqrt_g"] + 1e-6]
    print(f"{len(rows)} graphs; max d*/((1+sqrt g)/2) = {ratio:.4f}; violations: {len(viol)}")
    for r in rows:
        if r["family"] == "near-biclique":
            print("near-biclique a =", r["k"], "d* =", round(r["d_star"], 5), "sqrt(g)/2 =", np.sqrt(r["g"]) / 2)
    top = sorted(rows, key=lambda r: -r["d_star"] / r["bound_sqrt_g"])[:5]
    for r in top:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/partA_threshold.csv")

"""Block-coordinate ascent on max x^T (J - 2 gamma W) y over nonnegative unit vectors, on the first 20 random graphs
of the sizes 10+10, 15+15 and 20+20 and each p, from two kinds of starts: 100 random points (numpy.random.default_rng(s)
for the graph with seed index s) and the n vertices (y = e_j or x = e_i). Penalties: gamma_1 = (1 + sqrt g)/2,
which rounding kappa^(1) (from random.csv) gives on every graph, gamma_4 = (1 + |J - W|_2)/2, gamma_3 = (1 + sqrt|E(G^)|)/2,
gamma_2 = (1 + max(m, k))/2, and the penalties 1/2 + c sqrt(g)/2 below gamma_1 for c = 1/4, 1/2, 3/4 (gamma_c25, ...).

Per graph, penalty and kind of start: success, the fraction of runs whose supports form a maximum biindependent pair
(for gamma >= gamma_1 these are the runs that end at value sqrt(g); below gamma_1 a run can exceed sqrt(g) with an
edge in its support); repaired, the fraction that reach g after the greedy repair of penalty.repair; edge, nonmaximal
and maximal, the fractions of runs that end with an edge of G in the support, at a biindependent pair that is not
maximal, and at a maximal pair that is not maximum (penalty.end_state); edge_positive, the fraction that end with an
edge in the support and a positive value; fallback, the fraction whose first step finds
(M y)_+ = 0 and moves to a vertex; support, the mean fraction of the side in the support after the first step; steps.
Writes local_ascent.csv in the data folder.

Run with: bipen run local-ascent --workers N
"""

import csv
import os
from multiprocessing import Pool

import numpy as np

from bipen.brute import parameters
from bipen.graphs import random_bipartite
from bipen.penalty import ascent, end_state, repair
from experiments.random_graphs import seed

SIZES = [10, 15, 20]
PS = [0.3, 0.5, 0.7]
STARTS = 100
KAPPA = {}  # kappa^(1) by graph, read from random.csv by main before the workers start


def best(a):
    """The best unit vector against a, as in penalty.ascent."""
    if a.max() <= 0:
        return np.eye(len(a))[np.argmax(a)]
    return np.maximum(a, 0) / np.linalg.norm(np.maximum(a, 0))


def run(M, start):
    """One ascent from a random point (x, y), or from a vertex ("x", i) or ("y", j); returns x, y, value, steps and
    the first-step vector, that is, the x computed from y (or the y computed from x)."""
    if isinstance(start[0], str) and start[0] == "x":
        a = M[start[1]]
        y, x, val, t = ascent(M.T, best(a), np.eye(M.shape[0])[start[1]])
        return x, y, val, t, a
    if isinstance(start[0], str):
        a = M[:, start[1]]
        x, y, val, t = ascent(M, best(a), np.eye(M.shape[1])[start[1]])
        return x, y, val, t, a
    x, y = start
    x, y, val, t = ascent(M, x, y)
    return x, y, val, t, M @ start[1]


def job(args):
    m, p, s = args
    B = random_bipartite(m, m, p, np.random.default_rng(seed(m, m, p, s))).astype(bool)
    g = parameters(B)["g"]
    assert np.floor(4 * KAPPA[(m, m, p, s)] ** 2 + 1e-6) == g  # rounding kappa^(1) gives g, so gamma_1 is computable
    thresholds = {"gamma1": (1 + np.sqrt(g)) / 2,
                  "gamma4": (1 + np.linalg.norm(1 - B, 2)) / 2,
                  "gamma3": (1 + np.sqrt((1 - B).sum())) / 2, "gamma2": (1 + m) / 2}
    thresholds.update({f"gamma_c{int(100 * c)}": 0.5 + c * np.sqrt(g) / 2 for c in (0.25, 0.5, 0.75)})
    rng = np.random.default_rng(s)
    random_starts = []
    for _ in range(STARTS):
        x, y = rng.random(m), rng.random(m)
        random_starts.append((x / np.linalg.norm(x), y / np.linalg.norm(y)))
    kinds = {"random": random_starts, "vertex": [("x", i) for i in range(m)] + [("y", j) for j in range(m)]}
    rows = []
    for name, gamma in thresholds.items():
        M = 1 - 2 * gamma * B
        for kind, starts in kinds.items():
            runs = [run(M, start) for start in starts]
            ends = [end_state(B, g, x, y) for x, y, *_ in runs]
            rows.append({"m": m, "k": m, "p": p, "seed": s, "g": g, "threshold": name, "gamma": gamma,
                         "starts": kind,
                         "success": np.mean([e == "optimal" for e in ends]),
                         "repaired": np.mean([repair(B, x, y) == g for x, y, *_ in runs]),
                         "edge": np.mean([e == "edge" for e in ends]),
                         "edge_positive": np.mean([e == "edge" and r[2] > 0 for e, r in zip(ends, runs)]),
                         "nonmaximal": np.mean([e == "nonmaximal" for e in ends]),
                         "maximal": np.mean([e == "maximal" for e in ends]),
                         "fallback": np.mean([a.max() <= 0 for *_, a in runs]),
                         "support": np.mean([(a > 0).sum() / len(a) for *_, a in runs]),
                         "steps": np.mean([t for _, _, _, t, _ in runs])})
    return rows


def main(data, workers=2):
    KAPPA.update({(int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"])): float(r["k1"])
                  for r in csv.DictReader(open(os.path.join(data, "random.csv")))})
    todo = [(m, p, s) for m in SIZES for p in PS for s in range(20)]
    with open(os.path.join(data, "local_ascent.csv"), "w", newline="") as f, Pool(workers) as pool:
        w = None
        for rows in pool.imap_unordered(job, todo):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
            w.writerows(rows)
            f.flush()
    print("finished", flush=True)

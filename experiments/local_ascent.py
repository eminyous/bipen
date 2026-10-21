"""Block-coordinate ascent on max x^T (J - 2 gamma W) y over nonnegative unit vectors, from 100 random starts
(numpy.random.default_rng(s) for the graph with seed index s), for the thresholds gamma_1 = (1 + sqrt g)/2,
gamma_k = 1/2 + kappa^(1) (from random.csv), gamma_4 = (1 + |J - W|_2)/2, gamma_3 = (1 + sqrt|E(G^)|)/2 and
gamma_2 = (1 + max(m, k))/2, on the first 20 random graphs of the sizes 10+10, 15+15 and 20+20 and each p. A run
succeeds when it ends at value sqrt(g), that is, at a maximum biindependent pair; fallback is the fraction of runs
whose first step finds (M y)_+ = 0 and moves to a vertex. Writes local_ascent.csv in the data folder.

Run with: bipen run local-ascent --workers N
"""

import csv
import os
from multiprocessing import Pool

import numpy as np

from bipen.brute import parameters
from bipen.graphs import random_bipartite
from bipen.penalty import ascent
from experiments.random_graphs import seed

SIZES = [10, 15, 20]
PS = [0.3, 0.5, 0.7]
STARTS = 100
KAPPA = {}  # kappa^(1) by graph, read from random.csv by main before the workers start


def job(args):
    m, p, s = args
    B = random_bipartite(m, m, p, np.random.default_rng(seed(m, m, p, s)))
    g = parameters(B)["g"]
    thresholds = {"gamma1": (1 + np.sqrt(g)) / 2, "gammak": 0.5 + KAPPA[(m, m, p, s)],
                  "gamma4": (1 + np.linalg.norm(1 - B, 2)) / 2,
                  "gamma3": (1 + np.sqrt((1 - B).sum())) / 2, "gamma2": (1 + m) / 2}
    rng = np.random.default_rng(s)
    starts = [(rng.random(m), rng.random(m)) for _ in range(STARTS)]
    rows = []
    for name, gamma in thresholds.items():
        M = 1 - 2 * gamma * B
        runs = [ascent(M, x / np.linalg.norm(x), y / np.linalg.norm(y)) for x, y in starts]
        rows.append({"m": m, "k": m, "p": p, "seed": s, "g": g, "threshold": name, "gamma": gamma,
                     "success": np.mean([v > np.sqrt(g) - 1e-6 for _, _, v, _ in runs]),
                     "fallback": np.mean([(M @ (y / np.linalg.norm(y))).max() <= 0 for _, y in starts]),
                     "steps": np.mean([t for _, _, _, t in runs])})
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


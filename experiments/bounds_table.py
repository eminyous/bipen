"""h, sqrt(g)/2, h1, h1_dnn and the K^(1) copositive bound on graph families.

Writes bounds.csv in the data folder. K^(1) is computed for n <= K1_MAX only (cost grows like n^3).
"""

import csv
import os
import time
import warnings

import numpy as np

from bicop.brute import parameters
from bicop.graphs import knn_minus_matching, random_bipartite
from bicop.sdp import cop_k1, h1

warnings.filterwarnings("ignore")
K1_MAX = 30
FIELDS = ["family", "m", "k", "p", "seed", "edges", "g", "h", "half_sqrt_g", "h1", "h1_dnn", "k1", "seconds"]


def row(family, B, p="", seed=""):
    t0 = time.time()
    par = parameters(B)
    m, k = B.shape
    r = {"family": family, "m": m, "k": k, "p": p, "seed": seed, "edges": int(B.sum()), "g": par["g"],
         "h": par["h"], "half_sqrt_g": 0.5 * np.sqrt(par["g"]), "h1": h1(B), "h1_dnn": h1(B, dnn=True)}
    r["k1"] = cop_k1(B) if m + k <= K1_MAX else ""
    r["seconds"] = round(time.time() - t0, 2)
    return r


def graphs():
    yield "lpv-ex3.7", np.array([[1.0, 0.0], [0.0, 0.0]]), "", ""
    for n in range(2, 13):
        yield "matching", np.eye(n), "", ""
    for n in range(5, 13):
        yield "crown", knn_minus_matching(n), "", ""
    for m in [6, 8, 10, 12, 15, 20]:
        for p in [0.3, 0.5, 0.7]:
            for s in range(10):
                yield "random", random_bipartite(m, m, p, np.random.default_rng(1000 * m + 100 * int(10 * p) + s)), p, s


def main(data):
    with open(os.path.join(data, "bounds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for fam, B, p, s in graphs():
            r = row(fam, B, p, s)
            w.writerow(r)
            f.flush()
            print(fam, B.shape, p, s, r["seconds"], flush=True)


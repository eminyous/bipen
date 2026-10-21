"""kappa^(1) against the Lasserre bound sqrt(g_2)/2 of LPV, values and times, on the first ten random graphs of
each balanced size up to 15+15 and each p (the graphs of bounds_table). Writes ../data/lasserre.csv.

Run with: python -m experiments.lasserre_compare <workers>
"""

import csv
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

from bicop.brute import parameters
from bicop.graphs import random_bipartite
from bicop.sdp import cop_k1, g_las
from experiments.random_graphs import seed

warnings.filterwarnings("ignore")
OUT = "../data/lasserre.csv"
SIZES = [6, 8, 10, 12, 15]
PS = [0.3, 0.5, 0.7]


def job(args):
    m, p, s = args
    B = random_bipartite(m, m, p, np.random.default_rng(seed(m, m, p, s)))
    t = time.time()
    k1 = cop_k1(B)
    t_k1 = time.time() - t
    t = time.time()
    las = 0.5 * np.sqrt(g_las(B))
    t_las = time.time() - t
    return {"m": m, "k": m, "p": p, "seed": s, "half_sqrt_g": 0.5 * np.sqrt(parameters(B)["g"]), "k1": k1,
            "las2": las, "seconds_k1": round(t_k1, 2), "seconds_las2": round(t_las, 2)}


def main(workers):
    todo = sorted([(m, p, s) for m in SIZES for p in PS for s in range(10)], key=lambda t: -t[0])
    with open(OUT, "w", newline="") as f, Pool(workers) as pool:
        w = None
        for r in pool.imap_unordered(job, todo):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r))
                w.writeheader()
            w.writerow(r)
            f.flush()
            print(*r.values(), flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

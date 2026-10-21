"""The bounds of bounds_table on 100 random graphs for each size and each p in {0.3, 0.5, 0.7}.

Balanced sizes use the seeds of bounds_table, 1000 m + 100 int(10 p) + s, so its 180 graphs are the seeds
s < 10; they are copied from ../data/bounds.csv when it exists and has their K^(1) bound. Unbalanced sizes use
100000 k + 1000 m + 100 int(10 p) + s. Results are appended to ../data/random.csv, largest graphs first,
and a rerun skips the graphs already there. The 20+20 graphs need about 10 minutes and 5 GB each, and the
whole run about 60 CPU hours.

Run with: python -m experiments.random_graphs <workers>
"""

import csv
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

from bicop.graphs import random_bipartite
from experiments import bounds_table
from experiments.bounds_table import FIELDS, row

warnings.filterwarnings("ignore")
bounds_table.K1_MAX = 40  # bounds_table skips K^(1) above n = 30; here it is computed for 20+20 too
OUT = "../data/random.csv"
OLD = "../data/bounds.csv"
SIZES = [(6, 6), (8, 8), (10, 10), (12, 12), (15, 15), (20, 20), (5, 10), (8, 16), (10, 20), (6, 24)]
PS = [0.3, 0.5, 0.7]
SEEDS = 100


def seed(m, k, p, s):
    base = 1000 * m if m == k else 100000 * k + 1000 * m
    return base + 100 * int(10 * p) + s


def key(r):
    return int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"])


def job(args):
    m, k, p, s = args
    return row("random", random_bipartite(m, k, p, np.random.default_rng(seed(m, k, p, s))), p, s)


def main(workers):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        if new and os.path.exists(OLD):
            w.writerows(r for r in csv.DictReader(open(OLD)) if r["family"] == "random" and r["k1"] not in ("", "nan"))
        f.flush()
    done = {key(r) for r in csv.DictReader(open(OUT))}
    todo = [(m, k, p, s) for m, k in SIZES for p in PS for s in range(SEEDS) if (m, k, p, s) not in done]
    todo.sort(key=lambda t: -(t[0] + t[1]))
    print(len(done), "done,", len(todo), "to do", flush=True)
    with open(OUT, "a", newline="") as f, Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for r in pool.imap_unordered(job, todo):
            w.writerow(r)
            f.flush()
            print(r["m"], r["k"], r["p"], r["seed"], r["seconds"], flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

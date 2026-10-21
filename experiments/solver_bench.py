"""Time, peak memory and accuracy of MOSEK (tolerance 1e-10) against Clarabel (default and 1e-10) on h1, h1_dnn
and kappa^(1), for one random graph per balanced size and the three largest kappa^(1) gaps of k1_check.csv.

Each solve runs in its own process so that its peak memory can be read. Writes solver_bench.csv in the data folder.
Run with: bipen run solver-bench --workers N
"""

import csv
import os
import resource
import time
import warnings
from multiprocessing import Pool

import cvxpy as cp
import numpy as np

from bipen import sdp
from bipen.brute import parameters
from bipen.graphs import random_bipartite
from experiments.random_graphs import seed

warnings.filterwarnings("ignore")
SETTINGS = [(cp.MOSEK, 1e-10), (cp.CLARABEL, None), (cp.CLARABEL, 1e-10)]
BOUNDS = ["h1", "h1_dnn", "k1"]


def graphs(data):
    out = [(m, m, 0.5, 0) for m in (6, 8, 10, 12, 15, 20)]
    rows = sorted(csv.DictReader(open(os.path.join(data, "k1_check.csv"))),
                  key=lambda r: -(float(r["primal"]) - float(r["half_sqrt_g"])) / float(r["half_sqrt_g"]))
    out += [(int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"])) for r in rows[:3]]
    return out


def job(args):
    (m, k, p, s), (solver, tol), bound = args
    sdp.SOLVER, sdp.TOL = solver, tol
    B = random_bipartite(m, k, p, np.random.default_rng(seed(m, k, p, s)))
    t = time.time()
    value = {"h1": lambda: sdp.h1(B), "h1_dnn": lambda: sdp.h1(B, dnn=True), "k1": lambda: sdp.cop_k1(B)}[bound]()
    seconds = round(time.time() - t, 2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB
    return {"m": m, "k": k, "p": p, "seed": s, "solver": solver.lower(), "tol": tol or "default", "bound": bound,
            "value": value, "half_sqrt_g": 0.5 * np.sqrt(parameters(B)["g"]), "seconds": seconds,
            "peak_gb": round(rss, 2)}


def main(data, workers=2):
    todo = [(g, st, b) for g in graphs(data) for st in SETTINGS for b in BOUNDS]
    todo.sort(key=lambda t: t[0][0] + t[0][1])
    with open(os.path.join(data, "solver_bench.csv"), "w", newline="") as f, Pool(workers, maxtasksperchild=1) as pool:
        w = None
        for r in pool.imap_unordered(job, todo):
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(r))
                w.writeheader()
            w.writerow(r)
            f.flush()
            print(*r.values(), flush=True)
    print("finished", flush=True)


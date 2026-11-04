"""MOSEK against Clarabel on h_1, h_1^D and rho^(1): time, memory and accuracy.

MOSEK at tolerance 1e-10, and Clarabel with its default tolerances and at 1e-10, for one random graph per balanced
size of Table 1 and the three largest rho^(1) gaps of bounds/inexact.
Each solve runs in its own process, so that its peak memory can be read.
Run with: bipen run bounds/solvers --workers N
"""

import resource
import time
import warnings
from multiprocessing import Pool

import cvxpy as cp
import numpy as np

from bipen import sdp
from bipen.data import Column, File, graph_key, read_csv, write_csv
from bipen.exact import biclique_number
from bipen.graphs import bounds_graph

warnings.filterwarnings("ignore")
SETTINGS = [(cp.MOSEK, 1e-10), (cp.CLARABEL, None), (cp.CLARABEL, 1e-10)]
BOUNDS = {
    Column.H1: lambda U: sdp.h1(U),
    Column.H1_DNN: lambda U: sdp.h1(U, dnn=True),
    Column.RHO1: lambda U: sdp.rho1(U),
}


def graphs(data):
    relative_gap = lambda row: (
        (float(row[Column.PRIMAL]) - float(row[Column.HALF_SQRT_G])) / float(row[Column.HALF_SQRT_G])
    )
    largest_gaps = sorted(read_csv(data / File.BOUNDS_INEXACT), key=relative_gap, reverse=True)[:3]
    return [(m, m, 0.5, 0) for m in (6, 8, 10, 12, 15, 20)] + [graph_key(row) for row in largest_gaps]


def job(task):
    (m1, m2, p, s), (solver, tol), bound = task
    sdp.SOLVER, sdp.TOL = solver, tol
    U = bounds_graph(m1, m2, p, s)
    start = time.time()
    value = BOUNDS[bound](U)
    seconds = round(time.time() - start, 2)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return {
        Column.M1: m1,
        Column.M2: m2,
        Column.P: p,
        Column.INDEX: s,
        Column.SOLVER: solver.lower(),
        Column.TOLERANCE: tol or "default",
        Column.BOUND: bound,
        Column.VALUE: value,
        Column.HALF_SQRT_G: 0.5 * np.sqrt(biclique_number(U)),
        Column.SECONDS: seconds,
        Column.PEAK_GB: round(peak_gb, 2),
    }


def main(data, workers=2):
    todo = [(graph, setting, bound) for graph in graphs(data) for setting in SETTINGS for bound in BOUNDS]
    todo.sort(key=lambda task: task[0][0] + task[0][1])

    def rows(pool):
        for row in pool.imap_unordered(job, todo):
            print(*row.values(), flush=True)
            yield row

    with Pool(workers, maxtasksperchild=1) as pool:
        write_csv(data / File.BOUNDS_SOLVERS, rows(pool))
    print("finished", flush=True)

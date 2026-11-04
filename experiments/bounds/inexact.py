"""rho^(1) solved again at tolerance 1e-10 where it is not exact (Section 6).

On every random graph of Table 1 where rho^(1) is not exact, with the primal and dual objectives and residuals, to
check that the gaps are not numerical artefacts. The dual objective is a lower bound on rho^(1) up to the dual
residual.
Run with: bipen run bounds/inexact --workers N
"""

import time
import warnings
from multiprocessing import Pool

import cvxpy as cp
import mosek
import numpy as np

from bipen import sdp
from bipen.data import Column, File, graph_key, read_csv, write_csv
from bipen.exact import biclique_number
from bipen.graphs import bounds_graph
from bipen.sdp import dual_matrix, parrilo_cone

warnings.filterwarnings("ignore")
EXACT = 1e-5
TOL = 1e-10


def non_exact(data):
    """The random graphs where rho^(1) exceeds sqrt(g)/2 by more than EXACT, largest first."""
    graphs = [
        graph_key(row)
        for row in read_csv(data / File.RANDOM_BOUNDS)
        if float(row[Column.RHO1]) - float(row[Column.HALF_SQRT_G]) > EXACT
    ]
    return sorted(graphs, key=lambda graph: -(graph[0] + graph[1]))


def solve_with_residuals(problem):
    """Status, primal and dual objectives and residuals (absolute violations for MOSEK)."""
    data, chain, _ = problem.get_problem_data(sdp.SOLVER)
    out = chain.solver.solve_via_data(data, False, False, sdp.solver_options(TOL))
    if sdp.SOLVER == cp.CLARABEL:
        return str(out.status), out.obj_val, out.obj_val_dual, out.r_prim, out.r_dual
    # cvxpy hands MOSEK the dual problem, so the roles of the two iterates are exchanged
    task = out["task"]
    info = task.getsolutioninfo(mosek.soltype.itr)
    status = str(task.getsolsta(mosek.soltype.itr)).removeprefix("solsta.")
    return status, info[6], info[0], max(info[7:11]), max(info[1:6])


def job(graph):
    m1, m2, p, s = graph
    U = bounds_graph(m1, m2, p, s)
    lam, M, n = dual_matrix(U)
    problem = cp.Problem(cp.Minimize(lam), parrilo_cone(M, n))
    start = time.time()
    status, primal, dual, primal_residual, dual_residual = solve_with_residuals(problem)
    return {
        Column.M1: m1,
        Column.M2: m2,
        Column.P: p,
        Column.SEED: s,
        Column.STATUS: status,
        Column.PRIMAL: primal,
        Column.DUAL: dual,
        Column.PRIMAL_RESIDUAL: primal_residual,
        Column.DUAL_RESIDUAL: dual_residual,
        Column.HALF_SQRT_G: 0.5 * np.sqrt(biclique_number(U)),
        Column.SECONDS: round(time.time() - start, 1),
    }


def main(data, workers=2):
    def rows(pool):
        for row in pool.imap_unordered(job, non_exact(data)):
            print(*row.values(), flush=True)
            yield row

    with Pool(workers) as pool:
        write_csv(data / File.RHO_GAPS, rows(pool))

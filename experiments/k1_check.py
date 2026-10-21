"""Re-solve kappa^(1) on every random graph of ../data/random.csv where it is not exact, at tolerance 1e-10,
and record the primal and dual objectives and residuals, to check that the gaps are not numerical artefacts.
The dual objective is a lower bound on kappa^(1) up to the dual residual.

Writes ../data/k1_check.csv. Run with: python -m experiments.k1_check <workers>
"""

import csv
import sys
import time
import warnings
from multiprocessing import Pool

import cvxpy as cp
import mosek
import numpy as np

from bicop import sdp
from bicop.brute import parameters
from bicop.graphs import random_bipartite
from bicop.sdp import _dual_matrix, k1_constraints
from experiments.random_graphs import seed

warnings.filterwarnings("ignore")
OUT = "../data/k1_check.csv"
EXACT = 1e-5
TOL = 1e-10


def non_exact():
    rows = csv.DictReader(open("../data/random.csv"))
    todo = [(int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"])) for r in rows
            if float(r["k1"]) - float(r["half_sqrt_g"]) > EXACT]
    return sorted(todo, key=lambda t: -(t[0] + t[1]))


def solve(prob):
    """Status, primal and dual objectives and residuals (absolute violations for MOSEK)."""
    data, chain, _ = prob.get_problem_data(sdp.SOLVER)
    out = chain.solver.solve_via_data(data, False, False, sdp.solver_opts(TOL))
    if sdp.SOLVER == cp.CLARABEL:
        return str(out.status), out.obj_val, out.obj_val_dual, out.r_prim, out.r_dual
    # cvxpy hands MOSEK the dual problem, so the roles of the two iterates are exchanged
    task = out["task"]
    info = task.getsolutioninfo(mosek.soltype.itr)
    status = str(task.getsolsta(mosek.soltype.itr)).removeprefix("solsta.")
    return status, info[6], info[0], max(info[7:11]), max(info[1:6])


def job(args):
    m, k, p, s = args
    B = random_bipartite(m, k, p, np.random.default_rng(seed(m, k, p, s)))
    lam, M, n = _dual_matrix(B)
    prob = cp.Problem(cp.Minimize(lam), k1_constraints(M, n))
    t = time.time()
    status, primal, dual, r_prim, r_dual = solve(prob)
    return (m, k, p, s, status, primal, dual, r_prim, r_dual, 0.5 * np.sqrt(parameters(B)["g"]),
            round(time.time() - t, 1))


def main(workers):
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "k", "p", "seed", "status", "primal", "dual", "r_prim", "r_dual", "half_sqrt_g", "seconds"])
        with Pool(workers) as pool:
            for row in pool.imap_unordered(job, non_exact()):
                w.writerow(row)
                f.flush()
                print(*row, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

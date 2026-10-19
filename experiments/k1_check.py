"""Re-solve kappa^(1) on the random graphs where it is not exact with Clarabel at tolerance 1e-10, and
record the primal and dual objectives and residuals, to check that the gaps are not numerical artefacts.
The dual objective is a lower bound on kappa^(1) up to the dual residual.

Results are appended to ../data/k1_check.csv. Run with: python -m experiments.k1_check <workers>
"""

import csv
import sys
import time
import warnings
from multiprocessing import Pool

import cvxpy as cp
import numpy as np

from bicop.brute import parameters
from bicop.graphs import random_bipartite
from bicop.sdp import _dual_matrix, k1_constraints

warnings.filterwarnings("ignore")
OUT = "../data/k1_check.csv"
GRAPHS = [(10, 0.5, 7), (12, 0.5, 3), (20, 0.3, 2), (20, 0.5, 7)]  # (m, p, seed) with a gap above 1e-5
TOL = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}


def job(args):
    m, p, s = args
    B = random_bipartite(m, m, p, np.random.default_rng(1000 * m + 100 * int(10 * p) + s))
    lam, M, n = _dual_matrix(B)
    prob = cp.Problem(cp.Minimize(lam), k1_constraints(M, n))
    data, chain, _ = prob.get_problem_data(cp.CLARABEL)
    t = time.time()
    sol = chain.solver.solve_via_data(data, False, False, TOL)
    return (m, p, s, str(sol.status), sol.obj_val, sol.obj_val_dual, sol.r_prim, sol.r_dual,
            0.5 * np.sqrt(parameters(B)["g"]), round(time.time() - t, 1))


def main(workers):
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "p", "seed", "status", "primal", "dual", "r_prim", "r_dual", "half_sqrt_g", "seconds"])
        with Pool(workers) as pool:
            for row in pool.imap_unordered(job, GRAPHS):
                w.writerow(row)
                f.flush()
                print(*row, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

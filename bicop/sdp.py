"""Conic bounds on sqrt(g)/2 for a bipartite graph with biadjacency B.

h1:      max <C,X>  s.t. X PSD, tr X = 1, X_ij = 0 (ij in E)              (LPV (9))
h1_dnn:  the same with X >= 0                                          (= T(G)/2, LPV Rem. 3.4)
cop_k1:  min lambda s.t. lambda I - C + sum_E y_ij (E_ij + E_ji) in K^(1)
         (Parrilo's cone, LMIs of Bomze-de Klerk 2002, Thm 2.3)
cop_c1:  the same with the Polya LP cone C^(1) (Bomze-de Klerk Thm 2.4)
"""

import itertools
import os

import cvxpy as cp
import numpy as np

from bicop.graphs import cost_matrix, edges

# MOSEK unless BICOP_SOLVER=clarabel. MOSEK's default tolerances can stop short of the optimum while reporting it
# optimal, so they are tightened; TOL = None keeps Clarabel's defaults.
SOLVER = cp.CLARABEL if os.environ.get("BICOP_SOLVER") == "clarabel" else cp.MOSEK
TOL = None


def solver_opts(tol):
    if SOLVER == cp.CLARABEL:
        return {} if tol is None else {"tol_gap_abs": tol, "tol_gap_rel": tol, "tol_feas": tol}
    tol = tol or 1e-10
    params = {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol, "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
              "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol, "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
              "MSK_DPAR_INTPNT_TOL_PFEAS": tol, "MSK_DPAR_INTPNT_TOL_DFEAS": tol, "MSK_IPAR_NUM_THREADS": 2}
    return {"mosek_params": params, "accept_unknown": True}


def _solve(prob):
    prob.solve(solver=SOLVER, **solver_opts(TOL))
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return np.inf if prob.status == "infeasible" else np.nan
    return prob.value


def h1(B, dnn=False):
    m, k = B.shape
    n = m + k
    X = cp.Variable((n, n), PSD=True)
    cons = [cp.trace(X) == 1] + [X[i, j] == 0 for i, j in edges(B)]
    if dnn:
        cons.append(X >= 0)
    return _solve(cp.Problem(cp.Maximize(cp.trace(cost_matrix(m, k) @ X)), cons))


def _dual_matrix(B):
    m, k = B.shape
    n = m + k
    lam = cp.Variable()
    E = edges(B)
    y = cp.Variable(len(E))
    P = np.zeros((len(E), n * n))
    for t, (i, j) in enumerate(E):
        P[t, i * n + j] = P[t, j * n + i] = 1
    Y = cp.reshape(P.T @ y, (n, n), order="C") if E else 0
    return lam, lam * np.eye(n) - cost_matrix(m, k) + Y, n


def k1_constraints(M, n):
    """M in K^(1) (Bomze-de Klerk 2002, Thm 2.3)."""
    Ms = [cp.Variable((n, n), symmetric=True) for _ in range(n)]
    cons = []
    for i in range(n):
        cons += [M - Ms[i] >> 0, Ms[i][i, i] == 0]
    for i in range(n):
        for j in range(n):
            if i != j:
                cons.append(Ms[j][i, i] + 2 * Ms[i][i, j] == 0)
    for i, j, l in itertools.combinations(range(n), 3):
        cons.append(Ms[i][j, l] + Ms[j][i, l] + Ms[l][i, j] >= 0)
    return cons


def cop_k1(B):
    lam, M, n = _dual_matrix(B)
    return _solve(cp.Problem(cp.Minimize(lam), k1_constraints(M, n)))


def cop_c1(B):
    lam, M, n = _dual_matrix(B)
    cons = [cp.diag(M) >= 0]
    cons += [M[j, j] + 2 * M[i, j] >= 0 for i in range(n) for j in range(n) if i != j]
    cons += [M[i, j] + M[i, l] + M[j, l] >= 0 for i, j, l in itertools.combinations(range(n), 3)]
    return _solve(cp.Problem(cp.Minimize(lam), cons))

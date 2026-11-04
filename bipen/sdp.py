"""The semidefinite bounds of the paper, all upper bounds on sqrt(g)/2 except the Lasserre bound, which bounds g.

h1(U):             h_1 = max { <C, X> : X PSD, tr X = 1, X_ij = 0 for ij in E }, the first bound of Laurent, Polak
                   and Vargas (LPV)
h1(U, dnn=True):   h_1^D, the same with X >= 0 (doubly nonnegative)
rho1(U):           rho^(1) = min { lambda : lambda I - C + Y in R^(1) }, the copositive dual with Parrilo's cone R^(1),
                   where Y = sum over the edges ij of w_ij S_ij
lasserre_g(U, r):  g_r >= g, the Lasserre bound of LPV (29), in moment form
"""

import itertools
import os

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from bipen.graphs import adjacency, cost_matrix, edges

# MOSEK unless BIPEN_SOLVER=clarabel. MOSEK's default tolerances can stop short of the optimum while reporting it
# optimal, so they are tightened; TOL = None keeps Clarabel's defaults.
SOLVER = cp.CLARABEL if os.environ.get("BIPEN_SOLVER") == "clarabel" else cp.MOSEK
TOL = None
MOSEK_TOL = 1e-10


def solver_options(tol):
    if SOLVER == cp.CLARABEL:
        return {} if tol is None else {"tol_gap_abs": tol, "tol_gap_rel": tol, "tol_feas": tol}
    tol = tol or MOSEK_TOL
    params = {
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": tol,
        "MSK_DPAR_INTPNT_TOL_PFEAS": tol,
        "MSK_DPAR_INTPNT_TOL_DFEAS": tol,
        "MSK_IPAR_NUM_THREADS": 2,
    }
    return {"mosek_params": params, "accept_unknown": True}


def solve(problem):
    """The optimal value, inf if the problem is infeasible, and nan if the solver fails."""
    problem.solve(solver=SOLVER, **solver_options(TOL))
    if problem.status not in ("optimal", "optimal_inaccurate"):
        return np.inf if problem.status == "infeasible" else np.nan
    return problem.value


def h1(U, dnn=False):
    m1, m2 = U.shape
    n = m1 + m2
    X = cp.Variable((n, n), PSD=True)
    constraints = [cp.trace(X) == 1] + [X[i, j] == 0 for i, j in edges(U)]
    if dnn:
        constraints.append(X >= 0)
    return solve(cp.Problem(cp.Maximize(cp.trace(cost_matrix(m1, m2) @ X)), constraints))


def dual_matrix(U):
    """The matrix M = lambda I - C + Y of the copositive dual, with the variables lambda and w_ij (ij in E), where
    Y = sum of w_ij S_ij and S_ij has ones at (i, j) and (j, i). Returns lambda, M and n."""
    m1, m2 = U.shape
    n = m1 + m2
    lam = cp.Variable()
    E = edges(U)
    if not E:
        return lam, lam * np.eye(n) - cost_matrix(m1, m2), n
    w = cp.Variable(len(E))
    # Y as one linear map of w, with S_ij flattened row by row as the column of ij
    S = np.zeros((len(E), n * n))
    for t, (i, j) in enumerate(E):
        S[t, i * n + j] = S[t, j * n + i] = 1
    Y = cp.reshape(S.T @ w, (n, n), order="C")
    return lam, lam * np.eye(n) - cost_matrix(m1, m2) + Y, n


def parrilo_cone(M, n):
    """The constraints M in R^(1), Parrilo's first cone, by the linear matrix inequalities (13)-(16) of de Klerk and
    Pasechnik: there are symmetric matrices M_1, ..., M_n with M - M_i PSD, (M_i)_ii = 0,
    (M_j)_ii + 2 (M_i)_ij = 0 for i != j, and (M_i)_jl + (M_j)_il + (M_l)_ij >= 0 for i < j < l."""
    Ms = [cp.Variable((n, n), symmetric=True) for _ in range(n)]
    constraints = []
    for i in range(n):
        constraints += [M - Ms[i] >> 0, Ms[i][i, i] == 0]
    for i in range(n):
        for j in range(n):
            if i != j:
                constraints.append(Ms[j][i, i] + 2 * Ms[i][i, j] == 0)
    for i, j, l in itertools.combinations(range(n), 3):
        constraints.append(Ms[i][j, l] + Ms[j][i, l] + Ms[l][i, j] >= 0)
    return constraints


def rho1(U):
    lam, M, n = dual_matrix(U)
    return solve(cp.Problem(cp.Minimize(lam), parrilo_cone(M, n)))


def lasserre_g(U, r=2):
    """g_r(G): the maximum of the sum of y_{ij} over i in V1 and j in V2, over moment vectors y indexed by the stable
    sets of G, with y_{} = 1 and the moment matrix PSD; its rows and columns are the stable sets of size at most r, its
    entry at (S, T) is y_{S ∪ T}, and y vanishes on the sets that contain an edge."""
    m1, m2 = U.shape
    W = adjacency(U)
    is_stable = lambda T: not any(W[i, j] for i, j in itertools.combinations(T, 2))
    rows = [T for size in range(r + 1) for T in itertools.combinations(range(m1 + m2), size) if is_stable(T)]
    # the stable unions S ∪ T get one variable each; the entry (a, b) of the moment matrix, flattened as
    # a * len(rows) + b, is the variable of its union
    index, entries = {}, []
    for a, S in enumerate(rows):
        for b, T in enumerate(rows):
            union = tuple(sorted(set(S) | set(T)))
            if is_stable(union):
                entries.append((a * len(rows) + b, index.setdefault(union, len(index))))
    y = cp.Variable(len(index))
    placement = sp.coo_matrix((np.ones(len(entries)), np.array(entries).T), shape=(len(rows) ** 2, len(index)))
    moments = cp.reshape(placement @ y, (len(rows), len(rows)), order="C")
    objective = sum(y[index[(i, j)]] for i in range(m1) for j in range(m1, m1 + m2) if (i, j) in index)
    return solve(cp.Problem(cp.Maximize(objective), [moments >> 0, y[index[()]] == 1]))

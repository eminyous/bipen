"""The penalty formulation: its exact value and smallest valid penalty for small graphs, and the local ascent of
Section 6.

The exact value is mu = max { u^T Q u : u >= 0, |u| = 1 } for Q = C - gamma W, as in Appendix C. Every maximizer u is
a positive eigenvector of Q_S on its support S, for the eigenvalue mu = lambda_max(Q_S); conversely, a positive
eigenvector of a principal submatrix Q_S is feasible, with its eigenvalue as value. Hence mu is the largest
eigenvalue, over the supports S, of some Q_S whose eigenspace contains a positive vector, which an LP decides exactly,
also for repeated eigenvalues.
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from bipen.exact import biclique_number
from bipen.graphs import adjacency, cost_matrix


def contains_positive_vector(V):
    """Whether the span of the columns of V contains a positive vector, that is, V c >= 1 for some c."""
    k = V.shape[1]
    res = linprog(np.zeros(k), A_ub=-V, b_ub=-np.ones(V.shape[0]), bounds=[(None, None)] * k, method="highs")
    return res.status == 0


def max_on_nonnegative_sphere(Q, tol=1e-9, return_support=False):
    """max { u^T Q u : u >= 0, |u| = 1 }, and the support S of a maximizer if return_support."""
    n = Q.shape[0]
    best, best_support = -np.inf, None
    for size in range(1, n + 1):
        for S in itertools.combinations(range(n), size):
            eigenvalues, eigenvectors = np.linalg.eigh(Q[np.ix_(S, S)])  # eigenvalues in increasing order
            top = len(eigenvalues) - 1
            # the eigenvalues above the best value so far, from the largest down, one eigenspace at a time
            while top >= 0 and eigenvalues[top] > best + tol:
                bottom = top
                while bottom > 0 and abs(eigenvalues[bottom - 1] - eigenvalues[top]) < 1e-9:
                    bottom -= 1
                if contains_positive_vector(eigenvectors[:, bottom : top + 1]):
                    best, best_support = eigenvalues[top], S
                    break
                top = bottom - 1
    return (best, best_support) if return_support else best


def best_response(a):
    """The nonnegative unit vector v that maximizes a^T v: a_+ / |a_+|, or the vertex e_i of the largest entry a_i
    when a <= 0."""
    if a.max() <= 0:
        return np.eye(len(a))[np.argmax(a)]
    a = np.maximum(a, 0)
    return a / np.linalg.norm(a)


def ascent(H, y, z, iters=10000, tol=1e-12):
    """Block-coordinate ascent for max y^T H z over nonnegative unit vectors y and z: each step maximizes exactly over
    one side, y = best_response(H z), then z = best_response(H^T y). Stops when a step gains less than tol.
    Returns y, z, the value and the number of steps."""
    value = y @ H @ z
    for step in range(1, iters + 1):
        y = best_response(H @ z)
        z = best_response(H.T @ y)
        new_value = y @ H @ z
        if new_value - value < tol:
            return y, z, new_value, step
        value = new_value
    return y, z, value, iters


OPTIMAL, EDGE, MAXIMAL, NONMAXIMAL = "optimal", "edge", "maximal", "nonmaximal"  # the ends of a run


def end_state(U, g, y, z, tol=1e-9):
    """How an ascent run ends, from the supports S of y and R of z: OPTIMAL if (S, R) is biindependent with
    |S||R| = g, EDGE if an edge joins S and R, MAXIMAL if (S, R) is a maximal biindependent pair that is not maximum,
    and NONMAXIMAL otherwise."""
    S, R = y > tol, z > tol
    if U[np.ix_(S, R)].any():
        return EDGE
    if S.sum() * R.sum() == g:
        return OPTIMAL
    extendable_in_V1 = (~S & ~U[:, R].any(axis=1)).any()
    extendable_in_V2 = (~R & ~U[S, :].any(axis=0)).any()
    return NONMAXIMAL if extendable_in_V1 or extendable_in_V2 else MAXIMAL


def repair(U, y, z, tol=1e-9):
    """Greedy repair of the supports S of y and R of z. First, while an edge joins S and R, drop the vertex with the
    most neighbours on the other side. Then, until (S, R) is maximal, add a vertex of smallest degree that keeps the
    pair biindependent, on the side where it increases |S||R| most. Returns |S||R|."""
    S, R = y > tol, z > tol
    while U[np.ix_(S, R)].any():
        neighbours_in_R = U[:, R].sum(axis=1) * S
        neighbours_in_S = U[S, :].sum(axis=0) * R
        if neighbours_in_R.max() >= neighbours_in_S.max():
            S[np.argmax(neighbours_in_R)] = False
        else:
            R[np.argmax(neighbours_in_S)] = False
    while True:
        addable_in_V1 = np.flatnonzero(~S & ~U[:, R].any(axis=1))
        addable_in_V2 = np.flatnonzero(~R & ~U[S, :].any(axis=0))
        if len(addable_in_V1) == 0 and len(addable_in_V2) == 0:
            return int(S.sum() * R.sum())
        # adding a vertex to S increases |S||R| by |R|, and adding one to R increases it by |S|
        if len(addable_in_V2) == 0 or (len(addable_in_V1) > 0 and R.sum() >= S.sum()):
            S[addable_in_V1[np.argmin(U[addable_in_V1].sum(axis=1))]] = True
        else:
            R[addable_in_V2[np.argmin(U[:, addable_in_V2].sum(axis=0))]] = True


BISECTION_TOL = 1e-7


def smallest_penalty(U, tol=BISECTION_TOL):
    """gamma*(G), the smallest gamma with max { u^T (C - gamma W) u : u >= 0, |u| = 1 } = sqrt(g)/2, by bisection
    (Section 6), returned as the upper end of the final interval."""
    m1, m2 = U.shape
    C, W = cost_matrix(m1, m2), adjacency(U)
    target = 0.5 * np.sqrt(biclique_number(U))
    is_valid = lambda gamma: max_on_nonnegative_sphere(C - gamma * W) <= target + 1e-9
    low, high = 0.0, 1.0
    while not is_valid(high):
        low, high = high, 2 * high
    while high - low > tol:
        middle = (low + high) / 2
        low, high = (low, middle) if is_valid(middle) else (middle, high)
    return high

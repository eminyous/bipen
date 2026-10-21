"""Exact value of max { u^T Q u : u >= 0, |u| = 1 } for small n.

Every maximiser is a KKT point: on its support S it is a positive eigenvector of Q_S.
Conversely each positive eigenvector of a principal submatrix is feasible with value its
eigenvalue.  So the maximum is the largest eigenvalue lambda of some Q_S whose eigenspace
contains a positive vector; we test that with an LP (exact also for repeated eigenvalues).
"""

import itertools

import numpy as np
from scipy.optimize import linprog


def _has_positive_vector(V):
    # exists c with V c >= 1 ?
    p = V.shape[1]
    res = linprog(np.zeros(p), A_ub=-V, b_ub=-np.ones(V.shape[0]), bounds=[(None, None)] * p, method="highs")
    return res.status == 0


def max_on_nonneg_sphere(Q, tol=1e-9, return_support=False):
    n = Q.shape[0]
    best, best_S = -np.inf, None
    for r in range(1, n + 1):
        for S in itertools.combinations(range(n), r):
            w, V = np.linalg.eigh(Q[np.ix_(S, S)])
            i = len(w) - 1
            while i >= 0 and w[i] > best + tol:
                j = i
                while j > 0 and abs(w[j - 1] - w[i]) < 1e-9:
                    j -= 1
                if _has_positive_vector(V[:, j:i + 1]):
                    best, best_S = w[i], S
                    break
                i = j - 1
    return (best, best_S) if return_support else best


def ascent(M, x, y, iters=10000, tol=1e-12):
    """Block-coordinate ascent for max x^T M y over nonnegative unit vectors: each step maximizes over one side,
    x = (M y)_+ / |(M y)_+|, or the best vertex e_i when (M y)_+ = 0. Returns x, y, value, steps."""

    def best(a):
        if a.max() <= 0:
            return np.eye(len(a))[np.argmax(a)]
        a = np.maximum(a, 0)
        return a / np.linalg.norm(a)

    val = x @ M @ y
    for t in range(iters):
        x = best(M @ y)
        y = best(M.T @ x)
        new = x @ M @ y
        if new - val < tol:
            return x, y, new, t + 1
        val = new
    return x, y, val, iters

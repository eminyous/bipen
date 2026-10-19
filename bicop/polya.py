"""Polya LP cone C^(r) (Bomze-de Klerk 2002, Thm 2.4) applied to the copositive dual

    zeta^(r)(G) = min { lambda : lambda I - C + sum_E y_ij (E_ij + E_ji) in C^(r) },

with C^(r) = { M : m^T M m - m^T diag(M) >= 0 for all m in N^n with |m| = r + 2 }.
"""

import itertools

import cvxpy as cp
import numpy as np

from bicop.sdp import _dual_matrix, _solve


def multi_indices(n, s):
    for c in itertools.combinations_with_replacement(range(n), s):
        m = np.zeros(n)
        for i in c:
            m[i] += 1
        yield m


def zeta(B, r):
    lam, M, n = _dual_matrix(B)
    cons = [m @ M @ m - m @ cp.diag(M) >= 0 for m in multi_indices(n, r + 2)]
    return _solve(cp.Problem(cp.Minimize(lam), cons))

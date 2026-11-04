"""The Pólya bounds of Section 5: the copositive dual with the Pólya cone P^(r) of Bomze and de Klerk,

    zeta^(r)(G) = min { lambda : lambda I - C + Y in P^(r) },

where P^(r) = { M : nu^T M nu - nu^T diag(M) >= 0 for all nu in N^n with |nu| = r + 2 }; zeta^(r) is inf when the
program is infeasible.
"""

import itertools

import cvxpy as cp
import numpy as np

from bipen.sdp import dual_matrix, solve


def multi_indices(n, size):
    """The vectors nu in N^n with |nu| = size."""
    for support in itertools.combinations_with_replacement(range(n), size):
        nu = np.zeros(n)
        for i in support:
            nu[i] += 1
        yield nu


def zeta(U, r):
    lam, M, n = dual_matrix(U)
    constraints = [nu @ M @ nu - nu @ cp.diag(M) >= 0 for nu in multi_indices(n, r + 2)]
    return solve(cp.Problem(cp.Minimize(lam), constraints))

"""The DSOS and SDSOS bounds of Appendix B: the copositive dual with the cones of Ahmadi and Majumdar,

    zeta~^(r)(G) = min { lambda : lambda I - C + Y in P~^(r) }   (DSOS),
    rho~^(r)(G)  = min { lambda : lambda I - C + Y in R~^(r) }   (SDSOS),

where M is in P~^(r) (in R~^(r)) when (sum_i x_i^2)^r p_M(x) = [x]_d^T Gamma [x]_d for some diagonally dominant
matrix Gamma (some sum of 2 x 2 blocks), with p_M(x) = sum_ij M_ij x_i^2 x_j^2, d = r + 2 and [x]_d the vector of
the monomials x^nu of degree d.

The form (sum_i x_i^2)^r p_M has the coefficient (r!/nu!) (nu^T M nu - nu^T diag M) at x^(2 nu), as in polya.py, and
no other monomials. An entry Gamma_{nu, nu'} only enters the coefficient at x^(nu + nu'), which is of the form
x^(2 nu'') only when nu and nu' have the same parity in every coordinate; the other entries are set to 0, which keeps
Gamma diagonally dominant (a sum of 2 x 2 blocks).
"""

import itertools
import math
from collections import defaultdict

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from bipen.graphs import edges
from bipen.polya import multi_indices
from bipen.sdp import solve


def zeta_tilde(U, r):
    return _bound(U, r, scaled=False)


def rho_tilde(U, r):
    return _bound(U, r, scaled=True)


def _bound(U, r, scaled):
    m1, m2 = U.shape
    n, d = m1 + m2, r + 2
    nus = np.array(list(multi_indices(n, d)), dtype=int)
    position = {tuple(nu): t for t, nu in enumerate(nus)}

    # the off-diagonal entries Gamma_{nu, nu'} that can be nonzero: nu and nu' of the same parity
    by_parity = defaultdict(list)
    for t, nu in enumerate(nus):
        by_parity[tuple(nu % 2)].append(t)
    pairs = [pair for group in by_parity.values() for pair in itertools.combinations(group, 2)]
    N, P = len(nus), len(pairs)
    first, second = np.array(pairs).T
    middle = [position[tuple((nus[s] + nus[t]) // 2)] for s, t in pairs]  # the entry (s, t) enters x^(2 * middle)
    ones = np.ones(P)
    first_of = sp.csr_matrix((ones, (first, range(P))), shape=(N, P))
    second_of = sp.csr_matrix((ones, (second, range(P))), shape=(N, P))
    middle_of = sp.csr_matrix((ones, (middle, range(P))), shape=(N, P))

    # the coefficient of x^(2 nu) in (sum_i x_i^2)^r p_M for M = lambda I - C + Y, for every nu in nus
    E = edges(U)
    factor = np.array([math.factorial(r) / np.prod([math.factorial(v) for v in nu]) for nu in nus])
    lam = cp.Variable()
    coefficient = factor * ((nus**2).sum(1) - d) * lam - factor * nus[:, :m1].sum(1) * nus[:, m1:].sum(1)
    if E:
        w = cp.Variable(len(E))
        coefficient = coefficient + np.array([[2 * f * nu[i] * nu[j] for i, j in E] for f, nu in zip(factor, nus)]) @ w

    # Gamma by its diagonal and its entries on the pairs; the entry (s, t) and its mirror both enter x^(2 * middle)
    diagonal, off_diagonal = cp.Variable(N), cp.Variable(P)
    constraints = [diagonal + 2 * (middle_of @ off_diagonal) == coefficient]
    if scaled:
        # a sum of 2 x 2 blocks [[a, c], [c, b]] with 4 c^2 <= (a + b)^2 - (a - b)^2, one block per pair
        a, b = cp.Variable(P), cp.Variable(P)
        constraints += [
            cp.SOC(a + b, cp.vstack([2 * off_diagonal, a - b]), axis=0),
            diagonal >= first_of @ a + second_of @ b,
        ]
    else:
        constraints.append(diagonal >= (first_of + second_of) @ cp.abs(off_diagonal))
    return solve(cp.Problem(cp.Minimize(lam), constraints))

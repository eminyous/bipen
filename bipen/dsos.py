"""DSOS and SDSOS bounds (Ahmadi-Majumdar 2019) applied to the copositive dual

    zeta_L^(r)(G) = min { lambda : lambda I - C + sum_E y_ij (E_ij + E_ji) in L^(r) },
    L^(r) = { M : (sum_i x_i^2)^r p_M = m^T Z m with Z diagonally dominant },

with p_M(x) = sum_ij M_ij x_i^2 x_j^2 and m the vector of monomials of degree r + 2; zeta_R^(r) (scaled=True) asks
for Z scaled diagonally dominant, a sum of PSD matrices on 2 x 2 principal blocks. The form has the coefficient
(r!/nu!)(nu^T M nu - nu^T diag M) at x^(2 nu), as in polya.py, and no other monomials, so the entries Z_ab with
a + b odd somewhere only enter zero coefficients and are set to 0, which keeps Z (scaled) diagonally dominant.
"""

import itertools
import math
from collections import defaultdict

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from bipen.graphs import edges
from bipen.polya import multi_indices
from bipen.sdp import _solve


def zeta(B, r, scaled=False):
    m, k = B.shape
    n, d = m + k, r + 2
    mons = np.array(list(multi_indices(n, d)), dtype=int)
    index = {tuple(a): t for t, a in enumerate(mons)}
    classes = defaultdict(list)
    for t, a in enumerate(mons):
        classes[tuple(a % 2)].append(t)
    pairs = [p for c in classes.values() for p in itertools.combinations(c, 2)]
    N, P = len(mons), len(pairs)
    a, b = np.array(pairs).T
    centre = [index[tuple((mons[s] + mons[t]) // 2)] for s, t in pairs]
    ones = np.ones(P)
    left = sp.csr_matrix((ones, (a, range(P))), shape=(N, P))
    right = sp.csr_matrix((ones, (b, range(P))), shape=(N, P))
    centres = sp.csr_matrix((ones, (centre, range(P))), shape=(N, P))

    # coefficient of x^(2 nu) in (sum x_i^2)^r p_M for M = lambda I - C + Y, with nu running over mons
    E = edges(B)
    fac = np.array([math.factorial(r) / np.prod([math.factorial(v) for v in nu]) for nu in mons])
    lam = cp.Variable()
    coef = fac * ((mons ** 2).sum(1) - d) * lam - fac * mons[:, :m].sum(1) * mons[:, m:].sum(1)
    if E:
        y = cp.Variable(len(E))
        coef = coef + np.array([[2 * f * nu[i] * nu[j] for i, j in E] for f, nu in zip(fac, mons)]) @ y

    diag, z = cp.Variable(N), cp.Variable(P)
    cons = [diag + 2 * (centres @ z) == coef]
    if scaled:
        s, t = cp.Variable(P), cp.Variable(P)
        cons += [cp.SOC(s + t, cp.vstack([2 * z, s - t]), axis=0), diag >= left @ s + right @ t]
    else:
        cons.append(diag >= (left + right) @ cp.abs(z))
    return _solve(cp.Problem(cp.Minimize(lam), cons))

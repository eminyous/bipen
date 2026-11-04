import itertools

import cvxpy as cp
import numpy as np

from bipen.dsos import rho_tilde, zeta_tilde
from bipen.exact import biclique_number, biclique_ratio
from bipen.graphs import adjacency, cost_matrix, crown, matching, penalty_matrix, random_bipartite
from bipen.penalty import EDGE, NONMAXIMAL, OPTIMAL, ascent, end_state, max_on_nonnegative_sphere, repair
from bipen.polya import zeta
from bipen.sdp import h1, lasserre_g, parrilo_cone, rho1, solve


def naive_biclique_number(U):
    m1, m2 = U.shape
    best = 0
    for mask in range(1 << m1):
        A = [i for i in range(m1) if mask >> i & 1]
        B = [j for j in range(m2) if not any(U[i, j] for i in A)]
        best = max(best, len(A) * len(B))
    return best


def test_biclique_number_matches_naive_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(30):
        U = random_bipartite(5, 6, 0.5, rng)
        assert biclique_number(U) == naive_biclique_number(U)


def test_crown():
    U = crown(6)
    assert biclique_number(U) == 1 and biclique_ratio(U) == 0.5


def test_large_penalty_gives_half_sqrt_g():
    rng = np.random.default_rng(1)
    for _ in range(10):
        U = random_bipartite(4, 4, 0.5, rng)
        Q = cost_matrix(4, 4) - 10 * adjacency(U)
        assert abs(max_on_nonnegative_sphere(Q) - 0.5 * np.sqrt(biclique_number(U))) < 1e-8


def test_parrilo_cone_on_the_five_cycle():
    # theta^(1)(C_5) = 2 (de Klerk and Pasechnik 2002, Example 4.1)
    n = 5
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1
    t = cp.Variable()
    M = t * (np.eye(n) + W) - np.ones((n, n))
    assert abs(solve(cp.Problem(cp.Minimize(t), parrilo_cone(M, n))) - 2) < 1e-5


def test_h1_dnn_and_rho1_are_decreasing_upper_bounds():
    rng = np.random.default_rng(3)
    for _ in range(5):
        U = random_bipartite(4, 5, 0.5, rng)
        half_sqrt_g = 0.5 * np.sqrt(biclique_number(U))
        assert h1(U) >= h1(U, dnn=True) - 1e-6 >= rho1(U) - 2e-6 >= half_sqrt_g - 3e-6


def test_lasserre_bound_is_exact_at_order_alpha():
    U = matching(3)  # g = 2, alpha = 3
    assert lasserre_g(U, 2) >= 2 - 1e-6
    assert abs(lasserre_g(U, 3) - 2) < 1e-6


def test_ascent_stays_at_an_optimal_pair():
    U = matching(3)  # an optimal pair: ({v11}, {v22, v23})
    y, z = np.array([1.0, 0, 0]), np.array([0, 1.0, 1.0]) / np.sqrt(2)
    *_, value, _ = ascent(penalty_matrix(U, (1 + np.sqrt(2)) / 2), y, z)
    assert abs(value - np.sqrt(2)) < 1e-9


def test_end_state_and_repair():
    U = np.array([[1, 0], [0, 0]], dtype=bool)  # one edge on 2 + 2 vertices, g = 2
    y, z = np.array([1.0, 0]), np.array([1.0, 1.0]) / np.sqrt(2)
    assert end_state(U, 2, y, z) == EDGE
    assert repair(U, y, z) == 2
    assert end_state(U, 2, np.array([0, 1.0]), z) == OPTIMAL
    assert end_state(U, 2, np.array([0, 1.0]), np.array([1.0, 0])) == NONMAXIMAL


def test_dsos_and_sdsos_at_level_0():
    # Propositions B.3 and B.4: zeta~^(0) is half the maximum degree of the bipartite complement, and rho~^(0) lies
    # between half the spectral norm of J - U and zeta~^(0)
    rng = np.random.default_rng(4)
    for _ in range(5):
        U = random_bipartite(3, 4, 0.5, rng)
        max_degree = max((1 - U).sum(0).max(), (1 - U).sum(1).max())
        dsos, sdsos = zeta_tilde(U, 0), rho_tilde(U, 0)
        assert abs(dsos - max_degree / 2) < 1e-6
        assert 0.5 * np.linalg.norm(1 - U, 2) - 1e-6 <= sdsos <= dsos + 1e-6


def test_dsos_and_sdsos_between_polya_and_parrilo():
    # Proposition B.2 at level 1
    rng = np.random.default_rng(5)
    for _ in range(3):
        U = random_bipartite(2, 3, 0.5, rng)
        chain = [zeta(U, 1), zeta_tilde(U, 1), rho_tilde(U, 1), rho1(U), 0.5 * np.sqrt(biclique_number(U))]
        assert all(a >= b - 1e-6 for a, b in itertools.pairwise(chain))

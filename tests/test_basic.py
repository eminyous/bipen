import numpy as np

from bipen.brute import parameters
from bipen.graphs import cost_matrix, full_adjacency, knn_minus_matching, random_bipartite
from bipen.penalty import max_on_nonneg_sphere


def brute_g(B):
    m, k = B.shape
    best = 0
    for a in range(1 << m):
        A = [i for i in range(m) if a >> i & 1]
        free = [j for j in range(k) if not any(B[i, j] for i in A)]
        best = max(best, len(A) * len(free))
    return best


def test_g_matches_naive():
    rng = np.random.default_rng(0)
    for _ in range(30):
        B = random_bipartite(5, 6, 0.5, rng)
        assert parameters(B)["g"] == brute_g(B)


def test_knn_minus_matching():
    p = parameters(knn_minus_matching(6))
    assert p["g"] == 1 and p["h"] == 0.5


def test_penalty_large_d_gives_half_sqrt_g():
    rng = np.random.default_rng(1)
    for _ in range(10):
        B = random_bipartite(4, 4, 0.5, rng)
        Q = cost_matrix(4, 4) - 10 * full_adjacency(B)
        assert abs(max_on_nonneg_sphere(Q) - 0.5 * np.sqrt(parameters(B)["g"])) < 1e-8


def test_k1_matches_de_klerk_pasechnik_c5():
    # theta^(1)(C5) = 2 (de Klerk-Pasechnik 2002, Example 4.1)
    import cvxpy as cp

    from bipen.sdp import _solve, k1_constraints

    n = 5
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    t = cp.Variable()
    M = t * (np.eye(n) + A) - np.ones((n, n))
    assert abs(_solve(cp.Problem(cp.Minimize(t), k1_constraints(M, n))) - 2) < 1e-5


def test_dnn_and_k1_are_upper_bounds():
    from bipen.sdp import cop_k1, h1

    rng = np.random.default_rng(3)
    for _ in range(5):
        B = random_bipartite(4, 5, 0.5, rng)
        t = 0.5 * np.sqrt(parameters(B)["g"])
        assert h1(B) >= h1(B, dnn=True) - 1e-6 >= cop_k1(B) - 2e-6 >= t - 3e-6


def test_lasserre_bound_is_exact_at_order_alpha():
    from bipen.sdp import g_las

    B = np.eye(3)  # M_3: g = 2, alpha = 3
    assert g_las(B, 2) >= 2 - 1e-6
    assert abs(g_las(B, 3) - 2) < 1e-6


def test_ascent_stays_at_an_optimal_pair():
    from bipen.penalty import ascent

    B = np.eye(3)  # optimal pair ({1}, {2, 3})
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 1.0]) / np.sqrt(2)
    *_, val, _ = ascent(1 - 2 * (1 + np.sqrt(2)) / 2 * B, x, y)
    assert abs(val - np.sqrt(2)) < 1e-9


def test_end_state_and_repair():
    from bipen.penalty import end_state, repair

    B = np.array([[1, 0], [0, 0]], dtype=bool)  # one edge on 2+2 vertices, g = 2
    x, y = np.array([1.0, 0]), np.array([1.0, 1.0]) / np.sqrt(2)
    assert end_state(B, 2, x, y) == "edge"
    assert repair(B, x, y) == 2
    assert end_state(B, 2, np.array([0, 1.0]), y) == "optimal"
    assert end_state(B, 2, np.array([0, 1.0]), np.array([1.0, 0])) == "nonmaximal"


def test_dsos_level_0():
    # Propositions B.3 and B.4: zeta_L^(0) is half the maximum degree of the bipartite complement, and zeta_R^(0) lies
    # between half the spectral norm of J - U and zeta_L^(0)
    from bipen.dsos import zeta

    rng = np.random.default_rng(4)
    for _ in range(5):
        B = random_bipartite(3, 4, 0.5, rng)
        d = max((1 - B).sum(0).max(), (1 - B).sum(1).max())
        dsos, sdsos = zeta(B, 0), zeta(B, 0, scaled=True)
        assert abs(dsos - d / 2) < 1e-6
        assert 0.5 * np.linalg.norm(1 - B, 2) - 1e-6 <= sdsos <= dsos + 1e-6


def test_dsos_between_polya_and_k1():
    # Proposition B.2 at level 1
    from bipen.dsos import zeta as zeta_dsos
    from bipen.polya import zeta as zeta_polya
    from bipen.sdp import cop_k1

    rng = np.random.default_rng(5)
    for _ in range(3):
        B = random_bipartite(2, 3, 0.5, rng)
        chain = [zeta_polya(B, 1), zeta_dsos(B, 1), zeta_dsos(B, 1, scaled=True), cop_k1(B),
                 0.5 * np.sqrt(parameters(B)["g"])]
        assert all(a >= b - 1e-6 for a, b in zip(chain, chain[1:]))

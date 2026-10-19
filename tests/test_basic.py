import numpy as np

from bicop.brute import parameters
from bicop.graphs import cost_matrix, full_adjacency, knn_minus_matching, random_bipartite
from bicop.penalty import max_on_nonneg_sphere


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

    from bicop.sdp import _solve, k1_constraints

    n = 5
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    t = cp.Variable()
    M = t * (np.eye(n) + A) - np.ones((n, n))
    assert abs(_solve(cp.Problem(cp.Minimize(t), k1_constraints(M, n))) - 2) < 1e-5


def test_dnn_and_k1_are_upper_bounds():
    from bicop.sdp import cop_k1, h1

    rng = np.random.default_rng(3)
    for _ in range(5):
        B = random_bipartite(4, 5, 0.5, rng)
        t = 0.5 * np.sqrt(parameters(B)["g"])
        assert h1(B) >= h1(B, dnn=True) - 1e-6 >= cop_k1(B) - 2e-6 >= t - 3e-6

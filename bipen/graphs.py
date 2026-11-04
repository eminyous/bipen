"""Bipartite graphs G = (V1 ∪ V2, E), stored as their biadjacency matrix U.

U has m1 = |V1| rows and m2 = |V2| columns, and U[i, j] = 1 when the i-th vertex of V1 and the j-th vertex of V2 are
adjacent. The matrices indexed by the whole vertex set V, of size n = m1 + m2, list the vertices of V1 first.
"""

import numpy as np


def random_bipartite(m1, m2, p, rng):
    """A random graph on m1 + m2 vertices, each pair (i, j) in V1 x V2 being an edge with probability p."""
    return (rng.random((m1, m2)) < p).astype(float)


def matching(s):
    """The perfect matching sK_2 on s + s vertices."""
    return np.eye(s)


def crown(s):
    """The crown: K_{s,s} minus a perfect matching."""
    return np.ones((s, s)) - np.eye(s)


def near_biclique(k):
    """The graph N_k: V1 = {v10, ..., v1k}, V2 = {v21, ..., v2k} and the single edge v10 v21."""
    U = np.zeros((k + 1, k))
    U[0, 0] = 1
    return U


def edges(U):
    """The edges of G as pairs (i, j) of indices in V, with i in V1 and j in V2."""
    m1 = U.shape[0]
    return [(i, m1 + j) for i, j in zip(*np.nonzero(U))]


def adjacency(U):
    """The n x n adjacency matrix W of G."""
    m1, m2 = U.shape
    W = np.zeros((m1 + m2, m1 + m2))
    W[:m1, m1:] = U
    W[m1:, :m1] = U.T
    return W


def cost_matrix(m1, m2):
    """C = 1/2 [[0, J], [J^T, 0]], so that u^T C u = (sum of u over V1)(sum of u over V2)."""
    C = np.zeros((m1 + m2, m1 + m2))
    C[:m1, m1:] = 0.5
    C[m1:, :m1] = 0.5
    return C


def penalty_matrix(U, gamma):
    """H_gamma = J - 2 gamma U, the matrix of the bilinear form y^T H_gamma z of the penalty formulation."""
    return 1 - 2 * gamma * U


# The random graphs of the experiments. Each is fixed by its sides m1 and m2, its edge probability p and an index s,
# through the seed of its random generator, so that every experiment and table finds the same graphs.


def bounds_graph(m1, m2, p, s):
    """The random graphs of the bounds, with the seed 1000 m1 + 100 int(10 p) + s for balanced sizes and
    100000 m2 + 1000 m1 + 100 int(10 p) + s otherwise."""
    base = 1000 * m1 if m1 == m2 else 100000 * m2 + 1000 * m1
    return random_bipartite(m1, m2, p, np.random.default_rng(base + 100 * int(10 * p) + s))


def penalty_graph(m1, m2, p, s):
    """The random graphs of the smallest penalty, with the seed 100000 m2 + 1000 m1 + 100 int(10 p) + s."""
    return random_bipartite(m1, m2, p, np.random.default_rng(100000 * m2 + 1000 * m1 + 100 * int(10 * p) + s))

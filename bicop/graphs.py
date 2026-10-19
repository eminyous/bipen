"""Bipartite graphs as biadjacency matrices B (m x k, B[i, j] = 1 iff i ~ j)."""

import numpy as np


def full_adjacency(B):
    m, k = B.shape
    A = np.zeros((m + k, m + k))
    A[:m, m:] = B
    A[m:, :m] = B.T
    return A


def cost_matrix(m, k):
    """C = 1/2 [[0, J], [J, 0]], so u^T C u = (sum_{V1} u)(sum_{V2} u)."""
    C = np.zeros((m + k, m + k))
    C[:m, m:] = 0.5
    C[m:, :m] = 0.5
    return C


def knn_minus_matching(n):
    return np.ones((n, n)) - np.eye(n)


def random_bipartite(m, k, p, rng):
    return (rng.random((m, k)) < p).astype(float)


def edges(B):
    m, _ = B.shape
    return [(i, m + j) for i, j in zip(*np.nonzero(B))]

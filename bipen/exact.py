"""The exact values of g, h and alpha*, by enumerating the subsets A of V1.

For A ⊆ V1, the largest B ⊆ V2 such that (A, B) is biindependent is the set of common non-neighbours of A. The
quantities |A||B|, |A||B|/(|A| + |B|) and |A| + |B| all increase with |B|, so it suffices to enumerate A, which costs
2^m1 steps; the smaller side is taken as V1.
"""

import numpy as np


def pair_sizes(U):
    """For every subset A of V1, given by the bit mask whose bit i says whether the i-th vertex is in A: the sizes |A|
    and |B| of the pair (A, B), with B the set of common non-neighbours of A."""
    if U.shape[0] > U.shape[1]:
        U = U.T
    m1, m2 = U.shape
    # the neighbours of each vertex of V1, as a bit mask over V2
    neighbours = np.zeros(m1, dtype=np.int64)
    for i in range(m1):
        for j in range(m2):
            if U[i, j]:
                neighbours[i] |= 1 << j
    # the neighbours of A and the size of A, built by adding the vertices one at a time: the masks with highest bit i
    # are those below 2^i with the vertex i added
    neighbours_of = np.zeros(1 << m1, dtype=np.int64)
    size_of_A = np.zeros(1 << m1, dtype=np.int64)
    for i in range(m1):
        bit = 1 << i
        neighbours_of[bit : 2 * bit] = neighbours_of[:bit] | neighbours[i]
        size_of_A[bit : 2 * bit] = size_of_A[:bit] + 1
    covered = np.zeros(1 << m1, dtype=np.int64)
    for j in range(m2):
        covered += (neighbours_of >> j) & 1
    return size_of_A.astype(float), (m2 - covered).astype(float)


def biclique_number(U):
    """g(G), the largest |A||B| over the biindependent pairs (A, B)."""
    a, b = pair_sizes(U)
    return float(np.max(a * b))


def biclique_ratio(U):
    """h(G), the largest |A||B|/(|A| + |B|) over the biindependent pairs (A, B) with A ∪ B nonempty."""
    a, b = pair_sizes(U)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.max(np.where(a + b > 0, a * b / (a + b), 0.0)))


def alpha_star(U):
    """alpha*(G), the largest |A| + |B| over the biindependent pairs (A, B) with A and B nonempty (0 if there is
    none)."""
    a, b = pair_sizes(U)
    both_nonempty = (a > 0) & (b > 0)
    return int((a + b)[both_nonempty].max()) if both_nonempty.any() else 0

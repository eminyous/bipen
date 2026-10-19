"""Exact g, h and balanced variants by enumerating the side V1.

For A subset of V1 the largest compatible B is the set of common non-neighbours of A,
and |A||B|, |A||B|/(|A|+|B|) and min(|A|,|B|) all increase with |B|, so it suffices to
enumerate A.  Costs 2^m; pass the smaller side as V1.
"""

import numpy as np


def _nonneighbour_counts(B):
    m, k = B.shape
    nb = np.zeros(m, dtype=np.int64)
    for i in range(m):
        for j in range(k):
            if B[i, j]:
                nb[i] |= 1 << j
    union = np.zeros(1 << m, dtype=np.int64)
    size = np.zeros(1 << m, dtype=np.int64)
    for i in range(m):
        h = 1 << i
        union[h:2 * h] = union[:h] | nb[i]
        size[h:2 * h] = size[:h] + 1
    covered = np.zeros(1 << m, dtype=np.int64)
    for j in range(k):
        covered += (union >> j) & 1
    return size, k - covered


def parameters(B):
    """Return dict with g, h, g_bal, h_bal (h_bal = max |A|/2 over balanced pairs)."""
    if B.shape[0] > B.shape[1]:
        B = B.T
    a, b = _nonneighbour_counts(B)
    a = a.astype(float)
    b = b.astype(float)
    g = float(np.max(a * b))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(a + b > 0, a * b / (a + b), 0.0)
    s = np.minimum(a, b)
    return {"g": g, "h": float(np.max(r)), "g_bal": float(np.max(s * s)), "h_bal": float(np.max(s)) / 2}

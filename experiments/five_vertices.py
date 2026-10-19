"""Exact check of the proof that h1(G; D) = sqrt(g)/2 for every bipartite graph on five vertices (appendix).

Enumerates the bipartite graphs with sides 1+4 and 2+3 up to isomorphism, deletes full vertices, and checks
that each graph either has at most four vertices left, has no edges, or is one of the four graphs of the
appendix table. For those four, the certificate S = lambda I - C + Y - N is checked to be positive
semidefinite in exact arithmetic (sympy). Run with: python -m experiments.five_vertices
"""

import itertools

import numpy as np
import sympy as sp

from bicop.brute import parameters

HALF = sp.Rational(1, 2)
# vertices 0, 1 = a, b (V_1) and 2, 3, 4 = 1, 2, 3 (V_2); y is the entry of Y on every edge
CERTIFICATES = {
    "G1": dict(edges=[(1, 4)], g=4, y=sp.Integer(1), N={}),
    "G2": dict(edges=[(1, 3), (1, 4)], g=3, y=sp.Rational(3, 4), N={}),
    "G3": dict(edges=[(0, 4), (1, 3)], g=2, y=sp.Integer(1), N={(3, 4): sp.sqrt(2) / 2}),
    "G4": dict(edges=[(0, 4), (1, 2), (1, 3)], g=2, y=HALF, N={}),
}


def canonical(B):
    m, k = B.shape
    forms = [tuple(B[np.ix_(r, c)].flatten()) for r in itertools.permutations(range(m))
             for c in itertools.permutations(range(k))]
    return m, k, min(forms)


def delete_full(B):
    while True:
        m, k = B.shape
        rows = [i for i in range(m) if m > 1 and B[i].all()]
        cols = [j for j in range(k) if k > 1 and B[:, j].all()]
        if rows:
            B = np.delete(B, rows[0], 0)
        elif cols:
            B = np.delete(B, cols[0], 1)
        else:
            return B


def biadjacency(edges):
    B = np.zeros((2, 3), int)
    for i, j in edges:
        B[i, j - 2] = 1
    return B


def check_certificate(c):
    n = 5
    lam = sp.sqrt(c["g"]) / 2
    S = lam * sp.eye(n)
    for i in range(2):
        for j in range(2, n):
            S[i, j] = S[j, i] = -HALF
    for i, j in c["edges"]:
        S[i, j] += c["y"]
        S[j, i] += c["y"]
    for (i, j), v in c["N"].items():
        S[i, j] -= v
        S[j, i] -= v
    eigenvalues = S.eigenvals()
    assert all(sp.simplify(e).is_nonnegative for e in eigenvalues), eigenvalues
    return {sp.simplify(e): mult for e, mult in eigenvalues.items()}


def main():
    graphs = {}
    for m, k in [(1, 4), (2, 3)]:
        for bits in itertools.product([0, 1], repeat=m * k):
            B = np.array(bits).reshape(m, k)
            graphs.setdefault(canonical(B), B)
    hard = {canonical(biadjacency(c["edges"])): name for name, c in CERTIFICATES.items()}
    counts = {"n <= 4 after deleting full vertices": 0, "no edges": 0}
    for key, B in graphs.items():
        R = delete_full(B)
        if sum(R.shape) < 5:
            counts["n <= 4 after deleting full vertices"] += 1
        elif R.sum() == 0:
            counts["no edges"] += 1
        else:
            assert key in hard, B.tolist()
    print(len(graphs), "graphs:", counts, "and", len(hard), "with a certificate")
    for name, c in CERTIFICATES.items():
        assert parameters(biadjacency(c["edges"]).astype(float))["g"] == c["g"]
        print(name, "g =", c["g"], "eigenvalues of S:", check_certificate(c))


if __name__ == "__main__":
    main()

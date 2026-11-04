"""An exact check of the certificates of Appendix A (Proposition 3.5 for n = 5).

The proof shows h_1^D(G) = sqrt(g)/2 for every bipartite graph on five vertices.

Enumerates the graphs with sides 1+4 and 2+3 up to isomorphism, deletes full vertices, and checks that each graph
then has at most four vertices, or has no edge, or is one of the graphs G_1, ..., G_4 of Table A.1. For those four,
the certificate S = lambda I - C + Y - Z of the table is checked to be positive semidefinite in exact arithmetic.
Prints only.
Run with: bipen run five-vertex-certificates
"""

import itertools

import numpy as np
import sympy

from bipen.exact import biclique_number

HALF = sympy.Rational(1, 2)
# Table A.1, with the vertices 0, 1 = v11, v12 (V1) and 2, 3, 4 = v21, v22, v23 (V2): the edges, g, the value w of Y
# on every edge, and the nonzero entries of Z above the diagonal
CERTIFICATES = {
    "G1": {"edges": [(1, 4)], "g": 4, "w": sympy.Integer(1), "Z": {}},
    "G2": {"edges": [(1, 3), (1, 4)], "g": 3, "w": sympy.Rational(3, 4), "Z": {}},
    "G3": {"edges": [(0, 4), (1, 3)], "g": 2, "w": sympy.Integer(1), "Z": {(3, 4): sympy.sqrt(2) / 2}},
    "G4": {"edges": [(0, 4), (1, 2), (1, 3)], "g": 2, "w": HALF, "Z": {}},
}


def canonical_form(U):
    """The smallest biadjacency matrix over the permutations of both sides, which identifies G up to isomorphism."""
    m1, m2 = U.shape
    forms = [
        tuple(U[np.ix_(rows, columns)].flatten())
        for rows in itertools.permutations(range(m1))
        for columns in itertools.permutations(range(m2))
    ]
    return m1, m2, min(forms)


def delete_full_vertices(U):
    """Delete full vertices one at a time, as long as their side keeps another vertex (Lemma A.1)."""
    while True:
        m1, m2 = U.shape
        full_in_V1 = [i for i in range(m1) if m1 > 1 and U[i].all()]
        full_in_V2 = [j for j in range(m2) if m2 > 1 and U[:, j].all()]
        if full_in_V1:
            U = np.delete(U, full_in_V1[0], 0)
        elif full_in_V2:
            U = np.delete(U, full_in_V2[0], 1)
        else:
            return U


def biadjacency(edges):
    U = np.zeros((2, 3), int)
    for i, j in edges:
        U[i, j - 2] = 1
    return U


def certificate_eigenvalues(certificate):
    """The eigenvalues of S, with their multiplicities, after checking that they are nonnegative."""
    n = 5
    S = sympy.sqrt(certificate["g"]) / 2 * sympy.eye(n)  # lambda I
    for i in range(2):
        for j in range(2, n):
            S[i, j] = S[j, i] = -HALF  # - C
    for i, j in certificate["edges"]:
        S[i, j] += certificate["w"]  # + Y
        S[j, i] += certificate["w"]
    for (i, j), value in certificate["Z"].items():
        S[i, j] -= value  # - Z
        S[j, i] -= value
    eigenvalues = S.eigenvals()
    assert all(sympy.simplify(e).is_nonnegative for e in eigenvalues), eigenvalues
    return {sympy.simplify(e): multiplicity for e, multiplicity in eigenvalues.items()}


def main():
    graphs = {}
    for m1, m2 in [(1, 4), (2, 3)]:
        for bits in itertools.product([0, 1], repeat=m1 * m2):
            U = np.array(bits).reshape(m1, m2)
            graphs.setdefault(canonical_form(U), U)
    with_certificate = {canonical_form(biadjacency(c["edges"])): name for name, c in CERTIFICATES.items()}
    counts = {"n <= 4 after deleting full vertices": 0, "no edges": 0}
    for form, U in graphs.items():
        reduced = delete_full_vertices(U)
        if sum(reduced.shape) < 5:
            counts["n <= 4 after deleting full vertices"] += 1
        elif reduced.sum() == 0:
            counts["no edges"] += 1
        else:
            assert form in with_certificate, U.tolist()
    print(len(graphs), "graphs:", counts, "and", len(with_certificate), "with a certificate")
    for name, certificate in CERTIFICATES.items():
        assert biclique_number(biadjacency(certificate["edges"]).astype(float)) == certificate["g"]
        print(name, "g =", certificate["g"], "eigenvalues of S:", certificate_eigenvalues(certificate))

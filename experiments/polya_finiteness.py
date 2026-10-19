"""First level r at which the Polya LP bound zeta^(r) is finite, against alpha*(G) - 1.

alpha*(G) = max |A| + |B| over biindependent pairs with A, B nonempty.
"""

import itertools
import sys
import warnings

import numpy as np

from bicop.polya import zeta

warnings.filterwarnings("ignore")


def alpha_star(B):
    m, k = B.shape
    best = 0
    for a in range(1, 1 << m):
        A = [i for i in range(m) if a >> i & 1]
        free = [j for j in range(k) if not any(B[i, j] for i in A)]
        if free:
            best = max(best, len(A) + len(free))
    return best


def first_finite(B, rmax):
    for r in range(rmax + 1):
        if np.isfinite(zeta(B, r)):
            return r
    return None


def main(out):
    rng = np.random.default_rng(8)
    bad = 0
    with open(out, "w") as f:
        f.write("m,k,edges,alpha_star,first_finite_r\n")
        for m, k in [(2, 2), (2, 3), (3, 3), (3, 4)]:
            for _ in range(15):
                B = (rng.random((m, k)) < rng.uniform(0.2, 0.8)).astype(float)
                if B.sum() == m * k:
                    continue
                a = alpha_star(B)
                r = first_finite(B, a)
                f.write(f"{m},{k},{int(B.sum())},{a},{r}\n")
                bad += r != a - 1
    print("graphs where first finite level != alpha* - 1:", bad)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/polya_finiteness.csv")

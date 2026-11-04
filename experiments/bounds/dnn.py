"""A search for small graphs on which h_1^D is not exact (Propositions 3.4 and 3.5).

Every graph of the sizes 2+3 to 2+5, then 400 random graphs for each size 4+4 to 6+6, with an edge probability drawn
in [0.2, 0.8]. Writes one line per graph found.
Run with: bipen run bounds/dnn
"""

import itertools
import warnings

import numpy as np

from bipen.data import File
from bipen.exact import biclique_number
from bipen.sdp import h1

warnings.filterwarnings("ignore")
TOL = 1e-5


def gap(U):
    """h_1^D - sqrt(g)/2, sqrt(g)/2 and h_1^D."""
    half_sqrt_g = 0.5 * np.sqrt(biclique_number(U))
    h1_dnn = h1(U, dnn=True)
    return h1_dnn - half_sqrt_g, half_sqrt_g, h1_dnn


def main(data):
    with (data / File.DNN_EXACTNESS).open("w") as f:
        for m1, m2 in [(2, 3), (3, 3), (2, 4), (3, 4), (2, 5)]:
            worst = 0.0
            for bits in itertools.product([0, 1], repeat=m1 * m2):
                U = np.array(bits, float).reshape(m1, m2)
                difference, half_sqrt_g, h1_dnn = gap(U)
                worst = max(worst, difference)
                if difference > TOL:
                    f.write(
                        f"{m1}x{m2} gap={difference:.6f} sqrtg/2={half_sqrt_g:.6f} dnn={h1_dnn:.6f} "
                        f"B={U.astype(int).tolist()}\n"
                    )
            print(f"{m1}x{m2} exhaustive: worst gap {worst:.2e}", flush=True)
        rng = np.random.default_rng(11)
        for m1, m2 in [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]:
            worst, worst_graph = 0.0, None
            for _ in range(400):
                U = (rng.random((m1, m2)) < rng.uniform(0.2, 0.8)).astype(float)
                difference = gap(U)[0]
                if difference > worst:
                    worst, worst_graph = difference, U
            print(f"{m1}x{m2} sampled 400: worst gap {worst:.2e}", flush=True)
            if worst > TOL:
                f.write(f"{m1}x{m2} gap={worst:.6f} B={worst_graph.astype(int).tolist()}\n")
                f.flush()

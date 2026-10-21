"""Search small bipartite graphs for h1_dnn > sqrt(g)/2 (exhaustive for small sides, then sampled)."""

import itertools
import os
import warnings

import numpy as np

from bipen.brute import parameters
from bipen.sdp import h1

warnings.filterwarnings("ignore")
TOL = 1e-5


def check(B):
    t = 0.5 * np.sqrt(parameters(B)["g"])
    d = h1(B, dnn=True)
    return d - t, t, d


def main(data):
    found = []
    with open(os.path.join(data, "smallest_dnn_gap.txt"), "w") as f:
        for m, k in [(2, 3), (3, 3), (2, 4), (3, 4), (2, 5)]:
            worst = 0.0
            for bits in itertools.product([0, 1], repeat=m * k):
                B = np.array(bits, float).reshape(m, k)
                gap, t, d = check(B)
                worst = max(worst, gap)
                if gap > TOL:
                    found.append((m, k, B))
                    f.write(f"{m}x{k} gap={gap:.6f} sqrtg/2={t:.6f} dnn={d:.6f} B={B.astype(int).tolist()}\n")
            print(f"{m}x{k} exhaustive: worst gap {worst:.2e}", flush=True)
        rng = np.random.default_rng(11)
        for m, k in [(4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]:
            worst, hit = 0.0, None
            for _ in range(400):
                B = (rng.random((m, k)) < rng.uniform(0.2, 0.8)).astype(float)
                gap, t, d = check(B)
                if gap > worst:
                    worst, hit = gap, B
            print(f"{m}x{k} sampled 400: worst gap {worst:.2e}", flush=True)
            if worst > TOL:
                f.write(f"{m}x{k} gap={worst:.6f} B={hit.astype(int).tolist()}\n")
                f.flush()


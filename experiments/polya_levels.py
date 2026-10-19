"""Polya LP bounds zeta^(r) on small structured graphs, for r up to R (data/polya_levels.csv)."""

import csv
import sys
import warnings

import numpy as np

from bicop.brute import parameters
from bicop.polya import zeta
from experiments.polya_finiteness import alpha_star

warnings.filterwarnings("ignore")
R = 7


def graphs():
    yield "lpv-example-1", np.array([[1.0, 0.0], [0.0, 0.0]])
    yield "matching-3", np.eye(3)
    yield "matching-4", np.eye(4)
    yield "crown-4", np.ones((4, 4)) - np.eye(4)


def main(out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "half_sqrt_g", "alpha_star", "r", "zeta"])
        for name, B in graphs():
            t = 0.5 * np.sqrt(parameters(B)["g"])
            a = alpha_star(B)
            for r in range(R + 1):
                z = zeta(B, r)
                w.writerow([name, t, a, r, z])
                print(name, r, z, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/polya_levels.csv")

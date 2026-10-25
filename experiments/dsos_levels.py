"""DSOS bounds zeta_L^(r) on the graphs of the Polya figure and on N_2, for r up to R (dsos_levels.csv)."""

import csv
import os
import warnings

import numpy as np

from bipen.brute import parameters
from bipen.dsos import zeta
from experiments.polya_levels import graphs as polya_graphs

warnings.filterwarnings("ignore")
R = 2


def graphs():
    yield from polya_graphs()
    B = np.zeros((3, 2))
    B[0, 0] = 1
    yield "near-biclique-2", B


def main(data):
    with open(os.path.join(data, "dsos_levels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "half_sqrt_g", "r", "zeta_dsos"])
        for name, B in graphs():
            t = 0.5 * np.sqrt(parameters(B)["g"])
            for r in range(R + 1):
                z = zeta(B, r)
                w.writerow([name, t, r, z])
                print(name, r, z, flush=True)

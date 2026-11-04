"""The files of the data folder, their columns and their labels, named here once: the experiments write them, and
`bipen tables` and `bipen plots` read them.

Several names come from an earlier notation and are kept so that the existing results stay readable: m and k are m1
and m2, k1 is rho^(1), d_star is gamma*, bound_sqrt_g and bound_gg are gamma_1 and gamma_2, and zeta_dsos is
zeta~^(r).
"""

import csv


class File:
    STRUCTURED_BOUNDS = "bounds.csv"
    RANDOM_BOUNDS = "random.csv"
    UNBALANCED_BOUNDS = "bounds_unbalanced.csv"  # an earlier run on the unbalanced sizes, now part of random.csv
    RHO_GAPS = "k1_check.csv"
    LASSERRE_COMPARISON = "lasserre.csv"
    SOLVER_COMPARISON = "solver_bench.csv"
    SMALLEST_PENALTY = "smallest_penalty.csv"
    LOCAL_ASCENT = "local_ascent.csv"
    POLYA_FIRST_FINITE_LEVEL = "polya_finiteness.csv"
    POLYA_BOUNDS = "polya_levels.csv"
    DSOS_BOUNDS = "dsos_levels.csv"
    DNN_EXACTNESS = "smallest_dnn_gap.txt"

    @staticmethod
    def lasserre_size(m):
        """The Lasserre comparison run on one size m + m only."""
        return f"lasserre{m}.csv"


class Column:
    FAMILY = "family"
    GRAPH = "graph"
    M1 = "m"
    M2 = "k"
    P = "p"
    SEED = "seed"
    EDGES = "edges"
    G = "g"
    HALF_SQRT_G = "half_sqrt_g"
    H1 = "h1"
    H1_DNN = "h1_dnn"
    RHO1 = "k1"
    LASSERRE = "las2"
    LEVEL = "r"
    ALPHA_STAR = "alpha_star"
    ZETA = "zeta"
    ZETA_TILDE = "zeta_dsos"
    GAMMA_STAR = "d_star"
    GAMMA1 = "bound_sqrt_g"
    GAMMA2 = "bound_gg"
    H = "h"
    SECONDS = "seconds"
    # rho^(1) solved again (RHO_GAPS)
    STATUS = "status"
    PRIMAL = "primal"
    DUAL = "dual"
    PRIMAL_RESIDUAL = "r_prim"
    DUAL_RESIDUAL = "r_dual"
    # the Lasserre and solver comparisons
    RHO1_SECONDS = "seconds_k1"
    LASSERRE_SECONDS = "seconds_las2"
    SOLVER = "solver"
    TOLERANCE = "tol"
    BOUND = "bound"
    VALUE = "value"
    PEAK_GB = "peak_gb"
    # the Polya levels
    FIRST_FINITE_LEVEL = "first_finite_r"
    # the local ascent: the penalty, the kind of start, and the fractions of runs that end in each way
    PENALTY = "threshold"
    GAMMA = "gamma"
    STARTS = "starts"
    SUCCESS = "success"
    REPAIRED = "repaired"
    EDGE_END = "edge"
    EDGE_POSITIVE = "edge_positive"
    NONMAXIMAL = "nonmaximal"
    MAXIMAL = "maximal"
    FALLBACK = "fallback"
    SUPPORT = "support"
    STEPS = "steps"


class Family:
    """The values of the column family."""

    RANDOM = "random"
    MATCHING = "matching"
    CROWN = "crown"
    LPV_EXAMPLE = "lpv-ex3.7"  # the graph of LPV, Example 1
    NEAR_BICLIQUE = "near-biclique"  # N_k


class Graph:
    """The values of the column graph (Figure 1 and Table B.1)."""

    LPV_EXAMPLE = "lpv-example-1"
    MATCHING_3 = "matching-3"
    MATCHING_4 = "matching-4"
    CROWN_4 = "crown-4"
    NEAR_BICLIQUE_2 = "near-biclique-2"


class Penalty:
    """The values of the column threshold of local_ascent.csv: gamma_1 to gamma_4, then 1/2 + c sqrt(g)/2."""

    GAMMA1 = "gamma1"
    GAMMA2 = "gamma2"
    GAMMA3 = "gamma3"
    GAMMA4 = "gamma4"
    VALID = (GAMMA1, GAMMA2, GAMMA3, GAMMA4)  # the penalties at least gamma_1

    @staticmethod
    def below(c):
        """The penalty 1/2 + c sqrt(g)/2, for c = 0.25, 0.5 or 0.75."""
        return f"gamma_c{int(100 * c)}"


class Starts:
    """The values of the column starts of local_ascent.csv."""

    RANDOM = "random"
    VERTEX = "vertex"


def graph_key(row):
    """The random graph of a row: its sides, edge probability and index."""
    return int(row[Column.M1]), int(row[Column.M2]), float(row[Column.P]), int(row[Column.SEED])


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, mode="w"):
    """Write the dicts of rows to path as they come, with the keys of the first one as header (unless appending to a
    file that has one), flushing after each row so that a long run can be followed and resumed."""
    has_header = mode == "a" and path.exists() and path.stat().st_size > 0
    with path.open(mode, newline="") as f:
        writer = None
        for row in rows:
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not has_header:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()

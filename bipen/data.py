"""The files of the data folder, their columns and their labels, named here once: the experiments write them, and
`bipen tables` and `bipen plots` read them. Each file is named after its experiment: family/name writes
family/name.csv.
"""

import csv


class File:
    BOUNDS_STRUCTURED = "bounds/structured.csv"
    BOUNDS_SAMPLED = "bounds/sampled.csv"
    # read before BOUNDS_SAMPLED, which overrides its rows; it sets the order in which the bounds table sums them
    BOUNDS_UNBALANCED = "bounds/unbalanced.csv"
    BOUNDS_INEXACT = "bounds/inexact.csv"
    BOUNDS_LASSERRE = "bounds/lasserre.csv"
    BOUNDS_SOLVERS = "bounds/solvers.csv"
    BOUNDS_DNN = "bounds/dnn.txt"
    PENALTY_SMALLEST = "penalty/smallest.csv"
    PENALTY_ASCENT = "penalty/ascent.csv"
    POLYA_FINITENESS = "polya/finiteness.csv"
    POLYA_LEVELS = "polya/levels.csv"
    POLYA_DSOS = "polya/dsos.csv"

    @staticmethod
    def bounds_lasserre_size(m):
        """The Lasserre comparison run on one size m + m only."""
        return f"bounds/lasserre_{m}.csv"


class Column:
    FAMILY = "family"
    GRAPH = "graph"
    M1 = "m1"
    M2 = "m2"
    P = "p"
    INDEX = "index"  # the index s of a random graph among those of its size and p
    EDGES = "edges"
    G = "g"
    H = "h"
    HALF_SQRT_G = "half_sqrt_g"
    H1 = "h1"
    H1_DNN = "h1_dnn"
    RHO1 = "rho1"
    HALF_SQRT_G2 = "half_sqrt_g2"  # the Lasserre bound g_2, as sqrt(g_2)/2
    LEVEL = "level"
    ALPHA_STAR = "alpha_star"
    ZETA = "zeta"
    ZETA_TILDE = "zeta_tilde"
    GAMMA_STAR = "gamma_star"
    GAMMA1 = "gamma1"
    GAMMA2 = "gamma2"
    SECONDS = "seconds"
    # rho^(1) solved again (BOUNDS_INEXACT)
    STATUS = "status"
    PRIMAL = "primal"
    DUAL = "dual"
    PRIMAL_RESIDUAL = "primal_residual"
    DUAL_RESIDUAL = "dual_residual"
    # the Lasserre and solver comparisons
    SECONDS_RHO1 = "seconds_rho1"
    SECONDS_LASSERRE = "seconds_lasserre"
    SOLVER = "solver"
    TOLERANCE = "tolerance"
    BOUND = "bound"
    VALUE = "value"
    PEAK_GB = "peak_gb"
    # the Polya levels
    FIRST_FINITE_LEVEL = "first_finite_level"
    # the local ascent: the penalty, the kind of start, and the fractions of runs that end in each way
    PENALTY = "penalty"
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
    LPV_EXAMPLE = "lpv-example-1"  # the graph of LPV, Example 1
    NEAR_BICLIQUE = "near-biclique"  # N_k


class Graph:
    """The values of the column graph: the graphs of the Pólya and DSOS bounds."""

    LPV_EXAMPLE = "lpv-example-1"
    MATCHING_3 = "matching-3"
    MATCHING_4 = "matching-4"
    CROWN_4 = "crown-4"
    NEAR_BICLIQUE_2 = "near-biclique-2"


class Penalty:
    """The values of the column penalty of PENALTY_ASCENT: gamma_1 to gamma_4, then 1/2 + c sqrt(g)/2."""

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
    """The values of the column starts of PENALTY_ASCENT."""

    RANDOM = "random"
    VERTEX = "vertex"


def graph_key(row):
    """The random graph of a row: its sides, edge probability and index."""
    return int(row[Column.M1]), int(row[Column.M2]), float(row[Column.P]), int(row[Column.INDEX])


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, mode="w"):
    """Write the dicts of rows to path as they come, with the keys of the first one as header (unless appending to a
    file that has one), flushing after each row so that a long run can be followed and resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
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

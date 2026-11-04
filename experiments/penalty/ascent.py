"""The local ascent with seven penalties and two kinds of starts.

Block-coordinate ascent on max y^T H_gamma z over nonnegative unit vectors, with H_gamma = J - 2 gamma U.

Graphs: the first 20 random graphs of bounds/sampled of the sizes 10+10, 15+15 and 20+20 and each p in {0.3, 0.5, 0.7}.
Penalties: gamma_1 = (1 + sqrt g)/2, which rounding rho^(1) (read from bounds/sampled) gives on every graph;
gamma_2 = (1 + max(m1, m2))/2; gamma_3 = (1 + sqrt |E(G^)|)/2, with G^ the bipartite complement;
gamma_4 = (1 + |J - U|_2)/2; and 1/2 + c sqrt(g)/2 below gamma_1, for c = 1/4, 1/2 and 3/4.
Starts: 100 random pairs (y, z) (numpy.random.default_rng(s) for the graph of index s), and the n vertices, that is,
y = e_i or z = e_j.

For each graph, penalty and kind of start, the fraction of runs that
- success: end at a maximum biindependent pair (for gamma >= gamma_1, the runs that end at value sqrt(g));
- repaired: reach g after the greedy repair of penalty.repair;
- edge, nonmaximal, maximal: end with an edge of G in the support, at a biindependent pair that is not maximal, or at
  a maximal pair that is not maximum (penalty.end_state); edge_positive: end with an edge and a positive value;
- fallback: find (H z)_+ = 0 at the first step and move to a vertex;
and the mean fraction of the side in the support after the first step (support) and the mean number of steps.
Run with: bipen run penalty/ascent --workers N
"""

from multiprocessing import Pool

import numpy as np

from bipen.data import Column, File, Penalty, Starts, graph_key, read_csv, write_csv
from bipen.exact import biclique_number
from bipen.graphs import bounds_graph, penalty_matrix
from bipen.penalty import EDGE, MAXIMAL, NONMAXIMAL, OPTIMAL, ascent, best_response, end_state, repair

SIZES = [10, 15, 20]
PS = [0.3, 0.5, 0.7]
GRAPHS = 20
RANDOM_STARTS = 100
RHO = {}  # rho^(1) of each graph, read by main before the workers start


def run(H, start):
    """One ascent from start, which is (Starts.RANDOM, (y, z)), ("y", i) for y = e_i, or ("z", j) for z = e_j.
    Returns y, z, the value, the number of steps, and the vector a whose best response is the first step."""
    kind, where = start
    if kind == "y":  # the first step computes z from y = e_i
        a = H[where]
        z, y, value, steps = ascent(H.T, best_response(a), np.eye(H.shape[0])[where])
    elif kind == "z":  # the first step computes y from z = e_j
        a = H[:, where]
        y, z, value, steps = ascent(H, best_response(a), np.eye(H.shape[1])[where])
    else:
        y, z = where
        a = H @ z
        y, z, value, steps = ascent(H, y, z)
    return y, z, value, steps, a


def penalties(U, g):
    m = U.shape[0]
    gammas = {
        Penalty.GAMMA1: (1 + np.sqrt(g)) / 2,
        Penalty.GAMMA4: (1 + np.linalg.norm(1 - U, 2)) / 2,
        Penalty.GAMMA3: (1 + np.sqrt((1 - U).sum())) / 2,
        Penalty.GAMMA2: (1 + m) / 2,
    }
    gammas.update({Penalty.below(c): 0.5 + c * np.sqrt(g) / 2 for c in (0.25, 0.5, 0.75)})
    return gammas


def starts(m, s):
    rng = np.random.default_rng(s)
    random_starts = []
    for _ in range(RANDOM_STARTS):
        y, z = rng.random(m), rng.random(m)
        random_starts.append((Starts.RANDOM, (y / np.linalg.norm(y), z / np.linalg.norm(z))))
    return {Starts.RANDOM: random_starts, Starts.VERTEX: [("y", i) for i in range(m)] + [("z", j) for j in range(m)]}


def outcomes(U, g, runs):
    """The fractions of the runs that end in each way, as in the module docstring."""
    ends = [end_state(U, g, y, z) for y, z, *_ in runs]
    fraction = lambda condition: np.mean([condition(end, run) for end, run in zip(ends, runs)])
    return {
        Column.SUCCESS: fraction(lambda end, run: end == OPTIMAL),
        Column.REPAIRED: fraction(lambda end, run: repair(U, run[0], run[1]) == g),
        Column.EDGE_END: fraction(lambda end, run: end == EDGE),
        Column.EDGE_POSITIVE: fraction(lambda end, run: end == EDGE and run[2] > 0),
        Column.NONMAXIMAL: fraction(lambda end, run: end == NONMAXIMAL),
        Column.MAXIMAL: fraction(lambda end, run: end == MAXIMAL),
        Column.FALLBACK: fraction(lambda end, run: run[4].max() <= 0),
        Column.SUPPORT: fraction(lambda end, run: (run[4] > 0).sum() / len(run[4])),
        Column.STEPS: fraction(lambda end, run: run[3]),
    }


def job(graph):
    m, p, s = graph
    U = bounds_graph(m, m, p, s).astype(bool)
    g = biclique_number(U)
    assert np.floor(4 * RHO[(m, m, p, s)] ** 2 + 1e-6) == g  # rounding rho^(1) gives g, so gamma_1 is computable
    all_starts = starts(m, s)
    rows = []
    for name, gamma in penalties(U, g).items():
        H = penalty_matrix(U, gamma)
        for kind, kind_starts in all_starts.items():
            runs = [run(H, start) for start in kind_starts]
            rows.append(
                {
                    Column.M1: m,
                    Column.M2: m,
                    Column.P: p,
                    Column.INDEX: s,
                    Column.G: g,
                    Column.PENALTY: name,
                    Column.GAMMA: gamma,
                    Column.STARTS: kind,
                    **outcomes(U, g, runs),
                }
            )
    return rows


def main(data, workers=2):
    RHO.update({graph_key(row): float(row[Column.RHO1]) for row in read_csv(data / File.BOUNDS_SAMPLED)})
    todo = [(m, p, s) for m in SIZES for p in PS for s in range(GRAPHS)]
    with Pool(workers) as pool:
        write_csv(data / File.PENALTY_ASCENT, (row for rows in pool.imap_unordered(job, todo) for row in rows))
    print("finished", flush=True)

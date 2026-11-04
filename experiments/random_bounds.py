"""h_1, h_1^D and rho^(1) on the random graphs of Table 1.

100 graphs for each size and each p in {0.3, 0.5, 0.7}, with rho^(1) also on 20+20.
The 180 random graphs of structured_bounds are the first ten of each balanced size and p; they are copied from there
when it exists. The results are appended, largest graphs first, and a rerun skips the graphs already there. The 20+20
graphs need about 10 minutes and 5 GB each, and the whole run about 60 CPU hours.
Run with: bipen run random-bounds --workers N
"""

import warnings
from multiprocessing import Pool

from bipen.data import Column, Family, File, graph_key, read_csv, write_csv
from bipen.graphs import bounds_graph
from experiments import structured_bounds

warnings.filterwarnings("ignore")
structured_bounds.RHO_MAX_N = 40
SIZES = [(6, 6), (8, 8), (10, 10), (12, 12), (15, 15), (20, 20), (5, 10), (8, 16), (10, 20), (6, 24)]
PS = [0.3, 0.5, 0.7]
GRAPHS = 100


def job(graph):
    m1, m2, p, s = graph
    return structured_bounds.bounds_row(Family.RANDOM, bounds_graph(m1, m2, p, s), p, s)


def main(data, workers=2):
    out, structured = data / File.RANDOM_BOUNDS, data / File.STRUCTURED_BOUNDS
    if not out.exists() and structured.exists():
        write_csv(
            out,
            [
                row
                for row in read_csv(structured)
                if row[Column.FAMILY] == Family.RANDOM and row[Column.RHO1] not in ("", "nan")
            ],
        )
    done = {graph_key(row) for row in read_csv(out)} if out.exists() else set()
    todo = [(m1, m2, p, s) for m1, m2 in SIZES for p in PS for s in range(GRAPHS) if (m1, m2, p, s) not in done]
    todo.sort(key=lambda graph: -(graph[0] + graph[1]))
    print(len(done), "done,", len(todo), "to do", flush=True)

    def rows(pool):
        for row in pool.imap_unordered(job, todo):
            print(*graph_key(row), row[Column.SECONDS], flush=True)
            yield row

    with Pool(workers) as pool:
        write_csv(out, rows(pool), mode="a")
    print("finished", flush=True)

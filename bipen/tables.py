"""The numbers of the tables of the paper, read from the data folder and written as CSV files to the tables folder,
which the paper typesets. Column names use letters only, as csvsimple turns them into macros; the file and column
names of the data folder are those of bipen.data.
Run with: bipen tables
"""

import csv
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction

import numpy as np

from bipen.data import Column, Family, File, Graph, Penalty, Starts, graph_key, read_csv, require
from bipen.graphs import near_biclique, penalty_graph
from bipen.penalty import BISECTION_TOL
from bipen.sdp import MOSEK_TOL

TOL = 1e-5  # a bound is exact when it exceeds sqrt(g)/2 by less than TOL
ROUND = 1e-6  # added to 4 b^2 before rounding a bound b down to g
BOUNDS = {Column.H1: "hone", Column.H1_DNN: "honednn", Column.RHO1: "rhoone"}  # data column: name in the tables
PENALTIES = {
    Penalty.GAMMA1: "gammaone",
    Penalty.GAMMA2: "gammatwo",
    Penalty.GAMMA3: "gammathree",
    Penalty.GAMMA4: "gammafour",
    Penalty.below(0.25): "cquarter",
    Penalty.below(0.5): "chalf",
    Penalty.below(0.75): "cthreequarters",
}


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def half_sqrt_g(row):
    return float(row[Column.HALF_SQRT_G])


def is_exact(row, column):
    return float(row[column]) - half_sqrt_g(row) < TOL


def mean(fractions):
    """The exact mean of fractions of runs (k/100 or k/(2m)), to twelve decimals, so that the paper's rounding half up
    does not depend on the order of the rows."""
    q = sum(Fraction(f).limit_denominator(1000) for f in fractions) / len(fractions)
    return (Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal("1e-12")).normalize()


def sampled(data):
    """The random graphs of the bounds, from every results file, one row per graph."""
    rows = {}
    for name in (File.BOUNDS_STRUCTURED, File.BOUNDS_UNBALANCED, File.BOUNDS_SAMPLED):
        if (data / name).exists():
            for row in read_csv(data / name):
                if row[Column.FAMILY] == Family.RANDOM and row[Column.RHO1] not in ("", "nan"):
                    rows[graph_key(row)] = row
    return list(rows.values())


def sizes(data):
    """The random graphs by size, balanced sizes first, each block ordered by shape and then by size."""
    groups = defaultdict(list)
    for row in sampled(data):
        groups[(int(row[Column.M1]), int(row[Column.M2]))].append(row)
    order = sorted(groups, key=lambda size: (size[0] < size[1], size[1] // size[0], size[0] + size[1]))
    return [("balanced" if m1 == m2 else "unbalanced", m1, m2, groups[(m1, m2)]) for m1, m2 in order]


def gaps(graphs):
    """The number of graphs, and for each bound the number on which it is exact and its mean and maximum gap, in
    percent of sqrt(g)/2."""
    half = np.array([half_sqrt_g(row) for row in graphs])
    out = {"count": len(graphs)}
    for column, name in BOUNDS.items():
        gap = np.array([float(row[column]) for row in graphs]) - half
        gap[gap < TOL] = 0.0
        out["exact" + name] = int((gap == 0).sum())
        out["mean" + name] = float((100 * gap / half).mean())
        out["max" + name] = float((100 * gap / half).max())
    return out


def bounds(data, tables):
    """The bounds on the random graphs, one row per size, then one per size and p."""
    groups = sizes(data)
    rows = [{"block": block, "size": f"{m1}+{m2}", "p": "all", **gaps(graphs)} for block, m1, m2, graphs in groups]
    rows += [
        {
            "block": block,
            "size": f"{m1}+{m2}",
            "p": f"{p:g}",
            **gaps([row for row in graphs if float(row[Column.P]) == p]),
        }
        for block, m1, m2, graphs in groups
        for p in sorted({float(row[Column.P]) for row in graphs})
    ]
    write(tables / "bounds.csv", rows)


def matchings(data, tables):
    """The bounds on the perfect matchings sK_2."""
    rows = sorted(
        (row for row in read_csv(data / File.BOUNDS_STRUCTURED) if row[Column.FAMILY] == Family.MATCHING),
        key=lambda row: int(row[Column.M1]),
    )
    write(
        tables / "matchings.csv",
        [
            {
                "r": row[Column.M1],
                "halfsqrtg": row[Column.HALF_SQRT_G],
                **{name: row[column] for column, name in BOUNDS.items()},
            }
            for row in rows
        ],
    )


def graph(row):
    """The graph of a row of the smallest penalty: a random graph, or N_k."""
    m1, m2 = int(row[Column.M1]), int(row[Column.M2])
    if row[Column.FAMILY] == Family.RANDOM:
        return penalty_graph(m1, m2, float(row[Column.P]), int(row[Column.INDEX]))
    return near_biclique(m2)


def thresholds(data, tables):
    """The smallest valid penalty gamma* against gamma_1 = (1 + sqrt g)/2, and gamma_1 against gamma_2 and
    gamma_3 of Gillis and Glineur and the spectral threshold gamma_4, as ratios computed graph by graph: the mean and
    the maximum or minimum over the random graphs of each size, and the value on each N_k."""
    gamma_star = lambda row: float(row[Column.GAMMA_STAR])
    gamma1 = lambda row: float(row[Column.GAMMA1])
    gamma2 = lambda row: float(row[Column.GAMMA2])
    # Gillis and Glineur, Lemma 3: 2 gamma - 1 >= the square root of the number of edges of the bipartite complement
    gamma3 = lambda row: (1 + np.sqrt(int(row[Column.M1]) * int(row[Column.M2]) - int(row[Column.EDGES]))) / 2
    gamma4 = lambda row: (1 + np.linalg.norm(1 - graph(row), 2)) / 2
    rows = read_csv(data / File.PENALTY_SMALLEST)
    groups = defaultdict(list)
    for row in rows:
        if row[Column.FAMILY] == Family.RANDOM and float(row[Column.G]) >= 1:
            groups[f"{row[Column.M1]}+{row[Column.M2]}"].append(row)
    for row in sorted(
        (row for row in rows if row[Column.FAMILY] == Family.NEAR_BICLIQUE), key=lambda row: int(row[Column.M2])
    ):
        groups[row[Column.M2]] = [row]
    out = []
    for key, graphs in groups.items():
        row = {"family": "random" if "+" in key else "nearbiclique", "graphs": key, "count": len(graphs)}
        star = [gamma_star(r) / gamma1(r) for r in graphs]
        row["gammastaroveronemean"], row["gammastaroveronemax"] = float(np.mean(star)), max(star)
        for name, gamma in (("two", gamma2), ("three", gamma3), ("four", gamma4)):
            ratios = [gamma1(r) / gamma(r) for r in graphs]
            row["gammaoneover" + name + "mean"], row["gammaoneover" + name + "min"] = (
                float(np.mean(ratios)),
                min(ratios),
            )
        out.append(row)
    write(tables / "thresholds.csv", out)


def ascent(data, tables):
    """The local ascent: for each kind of start, size, p and penalty, the mean fraction of starts from which the
    ascent ends at a maximum pair, without and with repair, and the number of graphs where some start does."""
    groups = defaultdict(list)
    for row in read_csv(data / File.PENALTY_ASCENT):
        groups[(row[Column.STARTS], int(row[Column.M1]), float(row[Column.P]))].append(row)
    out = []
    for starts, m, p in sorted(groups, key=lambda key: (key[0] != Starts.RANDOM, key[1], key[2])):
        row = {"starts": starts, "size": f"{m}+{m}", "p": f"{p:g}"}
        for penalty, name in PENALTIES.items():
            runs = [r for r in groups[(starts, m, p)] if r[Column.PENALTY] == penalty]
            row[name + "success"] = mean([r[Column.SUCCESS] for r in runs])
            row[name + "repaired"] = mean([r[Column.REPAIRED] for r in runs])
            row[name + "graphs"] = sum(float(r[Column.SUCCESS]) > 0 for r in runs)
        out.append(row)
    write(tables / "ascent.csv", out)


def dsos(data, tables):
    """The DSOS bounds by level, one row per graph."""
    graphs = defaultdict(dict)
    for row in read_csv(data / File.POLYA_DSOS):
        graphs[row[Column.GRAPH]].update(
            {
                "graph": row[Column.GRAPH],
                "halfsqrtg": row[Column.HALF_SQRT_G],
                "level" + ["zero", "one", "two"][int(row[Column.LEVEL])]: row[Column.ZETA_TILDE],
            }
        )
    write(tables / "dsos.csv", list(graphs.values()))


def sampled_bounds(data):
    graphs = sampled(data)
    gap = [float(row[Column.RHO1]) - half_sqrt_g(row) for row in graphs]
    v = {"graphs": len(graphs)}
    v["rhooneexact"] = sum(is_exact(row, Column.RHO1) for row in graphs)
    v["rhooneinexact"] = len(graphs) - v["rhooneexact"]
    v["rhooneinexactsparse"] = sum(not is_exact(row, Column.RHO1) and float(row[Column.P]) == 0.3 for row in graphs)
    v["rhoonemaxgap"] = max(100 * x / half_sqrt_g(row) for row, x in zip(graphs, gap))
    v["gapbelow"] = max(x for x in gap if x < TOL)
    v["gapabove"] = min(x for x in gap if x >= TOL)
    v["solvererror"] = max(half_sqrt_g(row) - float(row[column]) for row in graphs for column in BOUNDS)
    for column, name in BOUNDS.items():
        v["round" + name] = sum(int(4 * float(row[column]) ** 2 + ROUND) == int(float(row[Column.G])) for row in graphs)
    return v


def lasserre_bounds(data):
    graphs = read_csv(data / File.BOUNDS_LASSERRE)
    v = {"lasserregraphs": len(graphs), "lasserrerhooneexact": sum(is_exact(row, Column.RHO1) for row in graphs)}
    # the one 20+20 graph on which neither rho^(1) nor the Lasserre bound is exact
    [weak] = [
        row
        for row in read_csv(data / File.bounds_lasserre_size(20))
        if not is_exact(row, Column.RHO1) and not is_exact(row, Column.HALF_SQRT_G2)
    ]
    v["weaklasserre"] = float(weak[Column.HALF_SQRT_G2])
    v["weakrhoone"] = float(weak[Column.RHO1])
    v["weakhalfsqrtg"] = half_sqrt_g(weak)
    return v


def polya_levels(data):
    levels = [row for row in read_csv(data / File.POLYA_LEVELS) if row[Column.GRAPH] == Graph.MATCHING_3]
    top = max(levels, key=lambda row: int(row[Column.LEVEL]))
    return {"polyalevel": int(top[Column.LEVEL]), "polyaratio": float(top[Column.ZETA]) / half_sqrt_g(top)}


def smallest_penalty(data):
    graphs = [row for row in read_csv(data / File.PENALTY_SMALLEST) if row[Column.FAMILY] == Family.RANDOM]
    return {
        "penaltygraphs": len(graphs),
        "penaltyratio": max(float(row[Column.GAMMA_STAR]) / float(row[Column.GAMMA1]) for row in graphs),
    }


def local_ascent(data):
    runs = read_csv(data / File.PENALTY_ASCENT)
    groups = defaultdict(list)
    for row in runs:
        groups[(row[Column.STARTS], int(row[Column.M1]), float(row[Column.P]), row[Column.PENALTY])].append(row)
    percent = lambda group, column: 100 * float(mean([row[column] for row in group]))
    select = lambda kind, keep: [group for (starts, m, p, t), group in groups.items() if starts == kind and keep(p, t)]
    starts_per_graph = lambda row: 100 if row[Column.STARTS] == Starts.RANDOM else 2 * int(row[Column.M1])
    valid = lambda p, t: t in Penalty.VALID
    v = {
        "ascentgraphs": sum(float(row[Column.SUCCESS]) > 0 for row in groups[(Starts.RANDOM, 15, 0.3, Penalty.GAMMA1)])
    }
    v["densefallback"] = min(
        percent(group, Column.FALLBACK) for group in select(Starts.RANDOM, lambda p, t: p == 0.7 and t in Penalty.VALID)
    )
    v["vertexfallback"] = max(percent(group, Column.FALLBACK) for group in select(Starts.VERTEX, valid))
    v["edgeend"] = max(
        percent(group, Column.EDGE_END) for kind in (Starts.RANDOM, Starts.VERTEX) for group in select(kind, valid)
    )
    v["edgeruns"] = sum(
        round(float(row[Column.EDGE_END]) * starts_per_graph(row))
        for row in runs
        if row[Column.PENALTY] in Penalty.VALID
    )
    v["edgerunspositive"] = sum(
        round(float(row[Column.EDGE_POSITIVE]) * starts_per_graph(row))
        for row in runs
        if row[Column.PENALTY] in Penalty.VALID
    )
    for penalty, name in ((Penalty.GAMMA1, "one"), (Penalty.GAMMA2, "two")):
        v["support" + name] = percent(groups[(Starts.RANDOM, 10, 0.3, penalty)], Column.SUPPORT)
        v["fallback" + name] = percent(groups[(Starts.RANDOM, 10, 0.3, penalty)], Column.FALLBACK)
    half_edge = [
        percent(group, Column.EDGE_END)
        for group in select(Starts.RANDOM, lambda p, t: p <= 0.5 and t == Penalty.below(0.5))
    ]
    v["halfedgemin"], v["halfedgemax"] = min(half_edge), max(half_edge)
    v["quarteredgemin"] = min(
        percent(group, Column.EDGE_END) for group in select(Starts.RANDOM, lambda p, t: t == Penalty.below(0.25))
    )
    return v


def inexact_bounds(data):
    return {
        "dualgap": max(
            abs(float(row[Column.PRIMAL]) - float(row[Column.DUAL])) for row in read_csv(data / File.BOUNDS_INEXACT)
        )
    }


def values(data, tables):
    """The numbers quoted in the text, one per row: the tolerances of the code, then counts and extremes."""
    v = {"exacttol": TOL, "roundoffset": ROUND, "solvertol": MOSEK_TOL, "bisectiontol": BISECTION_TOL}
    for numbers in (
        sampled_bounds,
        lasserre_bounds,
        polya_levels,
        smallest_penalty,
        local_ascent,
        inexact_bounds,
    ):
        v.update(numbers(data))
    write(tables / "values.csv", [{"name": name, "value": x} for name, x in v.items()])


# the data files the tables read; BOUNDS_UNBALANCED is read only when it exists
NEEDS = [
    File.BOUNDS_STRUCTURED,
    File.BOUNDS_SAMPLED,
    File.BOUNDS_INEXACT,
    File.BOUNDS_LASSERRE,
    File.bounds_lasserre_size(20),
    File.PENALTY_SMALLEST,
    File.PENALTY_ASCENT,
    File.POLYA_LEVELS,
    File.POLYA_DSOS,
]


def build(data, tables):
    require(data, NEEDS)
    tables.mkdir(parents=True, exist_ok=True)
    bounds(data, tables)
    matchings(data, tables)
    thresholds(data, tables)
    ascent(data, tables)
    dsos(data, tables)
    values(data, tables)

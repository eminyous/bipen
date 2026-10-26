"""The numbers of the tables of the paper, read from a data folder and written as CSV files to a tables folder; the
paper typesets them. Column names use letters only, as csvsimple turns them into macros. Run with: bipen tables
"""

import csv
import os
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction

import numpy as np

from bipen.graphs import random_bipartite
from bipen.sdp import MOSEK_TOL
from experiments.smallest_penalty import BISECTION_TOL

TOL = 1e-5
ROUND = 1e-6  # added to 4 b^2 before rounding a bound b down to g
BOUNDS = {"h1": "hone", "h1_dnn": "honednn", "k1": "rhoone"}
PENALTIES = {"gamma1": "gammaone", "gamma2": "gammatwo", "gamma3": "gammathree", "gamma4": "gammafour",
             "gamma_c25": "cquarter", "gamma_c50": "chalf", "gamma_c75": "cthreequarters"}


def write(tables, name, rows):
    with open(os.path.join(tables, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def mean(values):
    """Exact mean of fractions of runs (k/100 or k/(2m)), to twelve decimals, so that the paper's rounding half up
    does not depend on the order of the rows."""
    q = sum(Fraction(v).limit_denominator(1000) for v in values) / len(values)
    return (Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal("1e-12")).normalize()


def load(data):
    """Random graphs from every results file, one row per (m, k, p, seed)."""
    rows = {}
    for name in ("bounds.csv", "bounds_unbalanced.csv", "random.csv"):
        path = os.path.join(data, name)
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if r["family"] == "random" and r["k1"] not in ("", "nan"):
                    rows[(int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"]))] = r
    return list(rows.values())


def gaps(G):
    """Number of graphs, number on which each bound is exact, and mean and max gap in percent of 1/2 sqrt(g)."""
    half = np.array([float(r["half_sqrt_g"]) for r in G])
    out = {"count": len(G)}
    for key, name in BOUNDS.items():
        gap = np.array([float(r[key]) for r in G]) - half
        gap[gap < TOL] = 0.0
        out["exact" + name] = int((gap == 0).sum())
        out["mean" + name] = float((100 * gap / half).mean())
        out["max" + name] = float((100 * gap / half).max())
    return out


def sizes(data):
    """The random graphs by size, balanced sizes first, each block by shape and then by size."""
    groups = defaultdict(list)
    for r in load(data):
        groups[(int(r["m"]), int(r["k"]))].append(r)
    order = sorted(groups, key=lambda mk: (mk[0] < mk[1], mk[1] // mk[0], mk[0] + mk[1]))
    return [("balanced" if m == k else "unbalanced", m, k, groups[(m, k)]) for m, k in order]


def bounds(data, tables):
    """The bounds on the random graphs, one row per size with p = all, then one per size and edge probability."""
    rows = [{"block": b, "size": f"{m}+{k}", "p": "all", **gaps(G)} for b, m, k, G in sizes(data)]
    rows += [{"block": b, "size": f"{m}+{k}", "p": f"{p:g}", **gaps([r for r in G if float(r["p"]) == p])}
             for b, m, k, G in sizes(data) for p in sorted({float(r["p"]) for r in G})]
    write(tables, "bounds.csv", rows)


def matchings(data, tables):
    """The bounds on the perfect matchings M_r."""
    rows = [r for r in csv.DictReader(open(os.path.join(data, "bounds.csv"))) if r["family"] == "matching"]
    write(tables, "matchings.csv", [{"r": r["m"], "halfsqrtg": r["half_sqrt_g"], "hone": r["h1"], "honednn": r["h1_dnn"],
                                     "rhoone": r["k1"]} for r in sorted(rows, key=lambda r: int(r["m"]))])


def biadjacency(r):
    """The graph of a row of smallest_penalty.csv: a random graph from its seed, or N_k with its single edge."""
    m, k = int(r["m"]), int(r["k"])
    if r["family"] == "random":
        p, s = float(r["p"]), int(r["seed"])
        return random_bipartite(m, k, p, np.random.default_rng(100000 * k + 1000 * m + 100 * int(10 * p) + s))
    B = np.zeros((m, k))
    B[0, 0] = 1
    return B


def thresholds(data, tables):
    """The smallest valid penalty gamma* against the threshold (1 + sqrt g)/2, those of Gillis and Glineur and the
    spectral one, as ratios computed graph by graph: mean and max or min over the random graphs of each size, and the
    value on each N_k."""
    rows = list(csv.DictReader(open(os.path.join(data, "smallest_penalty.csv"))))
    ours = lambda r: float(r["bound_sqrt_g"])
    # Gillis-Glineur, Lemma 3: 2*gamma - 1 >= sqrt of the number of edges of the bipartite complement
    lemma3 = lambda r: (1 + np.sqrt(int(r["m"]) * int(r["k"]) - int(r["edges"]))) / 2
    spectral = lambda r: (1 + np.linalg.norm(1 - biadjacency(r), 2)) / 2
    ratios = {"gammastarover": lambda r: float(r["d_star"]) / ours(r), "gammaoneovertwo": lambda r: ours(r) / float(r["bound_gg"]),
              "gammaoneoverthree": lambda r: ours(r) / lemma3(r), "gammaoneoverfour": lambda r: ours(r) / spectral(r)}
    groups = defaultdict(list)
    for r in rows:
        if r["family"] == "random" and float(r["g"]) >= 1:
            groups[f"{r['m']}+{r['k']}"].append(r)
    for r in sorted((r for r in rows if r["family"] == "near-biclique"), key=lambda r: int(r["k"])):
        groups[r["k"]] = [r]
    out = []
    for key, G in groups.items():
        row = {"family": "random" if "+" in key else "nearbiclique", "graphs": key, "count": len(G)}
        for name, f in ratios.items():
            v = [f(r) for r in G]
            star = name == "gammastarover"
            row[name + ("one" if star else "") + "mean"] = float(np.mean(v))
            row[name + ("onemax" if star else "min")] = max(v) if star else min(v)
        out.append(row)
    write(tables, "thresholds.csv", out)


def local_ascent(data, tables):
    """Block-coordinate ascent with the seven penalties, from random starts and from the vertices: for each size, edge
    probability and penalty, the mean fraction of starts that end at a maximum pair, without and with repair, and the
    number of graphs where some start does."""
    groups = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(data, "local_ascent.csv"))):
        groups[(r["starts"], int(r["m"]), float(r["p"]))].append(r)
    out = []
    for starts, m, p in sorted(groups, key=lambda k: (k[0] != "random", k[1], k[2])):
        row = {"starts": starts, "size": f"{m}+{m}", "p": f"{p:g}"}
        for key, name in PENALTIES.items():
            G = [r for r in groups[(starts, m, p)] if r["threshold"] == key]
            row[name + "success"] = mean([r["success"] for r in G])
            row[name + "repaired"] = mean([r["repaired"] for r in G])
            row[name + "graphs"] = sum(float(r["success"]) > 0 for r in G)
        out.append(row)
    write(tables, "ascent.csv", out)


def dsos_levels(data, tables):
    """The DSOS bounds by level, one row per graph."""
    graphs = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(data, "dsos_levels.csv"))):
        graphs[r["graph"]].update({"graph": r["graph"], "halfsqrtg": r["half_sqrt_g"],
                                   "level" + ["zero", "one", "two"][int(r["r"])]: r["zeta_dsos"]})
    write(tables, "dsos.csv", list(graphs.values()))


def values(data, tables):
    """The numbers quoted in the text of the paper, one per row: counts, extremes and the tolerances of the code."""
    half = lambda r: float(r["half_sqrt_g"])
    exact = lambda r, key: float(r[key]) - half(r) < TOL
    v = {"exacttol": TOL, "roundoffset": ROUND, "solvertol": MOSEK_TOL, "bisectiontol": BISECTION_TOL}

    R = load(data)
    gap = [float(r["k1"]) - half(r) for r in R]
    v["graphs"] = len(R)
    v["rhooneexact"] = sum(exact(r, "k1") for r in R)
    v["rhooneinexact"] = len(R) - v["rhooneexact"]
    v["rhooneinexactsparse"] = sum(not exact(r, "k1") and float(r["p"]) == 0.3 for r in R)
    v["rhoonemaxgap"] = max(100 * x / half(r) for r, x in zip(R, gap))
    v["gapbelow"] = max(x for x in gap if x < TOL)
    v["gapabove"] = min(x for x in gap if x >= TOL)
    v["solvererror"] = max(half(r) - float(r[key]) for r in R for key in BOUNDS)
    for key, name in BOUNDS.items():
        v["round" + name] = sum(int(4 * float(r[key]) ** 2 + ROUND) == int(float(r["g"])) for r in R)

    L = list(csv.DictReader(open(os.path.join(data, "lasserre.csv"))))
    v["lasserregraphs"] = len(L)
    v["lasserrerhooneexact"] = sum(exact(r, "k1") for r in L)
    [w] = [r for r in csv.DictReader(open(os.path.join(data, "lasserre20.csv"))) if not exact(r, "k1") and not exact(r, "las2")]
    v["weaklasserre"], v["weakrhoone"], v["weakhalfsqrtg"] = float(w["las2"]), float(w["k1"]), half(w)

    m3 = [r for r in csv.DictReader(open(os.path.join(data, "polya_levels.csv"))) if r["graph"] == "matching-3"]
    top = max(m3, key=lambda r: int(r["r"]))
    v["polyalevel"], v["polyaratio"] = int(top["r"]), float(top["zeta"]) / half(top)

    S = [r for r in csv.DictReader(open(os.path.join(data, "smallest_penalty.csv"))) if r["family"] == "random"]
    v["penaltygraphs"] = len(S)
    v["penaltyratio"] = max(float(r["d_star"]) / float(r["bound_sqrt_g"]) for r in S)

    A = list(csv.DictReader(open(os.path.join(data, "local_ascent.csv"))))
    groups = defaultdict(list)
    for r in A:
        groups[(r["starts"], int(r["m"]), float(r["p"]), r["threshold"])].append(r)
    pct = lambda rows, col: 100 * float(mean([r[col] for r in rows]))
    large = ("gamma1", "gamma2", "gamma3", "gamma4")
    classes = lambda starts, keep: [G for (s, m, p, t), G in groups.items() if s == starts and keep(p, t)]
    v["ascentgraphs"] = sum(float(r["success"]) > 0 for r in groups[("random", 15, 0.3, "gamma1")])
    v["densefallback"] = min(pct(G, "fallback") for G in classes("random", lambda p, t: p == 0.7 and t in large))
    v["vertexfallback"] = max(pct(G, "fallback") for G in classes("vertex", lambda p, t: t in large))
    v["edgeend"] = max(pct(G, "edge") for s in ("random", "vertex") for G in classes(s, lambda p, t: t in large))
    starts = lambda r: 100 if r["starts"] == "random" else 2 * int(r["m"])
    v["edgeruns"] = sum(round(float(r["edge"]) * starts(r)) for r in A if r["threshold"] in large)
    v["edgerunspositive"] = sum(round(float(r["edge_positive"]) * starts(r)) for r in A if r["threshold"] in large)
    for t, name in (("gamma1", "one"), ("gamma2", "two")):
        v["support" + name] = pct(groups[("random", 10, 0.3, t)], "support")
        v["fallback" + name] = pct(groups[("random", 10, 0.3, t)], "fallback")
    half_edge = [pct(G, "edge") for G in classes("random", lambda p, t: p <= 0.5 and t == "gamma_c50")]
    v["halfedgemin"], v["halfedgemax"] = min(half_edge), max(half_edge)
    v["quarteredgemin"] = min(pct(G, "edge") for G in classes("random", lambda p, t: t == "gamma_c25"))

    K = list(csv.DictReader(open(os.path.join(data, "k1_check.csv"))))
    v["dualgap"] = max(abs(float(r["primal"]) - float(r["dual"])) for r in K)
    write(tables, "values.csv", [{"name": k, "value": x} for k, x in v.items()])


def build(data, tables):
    os.makedirs(tables, exist_ok=True)
    bounds(data, tables)
    matchings(data, tables)
    thresholds(data, tables)
    local_ascent(data, tables)
    dsos_levels(data, tables)
    values(data, tables)

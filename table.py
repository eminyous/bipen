"""Table 1 of the note (booktabs), written to data/table1.tex from the random graphs in data/bounds.csv,
data/bounds_unbalanced.csv and data/random.csv.

Run from code/:  python table.py
"""

import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-5


# decimals of the gap columns: mean h1, mean h1_dnn, max h1, max h1_dnn, max k1
DECIMALS = [1, 2, 1, 1, 3]


def fixed(x, d):
    """x with d decimals; an exact 0 is padded so that the decimal points stay aligned."""
    return rf"0\phantom{{.{'0' * d}}}" if x == 0 else f"{x:.{d}f}"


def load():
    """Random graphs from every results file, one row per (m, k, p, seed)."""
    rows = {}
    for name in ("bounds.csv", "bounds_unbalanced.csv", "random.csv"):
        path = os.path.join(ROOT, "data", name)
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                if r["family"] == "random" and r["k1"] not in ("", "nan"):
                    rows[(int(r["m"]), int(r["k"]), float(r["p"]), int(r["seed"]))] = r
    return list(rows.values())


def cells(G):
    half = np.array([float(r["half_sqrt_g"]) for r in G])
    gaps = {}
    for key in ("h1", "h1_dnn", "k1"):
        abs_gap = np.array([float(r[key]) for r in G]) - half
        abs_gap[abs_gap < TOL] = 0.0
        gaps[key] = (abs_gap, 100 * abs_gap / half)
    exact = [f"${int((gaps[k][0] == 0).sum())}$" for k in ("h1", "h1_dnn", "k1")]
    values = [gaps["h1"][1].mean(), gaps["h1_dnn"][1].mean(),
              gaps["h1"][1].max(), gaps["h1_dnn"][1].max(), gaps["k1"][1].max()]
    return [f"${len(G)}$"] + exact + [f"${fixed(v, d)}$" for v, d in zip(values, DECIMALS)]


def main():
    rows = load()
    sizes = defaultdict(list)
    for r in rows:
        sizes[(int(r["m"]), int(r["k"]))].append(r)
    out = [r"\begin{table}[ht]", r"\centering\footnotesize\setlength{\tabcolsep}{4.5pt}",
           r"\caption{Random bipartite graphs with $|V_1|=m_1$ and $|V_2|=m_2$, for each size and edge probability $p$, "
           r"and for all three values of $p$ together; \# is the number of graphs. "
           r"Columns 4--6: number of graphs on which the bound is within $10^{-5}$ of $\frac12\sqrt g$. "
           r"Columns 7--11: relative gap $(\text{bound}-\frac12\sqrt g)/\frac12\sqrt g$ in percent, "
           r"mean and maximum over the graphs of the row.}\label{tbl-random}",
           r"\begin{tabular}{@{}rrrrrrrrrrr@{}}", r"\toprule",
           r"& & & \multicolumn{3}{c}{exact} & \multicolumn{2}{c}{mean gap (\%)} & \multicolumn{3}{c}{max gap (\%)}\\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(l){9-11}",
           r"$m_1+m_2$ & $p$ & \# & $h_1$ & $\eta_1(G;\DNN)$ & $\kappa^{(1)}$ & $h_1$ & $\eta_1(G;\DNN)$ & $h_1$ & $\eta_1(G;\DNN)$ & $\kappa^{(1)}$\\"]
    for label, keep in (("balanced, $m_1=m_2$", lambda m, k: m == k), ("unbalanced, $m_1<m_2$", lambda m, k: m < k)):
        block = sorted((mk for mk in sizes if keep(*mk)), key=lambda mk: (mk[1] // mk[0], mk[0] + mk[1]))
        if not block:
            continue
        out += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{label}}}}}\\"]
        for i, (m, k) in enumerate(block):
            if i:
                out.append(r"\addlinespace")
            G = sizes[(m, k)]
            for j, p in enumerate(sorted({float(r["p"]) for r in G})):
                first = f"${m}+{k}$" if j == 0 else ""
                out.append(" & ".join([first, f"${p:g}$"] + cells([r for r in G if float(r["p"]) == p])) + r"\\")
            out.append(" & ".join(["", r"\emph{all}"] + cells(G)) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "table1.tex"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

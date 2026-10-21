"""Tables 1 to 4 and B.1 of the note (booktabs), written to data/table1.tex to table4.tex and tableB1.tex.

Table 1 is from the random graphs in data/bounds.csv,
data/bounds_unbalanced.csv and data/random.csv.

Run from code/:  python table.py
"""

import csv
import os
from collections import defaultdict

import numpy as np

from bicop.graphs import random_bipartite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-5


# decimals of the gap columns: mean h1, mean h1_dnn, mean k1, max h1, max h1_dnn, max k1
DECIMALS = [1, 2, 3, 1, 1, 3]


def fixed(x, d):
    """x with d decimals; an exact 0 is padded so that the decimal points stay aligned."""
    if x == 0:
        return rf"0\phantom{{.{'0' * d}}}"
    if x < 10 ** -d:
        return rf"{{<}}0.{'0' * (d - 1)}1"  # positive but below the last printed digit
    return f"{x:.{d}f}"


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
    values = [gaps["h1"][1].mean(), gaps["h1_dnn"][1].mean(), gaps["k1"][1].mean(),
              gaps["h1"][1].max(), gaps["h1_dnn"][1].max(), gaps["k1"][1].max()]
    return [f"${len(G)}$"] + exact + [f"${fixed(v, d)}$" for v, d in zip(values, DECIMALS)]


HEADER = (r"\# & $h_1$ & $\eta_1(\cdot;\DNN)$ & $\kappa^{(1)}$ & $h_1$ & $\eta_1(\cdot;\DNN)$ & $\kappa^{(1)}$ & $h_1$ & $\eta_1(\cdot;\DNN)$ & $\kappa^{(1)}$\\")
GAPS = (r"relative gap $(\text{bound}-\frac12\sqrt g)/\frac12\sqrt g$ in percent, mean and maximum over the graphs of the row")
BLOCKS = (("balanced, $m_1=m_2$", lambda m, k: m == k), ("unbalanced, $m_1<m_2$", lambda m, k: m < k))


def by_size():
    sizes = defaultdict(list)
    for r in load():
        sizes[(int(r["m"]), int(r["k"]))].append(r)
    return sizes


def blocks(sizes):
    for label, keep in BLOCKS:
        block = sorted((mk for mk in sizes if keep(*mk)), key=lambda mk: (mk[1] // mk[0], mk[0] + mk[1]))
        if block:
            yield label, block


def main():
    """Table 1 (Section 7): one row per size, the three edge probabilities together."""
    sizes = by_size()
    out = [r"\begin{table}[H]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
           r"\caption{Random bipartite graphs with $|V_1|=m_1$ and $|V_2|=m_2$, with the edge probabilities $p=0.3$, $0.5$ and "
           r"$0.7$ together (Table~\ref{tbl-random-p} separates them); \# is the number of graphs. "
           r"Columns 3--5: number of graphs on which the bound is within $10^{-5}$ of $\frac12\sqrt g$. "
           r"Columns 6--11: " + GAPS + r".}\label{tbl-random}",
           r"\begin{tabular}{@{}rrrrrrrrrrr@{}}", r"\toprule",
           r"& & \multicolumn{3}{c}{exact} & \multicolumn{3}{c}{mean gap (\%)} & \multicolumn{3}{c}{max gap (\%)}\\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
           r"$m_1+m_2$ & " + HEADER]
    for label, block in blocks(sizes):
        out += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{label}}}}}\\"]
        out += [" & ".join([f"${m}+{k}$"] + cells(sizes[(m, k)])) + r"\\" for m, k in block]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "table1.tex"), "w").write("\n".join(out) + "\n")


def per_probability():
    """Table B.1 (Appendix B): the same, for each size and each edge probability."""
    sizes = by_size()
    out = [r"\begin{table}[p]", r"\centering\footnotesize\setlength{\tabcolsep}{3.2pt}",
           r"\caption{The random graphs of Table~\ref{tbl-random}, for each size and each edge probability $p$; "
           r"the three probabilities together are in Table~\ref{tbl-random}. "
           r"Columns 4--6: number of graphs on which the bound is within $10^{-5}$ of $\frac12\sqrt g$. "
           r"Columns 7--12: " + GAPS + r".}\label{tbl-random-p}",
           r"\begin{tabular}{@{}rrrrrrrrrrrr@{}}", r"\toprule",
           r"& & & \multicolumn{3}{c}{exact} & \multicolumn{3}{c}{mean gap (\%)} & \multicolumn{3}{c}{max gap (\%)}\\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-12}",
           r"$m_1+m_2$ & $p$ & " + HEADER]
    for label, block in blocks(sizes):
        out += [r"\midrule", rf"\multicolumn{{12}}{{@{{}}l}}{{\emph{{{label}}}}}\\"]
        for i, (m, k) in enumerate(block):
            if i:
                out.append(r"\addlinespace")
            G = sizes[(m, k)]
            for j, p in enumerate(sorted({float(r["p"]) for r in G})):
                first = f"${m}+{k}$" if j == 0 else ""
                out.append(" & ".join([first, f"${p:g}$"] + cells([r for r in G if float(r["p"]) == p])) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "tableB1.tex"), "w").write("\n".join(out) + "\n")


def matchings():
    """Table 2: the bounds on the perfect matchings M_r."""
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "bounds.csv"))) if r["family"] == "matching"]
    out = [r"\begin{table}[H]", r"\centering\small",
           r"\caption{The bounds on the perfect matchings $M_r$, $2\le r\le12$, on which one of them is not exact, that is, for odd $r$; "
           r"there $\frac12\sqrt g=\frac14\sqrt{r^2-1}$, while for even $r$ all bounds equal $r/4$.}\label{tbl-matchings}",
           r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
           r"$r$ & $\frac12\sqrt g$ & $h_1$ & $\eta_1(\cdot;\DNN)$ & $\kappa^{(1)}$\\", r"\midrule"]
    for r in sorted(rows, key=lambda r: int(r["m"])):
        vals = [float(r[k]) for k in ("half_sqrt_g", "h1", "h1_dnn", "k1")]
        if max(vals[1:]) - vals[0] < TOL:
            continue  # every bound is exact
        out.append(" & ".join([f"${r['m']}$"] + [f"${v:.4f}$" for v in vals]) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "table2.tex"), "w").write("\n".join(out) + "\n")


def biadjacency(r):
    """The graph of a row of partA_threshold.csv: a random graph from its seed, or N_k with its single edge."""
    m, k = int(r["m"]), int(r["k"])
    if r["family"] == "random":
        p, s = float(r["p"]), int(r["seed"])
        return random_bipartite(m, k, p, np.random.default_rng(100000 * k + 1000 * m + 100 * int(10 * p) + s))
    B = np.zeros((m, k))
    B[0, 0] = 1
    return B


def thresholds():
    """Table 3: the smallest valid penalty gamma* against the thresholds of Theorem 4.3 and of Gillis and Glineur."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "partA_threshold.csv"))))
    ours = lambda r: float(r["bound_sqrt_g"])
    # Gillis-Glineur, Lemma 3: 2*gamma - 1 >= sqrt of the number of edges of the bipartite complement
    lemma3 = lambda r: (1 + np.sqrt(int(r["m"]) * int(r["k"]) - int(r["edges"]))) / 2
    spectral = lambda r: (1 + np.linalg.norm(1 - biadjacency(r), 2)) / 2
    out = [r"\begin{table}[H]", r"\centering\small",
           r"\caption{The smallest valid penalty $\gamma^*$ against the threshold $\gamma_1=\left(1+\sqrt g\right)/2$ of "
           r"Theorem~\ref{thm-penalty}, the thresholds $\gamma_2=\left(1+\max\left(m_1,m_2\right)\right)/2$ and $\gamma_3=\bigl(1+\sqrt{|E(\widehat G)|}\bigr)/2$ "
           r"of \cite[Theorem 2 and Lemma 3]{gillis2014continuous}, and $\gamma_4=\left(1+\|J-W\|_2\right)/2$, "
           r"on random graphs ($100$ for each edge probability $p=0.2$, $0.4$, $0.6$ and $0.8$, without the empty and complete ones) and on the graphs $N_k$ "
           r"of Proposition~\ref{prp-sharp}.}\label{tbl-threshold}",
           r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
           r"& & \multicolumn{2}{c}{$\gamma^*/\gamma_1$} & \multicolumn{2}{c}{$\gamma_1/\gamma_2$} & \multicolumn{2}{c}{$\gamma_1/\gamma_3$}"
           r" & \multicolumn{2}{c}{$\gamma_1/\gamma_4$}\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
           r"$G$ & \# & mean & max & mean & min & mean & min & mean & min\\", r"\midrule"]
    groups = defaultdict(list)
    for r in rows:
        if r["family"] == "random" and float(r["g"]) >= 1:
            groups[(int(r["m"]), int(r["k"]))].append(r)
    for m, k in sorted(groups):
        G = groups[(m, k)]
        a = [float(r["d_star"]) / ours(r) for r in G]
        b = [ours(r) / float(r["bound_gg"]) for r in G]
        c = [ours(r) / lemma3(r) for r in G]
        d = [ours(r) / spectral(r) for r in G]
        out.append(f"${m}+{k}$ & ${len(G)}$ & ${np.mean(a):.3f}$ & ${max(a):.3f}$ & ${np.mean(b):.3f}$ & ${min(b):.3f}$"
                   f" & ${np.mean(c):.3f}$ & ${min(c):.3f}$ & ${np.mean(d):.3f}$ & ${min(d):.3f}$" + r"\\")
    out.append(r"\midrule")
    for r in sorted((r for r in rows if r["family"] == "near-biclique"), key=lambda r: int(r["k"])):
        a = float(r["d_star"]) / ours(r)
        b = ours(r) / float(r["bound_gg"])
        c = ours(r) / lemma3(r)
        d = ours(r) / spectral(r)
        out.append(f"$N_{{{r['k']}}}$ & & \\multicolumn{{2}}{{c}}{{${a:.3f}$}} & \\multicolumn{{2}}{{c}}{{${b:.3f}$}}"
                   f" & \\multicolumn{{2}}{{c}}{{${c:.3f}$}} & \\multicolumn{{2}}{{c}}{{${d:.3f}$}}" + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "table3.tex"), "w").write("\n".join(out) + "\n")


def ascent():
    """Table 4: block-coordinate ascent on (7) from 100 random starts, for the four penalties."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "local_ascent.csv"))))
    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["m"]), float(r["p"]), r["threshold"])].append(float(r["success"]))
    names = ["gammak", "gamma4", "gamma3", "gamma2"]
    labels = {"gammak": r"\gamma_\kappa", "gamma4": r"\gamma_4", "gamma3": r"\gamma_3", "gamma2": r"\gamma_2"}
    out = [r"\begin{table}[H]", r"\centering\small\setlength{\tabcolsep}{5pt}",
           r"\caption{Block-coordinate ascent on \eqref{eq-sigma} from $100$ random starts, on the first $20$ random graphs "
           r"of each size and edge probability, with the penalty $\gamma_\kappa=\frac12+\kappa^{(1)}$ and the penalties "
           r"$\gamma_4$, $\gamma_3$ and $\gamma_2$ of Table~\ref{tbl-threshold}: the fraction of starts that end at a maximum "
           r"biindependent pair, and the number of graphs on which at least one start does.}\label{tbl-ascent}",
           r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
           "& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${labels[t]}$}}" for t in names) + r"\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
           r"$G$ & $p$" + r" & starts & graphs" * 4 + r"\\", r"\midrule"]
    for m in sorted({k[0] for k in groups}):
        for p in sorted({k[1] for k in groups}):
            cells = [f"${np.mean(groups[(m, p, t)]):.3f}$ & ${sum(x > 0 for x in groups[(m, p, t)])}$" for t in names]
            out.append(f"${m}+{m}$ & ${p}$ & " + " & ".join(cells) + r"\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "data", "table4.tex"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
    per_probability()
    matchings()
    thresholds()
    ascent()

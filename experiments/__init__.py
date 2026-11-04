"""The experiments of the paper, by family, in the order of the paper; `bipen run <family>/<name>` runs one. Each
writes its results to the data folder (file names in bipen.data), from which `bipen tables` and `bipen plots` compute
the numbers of the paper.
"""

NAMES = [
    "bounds/structured",
    "bounds/sampled",
    "bounds/inexact",
    "bounds/lasserre",
    "bounds/solvers",
    "bounds/dnn",
    "bounds/certificates",
    "penalty/smallest",
    "penalty/ascent",
    "polya/finiteness",
    "polya/levels",
    "polya/dsos",
]

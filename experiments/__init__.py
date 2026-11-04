"""The experiments of the paper, one module each, in the order of the paper; `bipen run <name>` runs one, with the
module name written with hyphens. Each writes its results to the data folder (file names in bipen.data), from which
`bipen tables` and `bipen plots` compute the numbers of the paper.
"""

NAMES = [
    "structured_bounds",
    "random_bounds",
    "rho_gaps",
    "lasserre_comparison",
    "solver_comparison",
    "dnn_exactness",
    "five_vertex_certificates",
    "smallest_penalty",
    "local_ascent",
    "polya_first_finite_level",
    "polya_bounds",
    "dsos_bounds",
]

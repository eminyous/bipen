# bipen

Code of the paper "Exact penalties for maximum-edge bicliques in bipartite graphs" (Y. Emine, B. Ghaddar).

## Install

```bash
uv sync
source .venv/bin/activate
pytest
```

The solvers use MOSEK, which needs a license. To use Clarabel instead, set `BIPEN_SOLVER=clarabel`.

## Workspace

`bipen` writes everything to a workspace folder: the results of the experiments to `data/`, and the numbers of the
paper's tables and figure to `tables/` and `figures/`. The workspace is `./workspace`, which git ignores; to put it
elsewhere, pass `--workspace DIR` or set `BIPEN_WORKSPACE=DIR`.

## Reproduce

```bash
bipen run bounds/structured
bipen run bounds/sampled --workers 12
bipen run bounds/inexact --workers 4
bipen run bounds/lasserre --workers 4
bipen run bounds/lasserre --workers 4 --size 20 --seeds 3
bipen run bounds/solvers --workers 4
bipen run bounds/dnn
bipen run bounds/certificates
bipen run penalty/smallest --workers 6
bipen run penalty/ascent --workers 6
bipen run polya/finiteness
bipen run polya/levels
bipen run polya/dsos
bipen tables
bipen plots
```

The runs are in the order of `bipen list`: `bounds/inexact` and `penalty/ascent` read `bounds/sampled`, and
`bounds/solvers` reads `bounds/inexact`. `bipen tables` and `bipen plots` then write the numbers of the paper.
`--quiet` hides the warnings of the solvers.

Measured in CPU hours with MOSEK: `bounds/sampled` 59, of which the 20+20 graphs take 5 GB each; `bounds/lasserre` 1, and
4 at size 20; `bounds/inexact` 2; `bounds/structured` and `bounds/solvers` less than 1.

## Layout

- `bipen/`: the bounds, the penalty formulation and the exact values of g, h and α*.
- `bipen/data.py`: the names of the data files and of their columns.
- `experiments/bounds/`, `experiments/penalty/`, `experiments/polya/`: one module per experiment; the
  workspace's `data/` has the same layout.

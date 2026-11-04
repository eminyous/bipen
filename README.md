# bipen

Code of the paper "Exact penalties for maximum-edge bicliques in bipartite graphs" (Y. Emine, B. Ghaddar).

## Install

```bash
uv sync
source .venv/bin/activate
pytest
```

The solvers use MOSEK, which needs a license. To use Clarabel instead, set `BIPEN_SOLVER=clarabel`.

## Run

```bash
bipen list
bipen run bounds/sampled --workers 12
bipen tables
bipen plots
```

- `bipen list` names the experiments, in the order of the paper.
- `bipen run <experiment>` runs one and writes its results to `data/`.
- `bipen tables` and `bipen plots` write the numbers of the tables and of the figure to `tables/` and `figures/`, where
  the paper reads them.

The folders are those of the paper folder that contains this code; `--data`, `--tables` and `--figures` choose others.
`--quiet` hides the warnings of the solvers.

## Layout

- `bipen/`: the bounds, the penalty formulation and the exact values, in the notation of the paper.
- `bipen/data.py`: the names of the data files and of their columns.
- `experiments/bounds/`, `experiments/penalty/`, `experiments/polya/`: one module per experiment; `data/` has the same
  layout.

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

- `bipen list` names the experiments.
- `bipen run <experiment>` runs one and writes its results to `data/`.
- `bipen tables` and `bipen plots` write the numbers of the paper's tables and figure to `tables/` and `figures/`.

`data/`, `tables/` and `figures/` are the folders next to `code/`; `--data`, `--tables` and `--figures` change them.
`--quiet` hides the warnings of the solvers.

## Layout

- `bipen/`: the bounds, the penalty formulation and the exact values of g, h and α*.
- `bipen/data.py`: the names of the data files and of their columns.
- `experiments/bounds/`, `experiments/penalty/`, `experiments/polya/`: one module per experiment; `data/` has the same
  layout.

# bipen

Code of the paper "Exact penalties for maximum-edge bicliques in bipartite graphs" (Y. Emine, B. Ghaddar).

```bash
uv sync
source .venv/bin/activate
pytest
bipen list
bipen run bounds/sampled --workers 12
bipen tables
bipen plots
```

`bipen list` names the experiments in the order of the paper, and `bipen run <experiment>` runs one of them. Results go to `data/` of the paper folder (file and column names in `bipen/data.py`), and the numbers of the tables and of the figure go as CSV files to `tables/` and `figures/`, which the paper reads; `--data`, `--tables` and `--figures` choose other folders. MOSEK needs a license; `BIPEN_SOLVER=clarabel` uses Clarabel instead.

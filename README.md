# bipen

Code of the paper "Penalty thresholds for the maximum-edge biclique problem" (Y. Emine, B. Ghaddar).

```bash
uv sync
source .venv/bin/activate
pytest
bipen list
bipen run random-graphs --workers 12
bipen tables
bipen plots
```

`bipen list` names the experiments and their output files, and `bipen run <experiment>` runs one of them. Results go to `data/` of the paper folder, tables to `tables/` and the figure to `figures/`; `--data`, `--tables` and `--figures` choose other folders. MOSEK needs a license; `BIPEN_SOLVER=clarabel` uses Clarabel instead.

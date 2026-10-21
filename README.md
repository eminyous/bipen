# bicop

Code of the paper "Penalty thresholds for the maximum-edge biclique problem" (Y. Emine, B. Ghaddar).

```bash
uv sync
source .venv/bin/activate
pytest
bicop list
bicop run random-graphs --workers 12
bicop tables
bicop plots
```

`bicop list` names the experiments and their output files, and `bicop run <experiment>` runs one of them. Results go to `data/` of the paper folder, tables to `tables/` and the figure to `figures/`; `--data`, `--tables` and `--figures` choose other folders. MOSEK needs a license; `BICOP_SOLVER=clarabel` uses Clarabel instead.

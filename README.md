# bicop

Code of the paper "Completely positive and copositive formulations for biindependent pairs in bipartite graphs" (Y. Emine, B. Ghaddar). Run from this folder; outputs go to `../data/`.

```bash
uv sync
source .venv/bin/activate
pytest
python -m experiments.bounds_table
python -m experiments.random_graphs 12
python -m experiments.k1_check 4
python -m experiments.solver_bench 8
python -m experiments.partA_threshold
python -m experiments.smallest_dnn_gap
python -m experiments.polya_finiteness
python -m experiments.polya_levels
python table.py
python plots.py
```

- `bounds_table`: bounds on the structured graphs and the first random ones
- `random_graphs`: the 3000 random graphs (about 60 CPU hours)
- `k1_check`: the graphs where κ⁽¹⁾ is not exact, at tolerance 1e-10
- `solver_bench`: MOSEK against Clarabel
- `partA_threshold`: the smallest valid penalty γ*
- `smallest_dnn_gap`: small graphs where the DNN bound is not exact
- `polya_finiteness`, `polya_levels`: the Pólya bounds
- `table.py`, `plots.py`: the tables and Figure 1, without a solver

MOSEK needs a license; `BICOP_SOLVER=clarabel` uses Clarabel instead.

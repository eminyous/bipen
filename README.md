# Completely positive and copositive formulations for biindependent pairs in bipartite graphs

Code (`bicop`) for the paper by Y. Emine and B. Ghaddar. Run it from this folder inside the paper repository; the results go to `../data/`.

```bash
uv sync
source .venv/bin/activate
pytest
python -m experiments.bounds_table
python -m experiments.random_graphs 12
python -m experiments.k1_check 4
python -m experiments.partA_threshold
python -m experiments.smallest_dnn_gap
python -m experiments.polya_finiteness
python -m experiments.polya_levels
python table.py
python plots.py
```

`random_graphs` solves the 3000 random graphs, with as many workers as its argument: about 60 CPU hours, and 5 GB per 20+20 graph. `table.py` writes Tables 1 to 3 and `plots.py` Figure 1, without a solver.

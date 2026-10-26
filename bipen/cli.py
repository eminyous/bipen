"""The bipen command: run the experiments of the paper and write the numbers of its tables and figure.

The folders default to data/, tables/ and figures/ of the paper folder that contains this code.
"""

import argparse
import importlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "bounds": "h1, h1_dnn and kappa^(1) on the structured graphs and the first random ones (bounds.csv)",
    "random-graphs": "the 3000 random graphs, about 60 CPU hours (random.csv)",
    "k1-check": "the random graphs where kappa^(1) is not exact, at tolerance 1e-10 (k1_check.csv)",
    "lasserre-compare": "kappa^(1) against the Lasserre bound g_2 (lasserre.csv, or lasserre<size>.csv)",
    "local-ascent": "block-coordinate ascent with seven penalties and two kinds of starts (local_ascent.csv)",
    "smallest-penalty": "the smallest valid penalty gamma* (smallest_penalty.csv)",
    "smallest-dnn-gap": "small graphs where the DNN bound is not exact (smallest_dnn_gap.txt)",
    "polya-finiteness": "the first finite Polya level against alpha* - 1 (polya_finiteness.csv)",
    "polya-levels": "the Polya bounds by level (polya_levels.csv)",
    "dsos-levels": "the DSOS bounds by level (dsos_levels.csv)",
    "solver-bench": "MOSEK against Clarabel (solver_bench.csv)",
    "five-vertices": "exact check of the five-vertex certificates (prints only)",
}


def run(args):
    module = importlib.import_module("experiments." + args.experiment.replace("-", "_"))
    given = {"data": str(args.data), "workers": args.workers, "size": args.size, "seeds": args.seeds}
    accepted = inspect.signature(module.main).parameters
    for name in ("workers", "size", "seeds"):
        if given[name] is not None and name not in accepted:
            raise SystemExit(f"{args.experiment} takes no --{name}")
    args.data.mkdir(parents=True, exist_ok=True)
    module.main(**{k: v for k, v in given.items() if k in accepted and v is not None})


def main():
    parser = argparse.ArgumentParser(prog="bipen", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments")
    p = sub.add_parser("run", help="run one experiment")
    p.add_argument("experiment", choices=EXPERIMENTS)
    p.add_argument("--workers", type=int, help="number of parallel processes")
    p.add_argument("--size", type=int, help="lasserre-compare only: a single balanced size")
    p.add_argument("--seeds", type=int, help="lasserre-compare only: graphs per size and probability")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are written and read")
    p = sub.add_parser("tables", help="write the numbers of the tables of the paper")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are read")
    p.add_argument("--tables", type=Path, default=ROOT / "tables", help="where the table data are written")
    p = sub.add_parser("plots", help="write the numbers of the figure of the paper")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are read")
    p.add_argument("--figures", type=Path, default=ROOT / "figures", help="where the figure data are written")
    args = parser.parse_args()

    if args.command == "list":
        for name, what in EXPERIMENTS.items():
            print(f"{name:18} {what}")
    elif args.command == "run":
        run(args)
    elif args.command == "tables":
        from bipen import tables

        tables.build(str(args.data), str(args.tables))
    else:
        from bipen import plots

        plots.build(str(args.data), str(args.figures))

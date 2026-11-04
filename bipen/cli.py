"""The bipen command: run the experiments of the paper and write the numbers of its tables and figure.

The folders default to data/, tables/ and figures/ of the paper folder that contains this code.
"""

import argparse
import importlib
import inspect
import os
import warnings
from pathlib import Path

from experiments import NAMES

ROOT = Path(__file__).resolve().parents[2]


def experiment(name):
    """The module of an experiment, from its name family/name."""
    return importlib.import_module("experiments." + name.replace("/", "."))


def run(args):
    if args.quiet:
        # the workers of the experiments inherit the filter, or read the variable when they are spawned
        os.environ["PYTHONWARNINGS"] = "ignore"
        warnings.filterwarnings("ignore")
    module = experiment(args.experiment)
    given = {"data": args.data, "workers": args.workers, "size": args.size, "seeds": args.seeds}
    accepted = inspect.signature(module.main).parameters
    for option in ("workers", "size", "seeds"):
        if given[option] is not None and option not in accepted:
            raise SystemExit(f"{args.experiment} takes no --{option}")
    args.data.mkdir(parents=True, exist_ok=True)
    module.main(**{key: value for key, value in given.items() if key in accepted and value is not None})


def main():
    parser = argparse.ArgumentParser(prog="bipen", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments, in the order of the paper")
    p = sub.add_parser("run", help="run one experiment")
    p.add_argument("experiment", choices=NAMES)
    p.add_argument("--workers", type=int, help="number of parallel processes")
    p.add_argument("--quiet", action="store_true", help="hide the warnings of the solvers and of numpy")
    p.add_argument("--size", type=int, help="bounds/lasserre only: a single balanced size")
    p.add_argument("--seeds", type=int, help="bounds/lasserre only: graphs per size and probability")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are written and read")
    p = sub.add_parser("tables", help="write the numbers of the tables of the paper")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are read")
    p.add_argument("--tables", type=Path, default=ROOT / "tables", help="where the table data are written")
    p = sub.add_parser("plots", help="write the numbers of the figure of the paper")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="where the results are read")
    p.add_argument("--figures", type=Path, default=ROOT / "figures", help="where the figure data are written")
    args = parser.parse_args()

    if args.command == "list":
        for name in NAMES:
            print(f"{name:22} {experiment(name).__doc__.splitlines()[0]}")
    elif args.command == "run":
        run(args)
    elif args.command == "tables":
        from bipen import tables

        tables.build(args.data, args.tables)
    else:
        from bipen import plots

        plots.build(args.data, args.figures)

"""The bipen command: run the experiments of the paper and write the numbers of its tables and figure.

Everything is written to a workspace folder: the results of the experiments to data/, and the numbers of the tables
and of the figure to tables/ and figures/. The workspace is ./workspace unless --workspace or BIPEN_WORKSPACE says
otherwise.
"""

import argparse
import importlib
import inspect
import os
import warnings
from pathlib import Path

from experiments import NAMES

WORKSPACE = Path(os.environ.get("BIPEN_WORKSPACE", "workspace"))


def experiment(name):
    """The module of an experiment, from its name family/name."""
    return importlib.import_module("experiments." + name.replace("/", "."))


def run(args, data):
    if args.quiet:
        # the workers of the experiments inherit the filter, or read the variable when they are spawned
        os.environ["PYTHONWARNINGS"] = "ignore"
        warnings.filterwarnings("ignore")
    module = experiment(args.experiment)
    given = {"data": data, "workers": args.workers, "size": args.size, "seeds": args.seeds}
    accepted = inspect.signature(module.main).parameters
    for option in ("workers", "size", "seeds"):
        if given[option] is not None and option not in accepted:
            raise SystemExit(f"{args.experiment} takes no --{option}")
    data.mkdir(parents=True, exist_ok=True)
    module.main(**{key: value for key, value in given.items() if key in accepted and value is not None})


def main():
    workspace = argparse.ArgumentParser(add_help=False)
    workspace.add_argument("--workspace", type=Path, default=WORKSPACE, help=f"default: {WORKSPACE}")
    parser = argparse.ArgumentParser(prog="bipen", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments, in the order to run them")
    p = sub.add_parser("run", parents=[workspace], help="run one experiment")
    p.add_argument("experiment", choices=NAMES)
    p.add_argument("--workers", type=int, help="number of parallel processes")
    p.add_argument("--quiet", action="store_true", help="hide the warnings of the solvers and of numpy")
    p.add_argument("--size", type=int, help="bounds/lasserre only: a single balanced size")
    p.add_argument("--seeds", type=int, help="bounds/lasserre only: graphs per size and probability")
    sub.add_parser("tables", parents=[workspace], help="write the numbers of the tables of the paper")
    sub.add_parser("plots", parents=[workspace], help="write the numbers of the figure of the paper")
    args = parser.parse_args()

    if args.command == "list":
        for name in NAMES:
            print(f"{name:22} {experiment(name).__doc__.splitlines()[0]}")
        return
    data, tables, figures = (args.workspace / folder for folder in ("data", "tables", "figures"))
    if args.command == "run":
        run(args, data)
    elif args.command == "tables":
        from bipen import tables as table_numbers

        table_numbers.build(data, tables)
    else:
        from bipen import plots

        plots.build(data, figures)

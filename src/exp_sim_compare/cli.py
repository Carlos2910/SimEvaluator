from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run_pipeline, run_selected_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp-sim-compare",
        description="Compare experimental test curves with simulation curves.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full comparison pipeline.")
    run.add_argument("config", help="Path to YAML config.")
    run.add_argument("--no-plots", action="store_true", help="Skip diagnostic plot generation.")

    plot = sub.add_parser("plot-selected", help="Plot preselected experimental/simulation cases.")
    plot.add_argument("config", help="Path to YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "run":
            run_pipeline(config, make_plots=not args.no_plots)
            return 0
        if args.command == "plot-selected":
            run_selected_plots(config)
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

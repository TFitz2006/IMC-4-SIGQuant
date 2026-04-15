from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imc_matplotlib"))

import matplotlib
if "--show" not in sys.argv[1:]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualizer.charts import create_price_dashboard, launch_interactive_dashboard
from visualizer.data_loader import (
    DEFAULT_ROUND1_ROOT,
    DEFAULT_LOGS_ROOT,
    load_round1_data,
    discover_official_logs,
    build_run_payload,
)
from visualizer.order_book import create_order_book_snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round 1 visualizer for the IMC Prosperity 4 DataCapsules dataset."
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROUND1_ROOT),
        help="Path to the round data capsule root. Defaults to DataCapsules/ROUND1.",
    )

    subparsers = parser.add_subparsers(dest="command")

    dashboard_parser = subparsers.add_parser("dashboard", help="Render a price/spread/volume dashboard.")
    dashboard_parser.add_argument("--product", action="append", dest="products", help="Filter to one or more products.")
    dashboard_parser.add_argument("--day", type=int, action="append", dest="days", help="Filter to one or more day values.")
    dashboard_parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help=f"Path to official logs directory. Defaults to {DEFAULT_LOGS_ROOT} if it exists.",
    )
    dashboard_parser.add_argument(
        "--output-log",
        type=Path,
        action="append",
        dest="output_logs",
        default=None,
        metavar="PATH",
        help=(
            "Path to an additional output log to include in the run dropdown. "
            "Can be a single run directory (containing .json/.log files) or a root "
            "directory of multiple run folders. Use multiple times to add more."
        ),
    )
    dashboard_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run folder name to load (e.g., 'Run1(137859)'). Uses latest if not specified.",
    )
    dashboard_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the rendered dashboard.",
    )
    dashboard_parser.add_argument("--show", action="store_true", help="Display the figure interactively after rendering.")

    snapshot_parser = subparsers.add_parser("snapshot", help="Render one order-book snapshot.")
    snapshot_parser.add_argument("--product", required=True, help="Product name, for example ASH_COATED_OSMIUM.")
    snapshot_parser.add_argument("--day", type=int, required=True, help="Day to render, for example -2, -1, or 0.")
    snapshot_parser.add_argument("--timestamp", type=int, required=True, help="Timestamp within the selected day.")
    snapshot_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the rendered snapshot.",
    )
    snapshot_parser.add_argument("--show", action="store_true", help="Display the figure interactively after rendering.")

    return parser


def _write_figure(figure, output_path: Path | None, show: bool) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def _resolve_output_log_dirs(path: Path) -> list[Path]:
    """
    Given a --output-log path, return the list of run directories it represents.
    If the path itself contains .json/.log files it is treated as a single run dir.
    Otherwise it is treated as a root containing multiple run sub-directories.
    """
    if not path.exists():
        print(f"Warning: --output-log path does not exist: {path}")
        return []
    json_files = list(path.glob("*.json"))
    log_files = list(path.glob("*.log"))
    if json_files or log_files:
        # The path itself is a run directory
        return [path]
    # Treat as a root directory and discover run sub-dirs
    run_dirs = discover_official_logs(path)
    if not run_dirs:
        print(f"Warning: no run directories found under --output-log path: {path}")
    return run_dirs




def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command or "dashboard"
    prices, trades = load_round1_data(args.root)

    if command == "dashboard":
        official_runs = {}
        selected_run_name = None

        logs_root = args.logs if args.logs else DEFAULT_LOGS_ROOT
        all_run_dirs: list[Path] = []
        if Path(logs_root).exists():
            all_run_dirs.extend(discover_official_logs(logs_root))

        for output_log_path in (args.output_logs or []):
            extra_dirs = _resolve_output_log_dirs(output_log_path)
            for d in extra_dirs:
                if d not in all_run_dirs:
                    all_run_dirs.append(d)

        if all_run_dirs:
            for run_dir in all_run_dirs:
                official_runs[run_dir.name] = build_run_payload(prices, run_dir, run_day=1)

            if args.run:
                matching = [name for name in official_runs if args.run in name]
                selected_run_name = matching[0] if matching else all_run_dirs[-1].name
            else:
                selected_run_name = all_run_dirs[-1].name

            selected = official_runs[selected_run_name]
            print(f"Loaded {len(official_runs)} run folder(s). Selected: {selected_run_name} (as Day 1)")
            print(
                f"  Added {selected['run_prices_count']} price rows | "
                f"Status: {selected['status']}, P&L: {selected['final_profit']:+,.2f}, "
                f"Your trades: {len(selected['your_trades'])}"
            )

        if args.show:
            controller = launch_interactive_dashboard(
                prices,
                trades,
                products=args.products,
                days=args.days,
                official_runs=official_runs,
                selected_run=selected_run_name,
            )
            plt.show()
            plt.close(controller.figure)
            return 0

        figure = create_price_dashboard(prices, trades, products=args.products, days=args.days)
        _write_figure(figure, args.output, args.show)
        return 0

    if command == "snapshot":
        figure = create_order_book_snapshot(prices, args.product, args.day, args.timestamp)
        _write_figure(figure, args.output, args.show)
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

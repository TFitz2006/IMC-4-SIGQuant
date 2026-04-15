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
    load_official_log,
    recalculate_time_columns,
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


def _align_time_to_prices(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    if aligned.empty:
        return aligned

    day_values = sorted(prices["day"].dropna().astype(int).unique().tolist())
    day_order = {day: idx for idx, day in enumerate(day_values)}
    day_span = int(prices.groupby("day")["timestamp"].max().max()) + 100

    aligned["day_sequence"] = aligned["day"].map(day_order).astype(int)
    aligned["global_timestamp"] = aligned["day_sequence"] * day_span + aligned["timestamp"]
    aligned["day_label"] = aligned["day"].map(lambda day: f"Day {day}")
    return aligned


def _final_position_map(log_data: dict) -> dict[str, int]:
    return {
        row["symbol"]: int(row["quantity"])
        for row in log_data.get("final_positions", [])
        if row.get("symbol") != "XIRECS"
    }


def _build_official_run_payload(base_prices: pd.DataFrame, run_dir: Path, run_day: int = 1) -> dict:
    log_data = load_official_log(run_dir, run_day=run_day)

    merged_prices = base_prices.copy()
    run_prices = log_data.get("run_prices")
    if run_prices is not None and not run_prices.empty:
        merged_prices = pd.concat([merged_prices, run_prices], ignore_index=True, sort=False)
        merged_prices = recalculate_time_columns(merged_prices)
        merged_prices = merged_prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    your_trades = log_data.get("your_trades", pd.DataFrame()).copy()
    pnl_series = log_data.get("pnl_series", pd.DataFrame()).copy()
    if not your_trades.empty:
        your_trades = _align_time_to_prices(your_trades, merged_prices)
    if not pnl_series.empty:
        pnl_series = _align_time_to_prices(pnl_series, merged_prices)

    return {
        "name": run_dir.name,
        "prices": merged_prices,
        "your_trades": your_trades,
        "pnl_series": pnl_series,
        "final_profit": log_data.get("final_profit"),
        "final_positions": _final_position_map(log_data),
        "status": log_data.get("status", "UNKNOWN"),
        "run_prices_count": 0 if run_prices is None else len(run_prices),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command or "dashboard"
    prices, trades = load_round1_data(args.root)

    if command == "dashboard":
        official_runs = {}
        selected_run_name = None

        logs_root = args.logs if args.logs else DEFAULT_LOGS_ROOT
        if Path(logs_root).exists():
            run_dirs = discover_official_logs(logs_root)
            if run_dirs:
                for run_dir in run_dirs:
                    official_runs[run_dir.name] = _build_official_run_payload(prices, run_dir, run_day=1)

                if args.run:
                    matching = [name for name in official_runs if args.run in name]
                    selected_run_name = matching[0] if matching else run_dirs[-1].name
                else:
                    selected_run_name = run_dirs[-1].name

                selected = official_runs[selected_run_name]
                print(f"Loaded {len(official_runs)} official run folder(s). Selected: {selected_run_name} (as Day 1)")
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

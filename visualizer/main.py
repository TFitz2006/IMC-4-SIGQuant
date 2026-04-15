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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command or "dashboard"
    prices, trades = load_round1_data(args.root)

    if command == "dashboard":
        # Load official logs if available
        your_trades = None
        pnl_series = None
        final_profit = None
        final_positions = None

        logs_root = args.logs if args.logs else DEFAULT_LOGS_ROOT
        if Path(logs_root).exists():
            run_dirs = discover_official_logs(logs_root)
            if run_dirs:
                # Use specified run or latest
                if args.run:
                    matching = [r for r in run_dirs if args.run in r.name]
                    run_dir = matching[0] if matching else run_dirs[-1]
                else:
                    run_dir = run_dirs[-1]  # Latest run

                # Assign run as Day 1 (after DataCapsules Day 0)
                run_day = 1
                print(f"Loading official logs from: {run_dir.name} (as Day {run_day})")
                log_data = load_official_log(run_dir, run_day=run_day)

                your_trades = log_data["your_trades"]
                pnl_series = log_data["pnl_series"]
                final_profit = log_data["final_profit"]
                final_positions = {
                    row["symbol"]: int(row["quantity"])
                    for row in log_data.get("final_positions", [])
                    if row.get("symbol") != "XIRECS"
                }

                # Merge run prices with DataCapsules prices
                run_prices = log_data.get("run_prices")
                if run_prices is not None and not run_prices.empty:
                    prices = pd.concat([prices, run_prices], ignore_index=True)
                    # Recalculate global timestamps now that all days are present
                    prices = recalculate_time_columns(prices)
                    prices = prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
                    print(f"  Added {len(run_prices)} price rows as Day {run_day}")

                # Recalculate timestamps using the same day mapping as prices
                day_values = sorted(prices["day"].unique().tolist())
                day_order = {day: idx for idx, day in enumerate(day_values)}
                tick_size = 100  # Standard tick size
                day_span = int(prices.groupby("day")["timestamp"].max().max()) + tick_size

                if not your_trades.empty:
                    your_trades["day_sequence"] = your_trades["day"].map(day_order).astype(int)
                    your_trades["global_timestamp"] = your_trades["day_sequence"] * day_span + your_trades["timestamp"]
                    your_trades["day_label"] = your_trades["day"].map(lambda d: f"Day {d}")

                if not pnl_series.empty:
                    pnl_series["day_sequence"] = pnl_series["day"].map(day_order).astype(int)
                    pnl_series["global_timestamp"] = pnl_series["day_sequence"] * day_span + pnl_series["timestamp"]
                    pnl_series["day_label"] = pnl_series["day"].map(lambda d: f"Day {d}")

                print(f"  Status: {log_data['status']}, P&L: {final_profit:+,.2f}, Your trades: {len(your_trades)}")

        if args.show:
            controller = launch_interactive_dashboard(
                prices,
                trades,
                products=args.products,
                days=args.days,
                your_trades=your_trades,
                pnl_series=pnl_series,
                final_profit=final_profit,
                final_positions=final_positions,
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

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imc_matplotlib"))

import matplotlib
if "--show" not in sys.argv[1:]:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualizer.charts import create_price_dashboard, launch_interactive_dashboard
from visualizer.data_loader import DEFAULT_ROUND1_ROOT, load_round1_data
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
        if args.show:
            controller = launch_interactive_dashboard(prices, trades, products=args.products, days=args.days)
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

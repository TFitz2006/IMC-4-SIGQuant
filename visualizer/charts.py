from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.widgets import Button, CheckButtons, RadioButtons

from visualizer.analytics import PRICE_LEVELS, add_order_book_features, aggregate_trade_volume, infer_trade_aggressor
from visualizer.data_loader import build_run_payload


SIDE_COLORS = {
    "buy": "#2a9d8f",
    "sell": "#e76f51",
    "neutral": "#6c757d",
}
QUOTE_COLORS = {
    "bid": "#2b4c7e",
    "ask": "#c44536",
}
# Your algo's trades - bright, distinct colors
YOUR_TRADE_COLORS = {
    "buy": "#00ff88",   # Bright green
    "sell": "#ff3366",  # Bright red/pink
}
PNL_COLOR = "#7c3aed"  # Purple for P&L line
DAY_SHADE = "#f8f9fa"
PRICE_COLOR = "#1f2937"
SPREAD_COLOR = "#6a4c93"
SPREAD_FILL = "#d7c4f0"
QUOTE_FILL = "#a8dadc"
MAX_QUOTE_POINTS_PER_SIDE = 18_000
HOVER_NEIGHBORHOOD = 280

# Abbreviations for long product names
PRODUCT_ABBREV = {
    "ASH_COATED_OSMIUM": "ASH_OSM",
    "INTARIAN_PEPPER_ROOT": "PEPPER",
}


def _abbrev_product(name: str) -> str:
    """Shorten product name for display."""
    return PRODUCT_ABBREV.get(name, name[:10])


def _abbrev_run_name(name: str) -> str:
    """Shorten official run folder names for the left-side selector."""
    return name.replace("Run", "R").replace("(", " ").replace(")", "")


def _ensure_list(values: Sequence[str | int] | None) -> list[str | int] | None:
    if values is None:
        return None
    return list(values)


def _apply_filters(
    frame: pd.DataFrame,
    products: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if products:
        filtered = filtered[filtered["product"].isin(products)]
    if days is not None:
        filtered = filtered[filtered["day"].isin(days)]
    return filtered


def _prepare_dashboard_data(
    prices: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    products: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    selected_products = _ensure_list(products)
    selected_days = _ensure_list(days)

    filtered_prices = _apply_filters(add_order_book_features(prices), selected_products, selected_days)
    if filtered_prices.empty:
        raise ValueError("No price data matched the requested filters.")

    filtered_trades = pd.DataFrame()
    if trades is not None and not trades.empty:
        filtered_trades = _apply_filters(trades, selected_products, selected_days)
        filtered_trades = infer_trade_aggressor(filtered_trades, filtered_prices)

    trade_volume = aggregate_trade_volume(filtered_trades) if not filtered_trades.empty else pd.DataFrame()
    product_names = filtered_prices["product"].dropna().drop_duplicates().tolist()
    return filtered_prices, filtered_trades, trade_volume, product_names


def _day_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("day", as_index=False).agg(
        min_timestamp=("global_timestamp", "min"),
        max_timestamp=("global_timestamp", "max"),
        day_sequence=("day_sequence", "min"),
    )


def _apply_day_ticks(ax, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    boundaries = _day_boundaries(frame)
    tick_positions = ((boundaries["min_timestamp"] + boundaries["max_timestamp"]) / 2.0).tolist()
    tick_labels = [
        f"Day {day} (pre)" if day < 0 else f"Day {day}"
        for day in boundaries["day"].tolist()
    ]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    for start in boundaries["min_timestamp"].tolist()[1:]:
        ax.axvline(start, color="#adb5bd", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)


def _apply_timestamp_ticks(ax, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    boundaries = _day_boundaries(frame).sort_values("day_sequence").reset_index(drop=True)
    if boundaries.empty:
        return

    start_values = boundaries["min_timestamp"].to_numpy(dtype=float)
    end_values = boundaries["max_timestamp"].to_numpy(dtype=float)
    local_starts = (
        frame.groupby("day", as_index=False)
        .agg(local_min_timestamp=("timestamp", "min"))
        .merge(boundaries[["day", "min_timestamp"]], on="day", how="right")
        .sort_values("min_timestamp")
        ["local_min_timestamp"]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    def _formatter(x_value: float, _position: int) -> str:
        index = int(np.searchsorted(start_values, x_value, side="right") - 1)
        index = min(max(index, 0), len(start_values) - 1)
        if x_value > end_values[index] and index < len(start_values) - 1:
            index += 1
        local_timestamp = int(round(x_value - start_values[index] + local_starts[index]))
        return f"{local_timestamp}"

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_formatter))
    ax.tick_params(axis="x", labelsize=8)


def _shade_days(ax, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    for index, row in _day_boundaries(frame).iterrows():
        if index % 2 == 0:
            ax.axvspan(row["min_timestamp"], row["max_timestamp"], color=DAY_SHADE, alpha=0.85, zorder=0)


def _annotate_day_labels(ax, frame: pd.DataFrame) -> None:
    """Overlay 'Day N' / 'Day -1 (pre)' labels at the top of each day section."""
    if frame.empty:
        return

    boundaries = _day_boundaries(frame)
    for _, row in boundaries.iterrows():
        day = int(row["day"])
        x_pos = (row["min_timestamp"] + row["max_timestamp"]) / 2.0
        if day < 0:
            label = f"Day {day} (pre)"
            color = "#9b2226"
            style = "italic"
        else:
            label = f"Day {day}"
            color = "#1f2937"
            style = "normal"
        ax.text(
            x_pos,
            0.985,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            style=style,
            alpha=0.72,
            zorder=5,
        )


def _clean_mid_price(frame: pd.DataFrame) -> pd.Series:
    clean_mid = frame["mid_price"].where(frame["mid_price"] > 0)
    return clean_mid.combine_first(frame["quoted_mid_price"])


def _smooth_mid_price(frame: pd.DataFrame, window: int = 41) -> pd.Series:
    mid = _clean_mid_price(frame)
    if mid.empty:
        return mid

    # Smooth the reference line so it reads as price trend rather than tick noise.
    return mid.groupby(frame["day"], group_keys=False).transform(
        lambda series: series.rolling(window=window, min_periods=1, center=True).median()
    )


def _style_axis(
    ax,
    frame: pd.DataFrame,
    xlabel: str = "Round 1 timeline",
    timestamp_ticks: bool = False,
) -> None:
    _shade_days(ax, frame)
    if timestamp_ticks:
        _apply_timestamp_ticks(ax, frame)
    else:
        _apply_day_ticks(ax, frame)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.22, color="#dbe2ea")
    ax.margins(x=0.01)


def _bucket_trade_volume(product_volume: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if product_volume.empty:
        return product_volume.copy(), 1_000.0

    min_ts = float(product_volume["global_timestamp"].min())
    max_ts = float(product_volume["global_timestamp"].max())
    span = max(max_ts - min_ts, 1.0)
    bucket_size = max(1_000.0, float(int(np.ceil(span / 120.0 / 100.0) * 100)))

    bucketed = product_volume.copy()
    bucketed["bucket_timestamp"] = (
        ((bucketed["global_timestamp"] - min_ts) // bucket_size) * bucket_size
        + min_ts
        + bucket_size / 2.0
    )
    grouped = (
        bucketed.groupby("bucket_timestamp", as_index=False)
        .agg(
            traded_quantity=("traded_quantity", "sum"),
            signed_quantity=("signed_quantity", "sum"),
            trade_count=("trade_count", "sum"),
        )
    )
    return grouped, bucket_size


def _plot_trade_markers(ax, trades: pd.DataFrame) -> None:
    if trades.empty:
        return

    alpha_map = {"buy": 0.85, "sell": 0.85, "neutral": 0.35}
    scale_map = {"buy": 5.0, "sell": 5.0, "neutral": 3.5}
    for side, trade_slice in trades.groupby("aggressor", dropna=False):
        side_key = side if side in SIDE_COLORS else "neutral"
        marker_size = np.clip(trade_slice["quantity"].fillna(0).to_numpy() * scale_map[side_key], 18.0, 180.0)
        ax.scatter(
            trade_slice["global_timestamp"],
            trade_slice["price"],
            s=marker_size,
            alpha=alpha_map[side_key],
            color=SIDE_COLORS[side_key],
            edgecolor="white",
            linewidth=0.4,
            label=f"{side_key.title()} trades",
            zorder=4,
        )


def _draw_summary_panel(
    price_ax,
    spread_ax,
    volume_ax,
    product: str,
    product_prices: pd.DataFrame,
    product_trades: pd.DataFrame,
    product_volume: pd.DataFrame,
) -> None:
    x_values = product_prices["global_timestamp"].to_numpy()
    clean_mid = _clean_mid_price(product_prices)
    smooth_mid = _smooth_mid_price(product_prices)
    valid_quotes = product_prices["best_bid"].notna() & product_prices["best_ask"].notna()

    price_ax.fill_between(
        x_values,
        product_prices["best_bid"].to_numpy(dtype=float),
        product_prices["best_ask"].to_numpy(dtype=float),
        where=valid_quotes.to_numpy(dtype=bool),
        color=QUOTE_FILL,
        alpha=0.26,
        linewidth=0,
        label="Quoted spread",
        zorder=1,
    )
    price_ax.plot(
        x_values,
        smooth_mid.to_numpy(dtype=float),
        color=PRICE_COLOR,
        linewidth=1.7,
        label="Mid trend",
        zorder=3,
    )
    _plot_trade_markers(price_ax, product_trades)
    price_ax.set_title(f"{product} market view", fontsize=13, pad=10)
    price_ax.set_ylabel("Price")

    valid_mid = clean_mid.dropna()
    valid_spreads = product_prices["spread"].dropna()
    summary_lines = [
        f"Last mid: {valid_mid.iloc[-1]:,.1f}" if not valid_mid.empty else "Last mid: n/a",
        f"Median spread: {valid_spreads.median():.1f}" if not valid_spreads.empty else "Median spread: n/a",
        f"Trades: {len(product_trades):,}",
    ]
    price_ax.text(
        0.99,
        0.98,
        "\n".join(summary_lines),
        transform=price_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#343a40",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dee2e6", "alpha": 0.9},
    )

    spread_raw = product_prices["spread"]
    spread_smooth = spread_raw.rolling(window=75, min_periods=5).median()
    spread_ax.plot(x_values, spread_raw.to_numpy(dtype=float), color=SPREAD_COLOR, linewidth=0.7, alpha=0.18, zorder=2)
    spread_ax.plot(x_values, spread_smooth.to_numpy(dtype=float), color=SPREAD_COLOR, linewidth=1.8, zorder=3)
    spread_ax.fill_between(
        x_values,
        0,
        spread_smooth.to_numpy(dtype=float),
        where=spread_smooth.notna().to_numpy(dtype=bool),
        color=SPREAD_FILL,
        alpha=0.28,
        linewidth=0,
        zorder=1,
    )
    spread_ax.set_title("Spread regime")
    spread_ax.set_ylabel("Spread")

    bucketed_volume, bucket_width = _bucket_trade_volume(product_volume)
    if bucketed_volume.empty:
        volume_ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=volume_ax.transAxes)
    else:
        bar_colors = np.where(
            bucketed_volume["signed_quantity"] > 0,
            SIDE_COLORS["buy"],
            np.where(bucketed_volume["signed_quantity"] < 0, SIDE_COLORS["sell"], SIDE_COLORS["neutral"]),
        )
        volume_ax.bar(
            bucketed_volume["bucket_timestamp"],
            bucketed_volume["traded_quantity"],
            color=bar_colors,
            width=bucket_width * 0.86,
            alpha=0.9,
            edgecolor="none",
        )
    volume_ax.set_title("Trade volume")
    volume_ax.set_ylabel("Quantity")

    _style_axis(price_ax, product_prices, xlabel="Timestamp", timestamp_ticks=True)
    _style_axis(spread_ax, product_prices)
    _style_axis(volume_ax, product_prices)

    handles, labels = price_ax.get_legend_handles_labels()
    deduped: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        deduped.setdefault(label, handle)
    price_ax.legend(deduped.values(), deduped.keys(), loc="upper left", fontsize="small", frameon=True)


def create_price_dashboard(
    prices: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    products: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
) -> Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    filtered_prices, filtered_trades, trade_volume, product_names = _prepare_dashboard_data(prices, trades, products, days)

    figure, axes = plt.subplots(
        len(product_names),
        3,
        figsize=(16, max(5.5, len(product_names) * 4.3)),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.5, 1.4, 1.6]},
    )

    for row_index, product in enumerate(product_names):
        product_prices = filtered_prices[filtered_prices["product"] == product].sort_values("global_timestamp")
        product_trades = filtered_trades[filtered_trades["product"] == product].sort_values("global_timestamp")
        product_volume = trade_volume[trade_volume["product"] == product].sort_values("global_timestamp")
        _draw_summary_panel(*axes[row_index], product, product_prices, product_trades, product_volume)

    figure.suptitle("IMC Prosperity 4 Round 1 market dashboard", fontsize=15, y=0.995)
    figure.tight_layout()
    return figure


def _downsample_points(points: pd.DataFrame, limit: int) -> pd.DataFrame:
    if points.empty or len(points) <= limit:
        return points.reset_index(drop=True)
    step = int(np.ceil(len(points) / limit))
    return points.iloc[::step].reset_index(drop=True)


def _build_quote_points(
    product_prices: pd.DataFrame,
    normalize: bool,
    active_levels: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_frames: dict[str, list[pd.DataFrame]] = {"bid": [], "ask": []}

    for side in ("bid", "ask"):
        for level in PRICE_LEVELS:
            if level not in active_levels:
                continue
            price_col = f"{side}_price_{level}"
            volume_col = f"{side}_volume_{level}"
            if price_col not in product_prices.columns or volume_col not in product_prices.columns:
                continue

            points = product_prices[
                ["day", "timestamp", "global_timestamp", "quoted_mid_price", price_col, volume_col]
            ].copy()
            points = points.rename(columns={price_col: "price", volume_col: "volume"})
            points = points[points["price"].notna() & points["volume"].notna()].reset_index(drop=True)
            if points.empty:
                continue

            points["side"] = side
            points["level"] = level
            base_reference = points["quoted_mid_price"].fillna(points["price"])
            points["display_price"] = points["price"] - base_reference if normalize else points["price"]
            quote_frames[side].append(points)

    bids = pd.concat(quote_frames["bid"], ignore_index=True) if quote_frames["bid"] else pd.DataFrame()
    asks = pd.concat(quote_frames["ask"], ignore_index=True) if quote_frames["ask"] else pd.DataFrame()
    bids = _downsample_points(bids.sort_values(["global_timestamp", "level"]), MAX_QUOTE_POINTS_PER_SIDE)
    asks = _downsample_points(asks.sort_values(["global_timestamp", "level"]), MAX_QUOTE_POINTS_PER_SIDE)
    return bids, asks


def _build_trade_points(
    product_trades: pd.DataFrame,
    normalize: bool,
    quantity_range: tuple[float, float],
) -> pd.DataFrame:
    if product_trades.empty:
        return pd.DataFrame()

    min_qty, max_qty = quantity_range
    points = product_trades[
        (product_trades["quantity"] >= min_qty) & (product_trades["quantity"] <= max_qty)
    ].copy()
    if points.empty:
        return points

    base_reference = points["quoted_mid_price"].fillna(points["price"])
    points["display_price"] = points["price"] - base_reference if normalize else points["price"]
    return points.sort_values("global_timestamp").reset_index(drop=True)


def _build_hover_points(
    bid_points: pd.DataFrame,
    ask_points: pd.DataFrame,
    trade_points: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for points, side_label in ((bid_points, "Bid"), (ask_points, "Ask")):
        if points.empty:
            continue
        quote_hover = points[
            ["day", "timestamp", "global_timestamp", "price", "display_price", "volume", "level", "side", "quoted_mid_price"]
        ].copy()
        quote_hover["kind"] = "quote"
        quote_hover["label"] = quote_hover["side"].map(lambda side: f"{side.title()} quote")
        quote_hover["aggressor"] = ""
        quote_hover["quantity"] = quote_hover["volume"]
        quote_hover["touch_relation"] = ""
        frames.append(quote_hover)

    if not trade_points.empty:
        trade_hover = trade_points[
            [
                "day",
                "timestamp",
                "global_timestamp",
                "price",
                "display_price",
                "quantity",
                "aggressor",
                "touch_relation",
                "quoted_mid_price",
            ]
        ].copy()
        trade_hover["kind"] = "trade"
        trade_hover["label"] = "Trade"
        trade_hover["volume"] = trade_hover["quantity"]
        trade_hover["level"] = 0
        trade_hover["side"] = "trade"
        frames.append(trade_hover)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    hover_points = pd.concat(frames, ignore_index=True)
    return hover_points.sort_values("global_timestamp").reset_index(drop=True)


def _set_checkbox_fontsize(widget, size: float) -> None:
    for label in widget.labels:
        label.set_fontsize(size)


def _format_metric_value(metric: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if metric in {"Bought qty", "Sold qty", "Position"}:
        return f"{int(value):+d}" if metric == "Position" else f"{int(value):,}"
    if metric in {"Avg buy", "Avg sell", "Mark"}:
        return f"{float(value):,.1f}"
    if metric in {"Cashflow PnL", "Marked PnL"}:
        return f"{float(value):+,.0f}"
    return str(value)


class ProductToggleDashboard:
    def __init__(
        self,
        prices: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        products: Sequence[str] | None = None,
        days: Sequence[int] | None = None,
        your_trades: pd.DataFrame | None = None,
        pnl_series: pd.DataFrame | None = None,
        final_profit: float | None = None,
        final_positions: dict[str, int] | None = None,
        official_runs: dict[str, dict] | None = None,
        selected_run: str | None = None,
    ) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        self.base_prices = prices.copy()
        self.base_trades = trades.copy() if trades is not None else pd.DataFrame()
        self.products_filter = products
        self.days_filter = days
        self.official_runs = official_runs or {}
        self.run_names = list(self.official_runs.keys())
        self.selected_run = selected_run if selected_run in self.official_runs else (self.run_names[-1] if self.run_names else None)
        self.standalone_official = {
            "your_trades": your_trades if your_trades is not None else pd.DataFrame(),
            "pnl_series": pnl_series if pnl_series is not None else pd.DataFrame(),
            "final_profit": final_profit,
            "final_positions": final_positions or {},
        }
        self._set_data_state()

        self.layer_visibility = {
            "Bids": True,
            "Asks": True,
            "Mkt": True,      # Market trades (shortened to fit)
            "Yours": self.has_official_data,  # Your algo trades
            "Mid": True,
            "Norm": False,    # Normalize (shortened)
        }
        self.level_visibility = {1: True, 2: True, 3: True}
        self.day_visibility = {day: True for day in self.available_days}
        if self.filtered_trades.empty:
            self.trade_quantity_range = (0.0, 1.0)
            self.trade_quantity_bounds = (0.0, 1.0)
        else:
            min_qty = float(self.filtered_trades["quantity"].min())
            max_qty = float(self.filtered_trades["quantity"].max())
            self.trade_quantity_range = (min_qty, max_qty)
            self.trade_quantity_bounds = (min_qty, max_qty)
        self.full_time_bounds = (
            float(self.filtered_prices["global_timestamp"].min()),
            float(self.filtered_prices["global_timestamp"].max()),
        )
        self.time_window = self.full_time_bounds
        self.price_zoom_scale = 1.0
        self._suspend_day_callback = False
        self._drag_start_pixel = None
        self._drag_start_window = None

        self.hover_points = pd.DataFrame()
        self.current_price_frame = pd.DataFrame()
        self.current_mid_points = pd.DataFrame()
        self.current_bid_points = pd.DataFrame()
        self.current_ask_points = pd.DataFrame()
        self.current_trade_points = pd.DataFrame()
        self.span_selector = None

        self.figure = plt.figure(figsize=(16.4, 9.1))
        grid = self.figure.add_gridspec(
            2,
            3,
            left=0.19,
            right=0.985,
            top=0.92,
            bottom=0.10,
            wspace=0.14,
            hspace=0.24,
            height_ratios=[2.3, 1.0],
        )
        self.price_ax = self.figure.add_subplot(grid[0, :])
        self.spread_ax = self.figure.add_subplot(grid[1, 0])
        self.pnl_ax = self.figure.add_subplot(grid[1, 1])
        self.summary_ax = self.figure.add_subplot(grid[1, 2])

        # Create abbreviated product labels for radio buttons
        self.product_labels = [_abbrev_product(p) for p in self.product_names]
        self.label_to_product = {_abbrev_product(p): p for p in self.product_names}

        radio_height = max(0.09, 0.045 * len(self.product_names))
        radio_ax = self.figure.add_axes([0.03, 0.80, 0.13, radio_height], facecolor="#f1f3f5")
        self.radio = RadioButtons(radio_ax, self.product_labels, active=0)
        _set_checkbox_fontsize(self.radio, 9)
        self.radio.on_clicked(self._on_product_label_selected)

        self.run_dropdown_open = False
        self.run_button = None
        self.run_option_buttons: list[Button] = []
        self.run_option_axes = []

        # "+" add-run button — always present
        add_run_ax = self.figure.add_axes([0.143, 0.69, 0.017, 0.035], facecolor="#e9ecef")
        self.add_run_button = Button(add_run_ax, "+")
        self.add_run_button.label.set_fontsize(11)
        self.add_run_button.on_clicked(self._on_add_run_clicked)

        self._rebuild_run_dropdown_controls()

        self.product_pnl = self._calculate_product_pnl()

        layer_ax = self.figure.add_axes([0.03, 0.42, 0.13, 0.18], facecolor="#f1f3f5")
        self.layer_check = CheckButtons(layer_ax, list(self.layer_visibility.keys()), list(self.layer_visibility.values()))
        _set_checkbox_fontsize(self.layer_check, 9.5)
        self.layer_check.on_clicked(self._on_layer_toggled)

        level_ax = self.figure.add_axes([0.03, 0.27, 0.13, 0.12], facecolor="#f1f3f5")
        self.level_check = CheckButtons(level_ax, [f"L{level}" for level in PRICE_LEVELS], [True, True, True])
        _set_checkbox_fontsize(self.level_check, 9.5)
        self.level_check.on_clicked(self._on_level_toggled)

        day_labels = [f"Pre{abs(day)}" if day < 0 else f"D{day}" for day in self.available_days]
        day_initial = [day > 0 for day in self.available_days]
        # Hide negative days by default; if there are no positive days, show everything
        if not any(day_initial):
            day_initial = [True for _ in self.available_days]
        self.day_visibility = {day: checked for day, checked in zip(self.available_days, day_initial)}
        self.time_window = self._selected_day_bounds()
        day_ax = self.figure.add_axes([0.03, 0.13, 0.13, 0.11], facecolor="#f1f3f5")
        self.day_check = CheckButtons(day_ax, day_labels, day_initial)
        _set_checkbox_fontsize(self.day_check, 9.5)
        self.day_check.on_clicked(self._on_day_toggled)

        self.figure.text(0.03, 0.895, "Product", fontsize=10, weight="bold", color="#212529")
        self.figure.text(0.03, 0.735, "Run", fontsize=10, weight="bold", color="#212529")
        self.figure.text(0.03, 0.615, "Layers", fontsize=10, weight="bold", color="#212529")
        self.figure.text(0.03, 0.395, "Depth", fontsize=10, weight="bold", color="#212529")
        self.figure.text(0.03, 0.245, "Days", fontsize=10, weight="bold", color="#212529")

        self.run_pnl_text = self.figure.text(0.03, 0.025, "", fontsize=10, weight="bold")
        self._update_run_pnl_text()

        self.figure.text(0.19, 0.055, "Arrows=product | Drag/scroll=zoom | Shift+scroll=price zoom | r=reset | Norm=center on mid", fontsize=8.5, color="#6b7280")

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.figure.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.figure.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_button_release)

        self.hover_annotation = None
        self.hover_vline = None
        self.hover_hline = None

        self.render(self.current_product)

    def _active_payload(self) -> dict:
        if self.selected_run and self.selected_run in self.official_runs:
            return self.official_runs[self.selected_run]
        return {
            "prices": self.base_prices,
            "your_trades": self.standalone_official["your_trades"],
            "pnl_series": self.standalone_official["pnl_series"],
            "final_profit": self.standalone_official["final_profit"],
            "final_positions": self.standalone_official["final_positions"],
        }

    def _set_data_state(self) -> None:
        payload = self._active_payload()
        active_prices = payload.get("prices", self.base_prices)
        self.filtered_prices, self.filtered_trades, self.trade_volume, self.product_names = _prepare_dashboard_data(
            active_prices,
            self.base_trades,
            self.products_filter,
            self.days_filter,
        )
        previous_product = getattr(self, "current_product", None)
        self.current_product = previous_product if previous_product in self.product_names else self.product_names[0]
        self.available_days = sorted(self.filtered_prices["day"].dropna().astype(int).unique().tolist())

        self.your_trades = payload.get("your_trades", pd.DataFrame())
        self.pnl_series = payload.get("pnl_series", pd.DataFrame())
        self.final_profit = payload.get("final_profit")
        self.final_positions = payload.get("final_positions", {}) or {}
        self.has_official_data = not self.your_trades.empty
        self.product_pnl = self._calculate_product_pnl()

        self.full_time_bounds = (
            float(self.filtered_prices["global_timestamp"].min()),
            float(self.filtered_prices["global_timestamp"].max()),
        )

        old_visibility = getattr(self, "day_visibility", {})
        self.day_visibility = {day: old_visibility.get(day, True) for day in self.available_days}
        if not any(self.day_visibility.values()):
            self.day_visibility = {day: True for day in self.available_days}

    def _calculate_product_pnl(self) -> dict[str, dict[str, float | int]]:
        product_pnl = {}
        if self.your_trades.empty:
            return product_pnl

        for product in self.product_names:
            pt = self.your_trades[self.your_trades["product"] == product]
            if pt.empty:
                product_pnl[product] = {"realized": 0.0, "net_qty": 0}
                continue
            buys = pt[pt["your_side"] == "buy"]
            sells = pt[pt["your_side"] == "sell"]
            buy_cost = (buys["price"] * buys["quantity"]).sum()
            sell_revenue = (sells["price"] * sells["quantity"]).sum()
            net_qty = buys["quantity"].sum() - sells["quantity"].sum()
            product_pnl[product] = {
                "realized": sell_revenue - buy_cost,
                "net_qty": int(self.final_positions.get(product, net_qty)),
            }
        return product_pnl

    def _update_run_pnl_text(self) -> None:
        if not hasattr(self, "run_pnl_text"):
            return
        if not self.has_official_data or self.final_profit is None:
            self.run_pnl_text.set_text("")
            return
        profit_color = "#16a34a" if self.final_profit >= 0 else "#dc2626"
        run_label = _abbrev_run_name(self.selected_run) if self.selected_run else "Run"
        self.run_pnl_text.set_text(f"{run_label} P&L: {self.final_profit:+,.2f}")
        self.run_pnl_text.set_color(profit_color)

    def _build_position_table_rows(self, product: str, product_prices: pd.DataFrame) -> list[list[str]]:
        if self.your_trades.empty:
            return []

        pt = self.your_trades[self.your_trades["product"] == product].copy()
        if pt.empty:
            return []

        buys = pt[pt["your_side"] == "buy"]
        sells = pt[pt["your_side"] == "sell"]

        buy_qty = float(buys["quantity"].sum()) if not buys.empty else 0.0
        sell_qty = float(sells["quantity"].sum()) if not sells.empty else 0.0
        buy_cost = float((buys["price"] * buys["quantity"]).sum()) if not buys.empty else 0.0
        sell_revenue = float((sells["price"] * sells["quantity"]).sum()) if not sells.empty else 0.0

        avg_buy = (buy_cost / buy_qty) if buy_qty > 0 else None
        avg_sell = (sell_revenue / sell_qty) if sell_qty > 0 else None

        trade_net_qty = int(buy_qty - sell_qty)
        net_qty = int(self.final_positions.get(product, trade_net_qty))
        cashflow_pnl = sell_revenue - buy_cost

        latest_row = product_prices.sort_values("global_timestamp").iloc[-1] if not product_prices.empty else None
        if latest_row is not None:
            bid = latest_row.get("bid_price_1")
            ask = latest_row.get("ask_price_1")
            mid = latest_row.get("mid_price")
            if pd.notna(bid) and pd.notna(ask):
                mark_price = float((bid + ask) / 2.0)
            elif pd.notna(mid):
                mark_price = float(mid)
            else:
                mark_price = None
        else:
            mark_price = None

        marked_pnl = cashflow_pnl + net_qty * mark_price if mark_price is not None else None

        metrics = [
            ("Bought qty", buy_qty),
            ("Avg buy", avg_buy),
            ("Sold qty", sell_qty),
            ("Avg sell", avg_sell),
            ("Position", net_qty),
            ("Mark", mark_price),
            ("Cashflow PnL", cashflow_pnl),
            ("Marked PnL", marked_pnl),
        ]
        return [[metric, _format_metric_value(metric, value)] for metric, value in metrics]

    def _rebuild_run_dropdown_controls(self) -> None:
        """Create (or recreate) the run-selector button and dropdown option buttons."""
        # Remove any existing run selector + options
        if self.run_button is not None:
            self.figure.delaxes(self.run_button.ax)
            self.run_button = None
        for ax in self.run_option_axes:
            self.figure.delaxes(ax)
        self.run_option_axes = []
        self.run_option_buttons = []
        self.run_dropdown_open = False

        if not self.run_names:
            return

        # Selector button — narrower to leave room for the "+" button
        run_button_ax = self.figure.add_axes([0.03, 0.69, 0.108, 0.035], facecolor="#f8f9fa")
        self.run_button = Button(run_button_ax, _abbrev_run_name(self.selected_run or self.run_names[-1]))
        self.run_button.label.set_fontsize(8.5)
        self.run_button.on_clicked(self._toggle_run_dropdown)

        for index, run_name in enumerate(self.run_names):
            option_ax = self.figure.add_axes([0.03, 0.65 - index * 0.036, 0.13, 0.034], facecolor="#ffffff")
            option_button = Button(option_ax, _abbrev_run_name(run_name))
            option_button.label.set_fontsize(8.2)
            option_button.on_clicked(lambda _event, selected=run_name: self._select_run_name(selected))
            option_ax.set_visible(False)
            self.run_option_axes.append(option_ax)
            self.run_option_buttons.append(option_button)

    def _on_add_run_clicked(self, _event) -> None:
        """Open the native macOS folder picker in a background thread so the window stays live."""
        import subprocess
        import threading

        def _pick() -> None:
            try:
                result = subprocess.run(
                    [
                        "osascript", "-e",
                        'tell app "Finder" to POSIX path of '
                        '(choose folder with prompt "Select run directory")',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    return
                chosen = result.stdout.strip().rstrip("/")
                if not chosen:
                    return
                # Schedule _add_run back on the main Tk thread so matplotlib is safe
                try:
                    self.figure.canvas.get_tk_widget().after(0, lambda: self._add_run(chosen))
                except Exception:
                    self._add_run(chosen)
            except Exception as exc:
                print(f"Directory picker failed: {exc}")

        threading.Thread(target=_pick, daemon=True).start()

    def _add_run(self, run_dir: str) -> None:
        """Load a run directory, copy it into OfficialLogs/ if needed, and add to dropdown."""
        import shutil
        from pathlib import Path
        from visualizer.data_loader import DEFAULT_LOGS_ROOT

        run_path = Path(run_dir).expanduser().resolve()
        if not run_path.exists():
            print(f"Run directory not found: {run_path}")
            return

        run_name = run_path.name

        # Copy into OfficialLogs/ if it lives elsewhere
        logs_root = Path(DEFAULT_LOGS_ROOT).resolve()
        dest_path = logs_root / run_name
        if run_path != dest_path:
            if not dest_path.exists():
                logs_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(run_path, dest_path)
                print(f"Copied '{run_name}' → {dest_path}")
            run_path = dest_path

        if run_name in self.official_runs:
            self._select_run_name(run_name)
            return

        try:
            payload = build_run_payload(self.base_prices, run_path, run_day=1)
        except Exception as exc:
            print(f"Failed to load run '{run_name}': {exc}")
            return

        self.official_runs[run_name] = payload
        self.run_names = list(self.official_runs.keys())
        self.selected_run = run_name

        self._rebuild_run_dropdown_controls()
        self.figure.canvas.draw_idle()

        self._set_run_dropdown_visible(False)
        self._set_data_state()
        self.layer_visibility["Yours"] = self.has_official_data
        self._reset_time_window_to_selected_days()
        self.price_zoom_scale = 1.0
        self._update_run_pnl_text()
        self._hide_hover()
        self.render(self.current_product)

        print(f"Loaded run '{run_name}' | Status: {payload['status']}, "
              f"P&L: {payload['final_profit']:+,.2f}, "
              f"Your trades: {len(payload['your_trades'])}")

    def _on_product_label_selected(self, label: str) -> None:
        """Handle radio button click with abbreviated label."""
        product = self.label_to_product.get(label, label)
        self._on_product_selected(product)

    def _set_run_dropdown_visible(self, visible: bool) -> None:
        self.run_dropdown_open = visible
        for axis in self.run_option_axes:
            axis.set_visible(visible)
        self.figure.canvas.draw_idle()

    def _toggle_run_dropdown(self, _event) -> None:
        self._set_run_dropdown_visible(not self.run_dropdown_open)

    def _select_run_name(self, run_name: str) -> None:
        self._set_run_dropdown_visible(False)
        if run_name == self.selected_run:
            return

        self.selected_run = run_name
        if self.run_button is not None:
            self.run_button.label.set_text(_abbrev_run_name(run_name))
        previous_yours_visibility = self.layer_visibility.get("Yours", True)
        self._set_data_state()
        self.layer_visibility["Yours"] = previous_yours_visibility and self.has_official_data
        self._reset_time_window_to_selected_days()
        self.price_zoom_scale = 1.0
        self._update_run_pnl_text()
        self._hide_hover()
        self.render(self.current_product)

    def _on_product_selected(self, product: str) -> None:
        self.price_zoom_scale = 1.0
        self.render(product)

    def _on_layer_toggled(self, label: str) -> None:
        self.layer_visibility[label] = not self.layer_visibility[label]
        self.render(self.current_product)

    def _on_level_toggled(self, label: str) -> None:
        level = int(label[1:])
        self.level_visibility[level] = not self.level_visibility[level]
        self.render(self.current_product)

    def _on_trade_filter_changed(self, values) -> None:
        self.trade_quantity_range = (float(values[0]), float(values[1]))
        self.render(self.current_product)

    def _on_day_toggled(self, label: str) -> None:
        if self._suspend_day_callback:
            return

        if label.startswith("Pre"):
            day = -int(label[3:])
        else:
            day = int(label[1:])
        self.day_visibility[day] = not self.day_visibility[day]

        if not any(self.day_visibility.values()):
            self._suspend_day_callback = True
            self.day_visibility[day] = True
            day_index = self.available_days.index(day)
            self.day_check.set_active(day_index)
            self._suspend_day_callback = False
            return

        self._reset_time_window_to_selected_days()
        self.price_zoom_scale = 1.0
        self.render(self.current_product)

    def _on_key_press(self, event) -> None:
        if event.key in {"left", "right"}:
            current_index = self.product_names.index(self.current_product)
            if event.key == "right":
                next_index = (current_index + 1) % len(self.product_names)
            else:
                next_index = (current_index - 1) % len(self.product_names)
            self.radio.set_active(next_index)
            return

        if event.key == "r":
            self._reset_time_window_to_selected_days()
            self.price_zoom_scale = 1.0
            self._apply_time_window()
            return

    def _hide_hover(self) -> None:
        if self.hover_annotation is not None:
            self.hover_annotation.set_visible(False)
        if self.hover_vline is not None:
            self.hover_vline.set_visible(False)
        if self.hover_hline is not None:
            self.hover_hline.set_visible(False)
        self.figure.canvas.draw_idle()

    def _on_axes_leave(self, event) -> None:
        if event.inaxes == self.price_ax:
            self._hide_hover()

    def _on_figure_leave(self, event) -> None:
        self._hide_hover()

    def _selected_days(self) -> list[int]:
        return [day for day, visible in self.day_visibility.items() if visible]

    def _selected_day_bounds(self) -> tuple[float, float]:
        selected_days = self._selected_days()
        selected_prices = self.filtered_prices[self.filtered_prices["day"].isin(selected_days)]
        if selected_prices.empty:
            return self.full_time_bounds
        return (
            float(selected_prices["global_timestamp"].min()),
            float(selected_prices["global_timestamp"].max()),
        )

    def _reset_time_window_to_selected_days(self) -> None:
        self.time_window = self._selected_day_bounds()

    def _autoscale_price_axis(self) -> None:
        lower, upper = self.time_window
        visible_series: list[pd.Series] = []

        if self.layer_visibility["Mid"] and not self.current_mid_points.empty:
            visible_series.append(
                self.current_mid_points[
                    (self.current_mid_points["global_timestamp"] >= lower) & (self.current_mid_points["global_timestamp"] <= upper)
                ]["display_price"]
            )
        if self.layer_visibility["Bids"] and not self.current_bid_points.empty:
            visible_series.append(
                self.current_bid_points[
                    (self.current_bid_points["global_timestamp"] >= lower) & (self.current_bid_points["global_timestamp"] <= upper)
                ]["display_price"]
            )
        if self.layer_visibility["Asks"] and not self.current_ask_points.empty:
            visible_series.append(
                self.current_ask_points[
                    (self.current_ask_points["global_timestamp"] >= lower) & (self.current_ask_points["global_timestamp"] <= upper)
                ]["display_price"]
            )
        if self.layer_visibility["Mkt"] and not self.current_trade_points.empty:
            visible_series.append(
                self.current_trade_points[
                    (self.current_trade_points["global_timestamp"] >= lower) & (self.current_trade_points["global_timestamp"] <= upper)
                ]["display_price"]
            )

        visible_series = [series.dropna() for series in visible_series if not series.empty]
        if not visible_series:
            return

        combined = pd.concat(visible_series, ignore_index=True)
        if combined.empty:
            return

        if self.layer_visibility["Norm"]:
            center = 0.0
            base_half_width = max(8.0, float(combined.abs().quantile(0.998)) + 2.5)
        else:
            center = float(combined.mean())
            q_low = float(combined.quantile(0.002))
            q_high = float(combined.quantile(0.998))
            base_half_width = max(center - q_low, q_high - center, 10.0)

        half_width = base_half_width * self.price_zoom_scale
        self.price_ax.set_ylim(center - half_width * 1.08, center + half_width * 1.08)

    def _apply_time_window(self) -> None:
        day_lower, day_upper = self._selected_day_bounds()
        lower = min(max(self.time_window[0], day_lower), day_upper)
        upper = max(min(self.time_window[1], day_upper), day_lower)
        if upper <= lower:
            lower, upper = day_lower, day_upper
        self.time_window = (lower, upper)

        for axis in (self.price_ax, self.spread_ax, self.pnl_ax):
            axis.set_xlim(lower, upper)
        self._autoscale_price_axis()
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes not in {self.price_ax, self.spread_ax, self.pnl_ax} or event.xdata is None:
            return

        if event.inaxes == self.price_ax and event.key == "shift":
            if event.button == "up":
                self.price_zoom_scale *= 0.82
            else:
                self.price_zoom_scale *= 1.22
            self.price_zoom_scale = min(max(self.price_zoom_scale, 0.2), 8.0)
            self._autoscale_price_axis()
            self.figure.canvas.draw_idle()
            return

        lower, upper = self.time_window
        current_span = max(upper - lower, 1.0)
        scale = 0.75 if event.button == "up" else 1.35
        new_span = current_span * scale
        cursor_ratio = (event.xdata - lower) / current_span if current_span else 0.5
        new_lower = event.xdata - new_span * cursor_ratio
        new_upper = event.xdata + new_span * (1.0 - cursor_ratio)
        self.time_window = (new_lower, new_upper)
        self._apply_time_window()

    def _on_button_press(self, event) -> None:
        if event.inaxes not in {self.price_ax, self.spread_ax, self.pnl_ax}:
            return
        if event.dblclick:
            self._reset_time_window_to_selected_days()
            self.price_zoom_scale = 1.0
            self._apply_time_window()
            return
        if event.button == 1 and event.x is not None:
            self._drag_start_pixel = event.x
            self._drag_start_window = self.time_window

    def _on_button_release(self, event) -> None:
        self._drag_start_pixel = None
        self._drag_start_window = None

    def _on_mouse_move(self, event) -> None:
        # Pan: left-button drag in any chart axis
        if self._drag_start_pixel is not None and event.x is not None:
            bbox = self.price_ax.get_window_extent()
            ax_width = max(bbox.width, 1.0)
            orig_lower, orig_upper = self._drag_start_window
            data_delta = (self._drag_start_pixel - event.x) / ax_width * (orig_upper - orig_lower)
            self.time_window = (orig_lower + data_delta, orig_upper + data_delta)
            self._apply_time_window()
            return

        if event.inaxes != self.price_ax or event.xdata is None or event.ydata is None or self.hover_points.empty:
            return

        times = self.hover_points["global_timestamp"].to_numpy(dtype=float)
        insert_index = int(np.searchsorted(times, event.xdata))
        left = max(0, insert_index - HOVER_NEIGHBORHOOD)
        right = min(len(times), insert_index + HOVER_NEIGHBORHOOD)
        candidates = self.hover_points.iloc[left:right].copy()
        if candidates.empty:
            self._hide_hover()
            return

        x_span = max(self.price_ax.get_xlim()[1] - self.price_ax.get_xlim()[0], 1.0)
        y_span = max(self.price_ax.get_ylim()[1] - self.price_ax.get_ylim()[0], 1.0)
        distance = (
            ((candidates["global_timestamp"] - event.xdata) / x_span) ** 2
            + ((candidates["display_price"] - event.ydata) / y_span) ** 2 * 1.6
        )
        closest_index = int(distance.idxmin())
        closest_distance = float(distance.loc[closest_index])
        if closest_distance > 0.0018:
            self._hide_hover()
            return

        point = self.hover_points.loc[closest_index]
        if point["kind"] == "quote":
            text = (
                f"{point['label']} | L{int(point['level'])}\n"
                f"Day {int(point['day'])}  ts {int(point['timestamp'])}\n"
                f"Price {float(point['price']):,.1f}  Vol {int(point['volume'])}"
            )
        else:
            aggressor = point["aggressor"] if pd.notna(point["aggressor"]) else "neutral"
            touch_relation = point["touch_relation"] if pd.notna(point["touch_relation"]) else "unknown"
            text = (
                f"Trade | {str(aggressor).title()}\n"
                f"Day {int(point['day'])}  ts {int(point['timestamp'])}\n"
                f"Price {float(point['price']):,.1f}  Qty {int(point['quantity'])}\n"
                f"Touch {touch_relation}"
            )

        if pd.notna(point.get("quoted_mid_price")):
            text += f"\nQuoted mid {float(point['quoted_mid_price']):,.1f}"

        self.hover_annotation.xy = (float(point["global_timestamp"]), float(point["display_price"]))
        self.hover_annotation.set_text(text)
        self.hover_annotation.set_visible(True)
        self.hover_vline.set_data([float(point["global_timestamp"]), float(point["global_timestamp"])], [0.0, 1.0])
        self.hover_hline.set_data([0.0, 1.0], [float(point["display_price"]), float(point["display_price"])])
        self.hover_vline.set_visible(True)
        self.hover_hline.set_visible(True)
        self.figure.canvas.draw_idle()

    def _draw_orderbook_explorer(
        self,
        product: str,
        product_prices: pd.DataFrame,
        product_trades: pd.DataFrame,
        product_volume: pd.DataFrame,
    ) -> None:
        normalize = self.layer_visibility["Norm"]
        active_levels = {level for level, enabled in self.level_visibility.items() if enabled}
        if not active_levels:
            active_levels = {1}

        bid_points, ask_points = _build_quote_points(product_prices, normalize=normalize, active_levels=active_levels)
        trade_points = _build_trade_points(product_trades, normalize=normalize, quantity_range=self.trade_quantity_range)
        self.hover_points = _build_hover_points(bid_points, ask_points, trade_points)
        self.current_price_frame = product_prices

        _shade_days(self.price_ax, product_prices)
        _annotate_day_labels(self.price_ax, product_prices)
        x_values = product_prices["global_timestamp"].to_numpy()
        clean_mid = _clean_mid_price(product_prices)
        smooth_mid = _smooth_mid_price(product_prices)
        mid_reference = product_prices["quoted_mid_price"].fillna(clean_mid)
        if normalize:
            display_mid = pd.Series(np.zeros(len(product_prices)), index=product_prices.index, dtype=float)
        else:
            display_mid = smooth_mid
        self.current_mid_points = pd.DataFrame({"global_timestamp": x_values, "display_price": display_mid})
        self.current_bid_points = bid_points.copy()
        self.current_ask_points = ask_points.copy()
        self.current_trade_points = trade_points.copy()

        legend_handles: list[Line2D] = []

        if self.layer_visibility["Mid"]:
            if normalize:
                self.price_ax.axhline(0.0, color=PRICE_COLOR, linewidth=1.1, alpha=0.85, zorder=3)
                legend_handles.append(Line2D([0], [0], color=PRICE_COLOR, lw=1.5, label="Quoted mid baseline"))
            else:
                self.price_ax.plot(
                    x_values,
                    display_mid.to_numpy(dtype=float),
                    color=PRICE_COLOR,
                    linewidth=1.4,
                    alpha=0.95,
                    zorder=3,
                )
                legend_handles.append(Line2D([0], [0], color=PRICE_COLOR, lw=1.5, label="Mid trend"))

        if self.layer_visibility["Bids"] and not bid_points.empty:
            bid_sizes = np.clip(np.sqrt(bid_points["volume"].to_numpy(dtype=float)) * 7.5, 10.0, 70.0)
            self.price_ax.scatter(
                bid_points["global_timestamp"],
                bid_points["display_price"],
                s=bid_sizes,
                color=QUOTE_COLORS["bid"],
                alpha=0.28,
                marker="s",
                linewidths=0,
                zorder=2,
            )
            legend_handles.append(
                Line2D([0], [0], marker="s", linestyle="", color=QUOTE_COLORS["bid"], markersize=6, label="Bid quotes")
            )

        if self.layer_visibility["Asks"] and not ask_points.empty:
            ask_sizes = np.clip(np.sqrt(ask_points["volume"].to_numpy(dtype=float)) * 7.5, 10.0, 70.0)
            self.price_ax.scatter(
                ask_points["global_timestamp"],
                ask_points["display_price"],
                s=ask_sizes,
                color=QUOTE_COLORS["ask"],
                alpha=0.26,
                marker="s",
                linewidths=0,
                zorder=2,
            )
            legend_handles.append(
                Line2D([0], [0], marker="s", linestyle="", color=QUOTE_COLORS["ask"], markersize=6, label="Ask quotes")
            )

        if self.layer_visibility["Mkt"] and not trade_points.empty:
            trade_markers = {"buy": "^", "sell": "v", "neutral": "o"}
            for side in ("buy", "sell", "neutral"):
                side_points = trade_points[trade_points["aggressor"] == side]
                if side_points.empty:
                    continue
                sizes = np.clip(side_points["quantity"].to_numpy(dtype=float) * (6.0 if side != "neutral" else 4.0), 18.0, 180.0)
                self.price_ax.scatter(
                    side_points["global_timestamp"],
                    side_points["display_price"],
                    s=sizes,
                    color=SIDE_COLORS[side],
                    alpha=0.8 if side != "neutral" else 0.35,
                    marker=trade_markers[side],
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=4,
                )
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=trade_markers[side],
                        linestyle="",
                        markerfacecolor=SIDE_COLORS[side],
                        markeredgecolor="white",
                        markeredgewidth=0.35,
                        markersize=7,
                        label=f"Mkt {side}",
                    )
                )

        # Your algorithm's trades (from official logs) - bright distinct markers
        if self.layer_visibility["Yours"] and not self.your_trades.empty:
            product_your_trades = self.your_trades[self.your_trades["product"] == product].copy()
            if not product_your_trades.empty:
                # Apply normalization if needed
                if normalize:
                    product_your_trades = product_your_trades.merge(
                        product_prices[["global_timestamp", "quoted_mid_price"]].drop_duplicates("global_timestamp"),
                        on="global_timestamp",
                        how="left",
                    )
                    product_your_trades["display_price"] = product_your_trades["price"] - product_your_trades["quoted_mid_price"].fillna(product_your_trades["price"])
                else:
                    product_your_trades["display_price"] = product_your_trades["price"]

                your_markers = {"buy": "^", "sell": "v"}
                for side in ("buy", "sell"):
                    side_trades = product_your_trades[product_your_trades["your_side"] == side]
                    if side_trades.empty:
                        continue
                    sizes = np.clip(side_trades["quantity"].to_numpy(dtype=float) * 12.0, 60.0, 300.0)
                    self.price_ax.scatter(
                        side_trades["global_timestamp"],
                        side_trades["display_price"],
                        s=sizes,
                        color=YOUR_TRADE_COLORS[side],
                        alpha=0.95,
                        marker=your_markers[side],
                        edgecolors="#1f2937",
                        linewidths=1.2,
                        zorder=6,  # On top of everything
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker=your_markers[side],
                            linestyle="",
                            markerfacecolor=YOUR_TRADE_COLORS[side],
                            markeredgecolor="#1f2937",
                            markeredgewidth=1.0,
                            markersize=9,
                            label=f"You {side}",
                        )
                    )

        title_abbrev = _abbrev_product(product)
        norm_suffix = " (normalized)" if normalize else ""
        self.price_ax.set_title(f"{title_abbrev} order book{norm_suffix}", fontsize=12, pad=8)
        self.price_ax.set_ylabel("Price - quoted mid" if normalize else "Price")
        _apply_timestamp_ticks(self.price_ax, product_prices)
        self.price_ax.set_xlabel("Timestamp")
        self.price_ax.grid(alpha=0.18, color="#dbe2ea")
        self.price_ax.margins(x=0.01)

        valid_mid = clean_mid.dropna()
        valid_spread = product_prices["spread"].dropna()
        visible_volume = trade_points["quantity"].sum() if not trade_points.empty else 0

        # Count your trades for this product
        your_product_trades = self.your_trades[self.your_trades["product"] == product] if not self.your_trades.empty else pd.DataFrame()
        your_buys = your_product_trades[your_product_trades["your_side"] == "buy"]["quantity"].sum() if not your_product_trades.empty else 0
        your_sells = your_product_trades[your_product_trades["your_side"] == "sell"]["quantity"].sum() if not your_product_trades.empty else 0

        summary_lines = [
            f"Mid: {valid_mid.iloc[-1]:,.1f}" if not valid_mid.empty else "Mid: n/a",
            f"Spread: {valid_spread.median():.1f}" if not valid_spread.empty else "Spread: n/a",
            f"Mkt trades: {len(trade_points):,}",
        ]
        if not your_product_trades.empty:
            summary_lines.append(f"You: +{int(your_buys)} / -{int(your_sells)}")
        summary_lines.append(f"Levels: {','.join(f'L{l}' for l in sorted(active_levels))}")
        self.price_ax.text(
            0.99,
            0.98,
            "\n".join(summary_lines),
            transform=self.price_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.3,
            color="#343a40",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#dee2e6", "alpha": 0.92},
        )

        if legend_handles:
            unique_handles: dict[str, Line2D] = {}
            for handle in legend_handles:
                unique_handles.setdefault(handle.get_label(), handle)
            self.price_ax.legend(unique_handles.values(), unique_handles.keys(), loc="upper left", fontsize="small", frameon=True)

        spread_raw = product_prices["spread"]
        spread_smooth = spread_raw.rolling(window=75, min_periods=5).median()
        self.spread_ax.plot(x_values, spread_raw.to_numpy(dtype=float), color=SPREAD_COLOR, linewidth=0.7, alpha=0.14, zorder=2)
        self.spread_ax.plot(x_values, spread_smooth.to_numpy(dtype=float), color=SPREAD_COLOR, linewidth=1.7, zorder=3)
        self.spread_ax.fill_between(
            x_values,
            0,
            spread_smooth.to_numpy(dtype=float),
            where=spread_smooth.notna().to_numpy(dtype=bool),
            color=SPREAD_FILL,
            alpha=0.32,
            linewidth=0,
            zorder=1,
        )
        self.spread_ax.set_title("Spread regime")
        self.spread_ax.set_ylabel("Spread")
        _style_axis(self.spread_ax, product_prices)

        self.summary_ax.clear()

        # P&L panel (if official data available) or Trade volume
        if self.has_official_data and not self.pnl_series.empty:
            selected_days = self._selected_days()
            pnl_data = self.pnl_series[self.pnl_series["day"].isin(selected_days)].copy()
            if not pnl_data.empty:
                pnl_x = pnl_data["global_timestamp"].to_numpy()
                pnl_y = pnl_data["pnl"].to_numpy()
                self.pnl_ax.fill_between(
                    pnl_x, 0, pnl_y,
                    where=pnl_y >= 0,
                    color="#16a34a",
                    alpha=0.3,
                    linewidth=0,
                )
                self.pnl_ax.fill_between(
                    pnl_x, 0, pnl_y,
                    where=pnl_y < 0,
                    color="#dc2626",
                    alpha=0.3,
                    linewidth=0,
                )
                self.pnl_ax.plot(pnl_x, pnl_y, color=PNL_COLOR, linewidth=1.8, zorder=3)
                self.pnl_ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.5)
                self.pnl_ax.set_title("P&L (Your Run)")
                self.pnl_ax.set_ylabel("Profit")
                _style_axis(self.pnl_ax, product_prices)
            else:
                self.pnl_ax.text(0.5, 0.5, "P&L: Select Day 1", ha="center", va="center", transform=self.pnl_ax.transAxes, color="#6b7280")
                self.pnl_ax.set_title("P&L (Your Run)")
                _style_axis(self.pnl_ax, product_prices)

            self.summary_ax.axis("off")
            self.summary_ax.set_title("Position & PnL Summary")

            product_pnl_info = self.product_pnl.get(product, {})
            realized = product_pnl_info.get("realized", 0)
            net_qty = product_pnl_info.get("net_qty", 0)
            color = "#16a34a" if realized >= 0 else "#dc2626"
            abbrev = _abbrev_product(product)

            if self.final_profit is not None:
                total_color = "#16a34a" if self.final_profit >= 0 else "#dc2626"
                self.summary_ax.text(
                    0.02,
                    0.97,
                    f"Total: {self.final_profit:+,.0f}",
                    transform=self.summary_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    weight="bold",
                    color=total_color,
                )

            self.summary_ax.text(
                0.98,
                0.97,
                f"{abbrev}: {realized:+,.0f}\nPos: {net_qty:+d}",
                transform=self.summary_ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                weight="bold",
                color=color,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dee2e6", "alpha": 0.9},
            )

            table_rows = self._build_position_table_rows(product, product_prices)
            if table_rows:
                table = self.summary_ax.table(
                    cellText=table_rows,
                    colLabels=["Metric", "Value"],
                    cellLoc="left",
                    colLoc="left",
                    bbox=[0.05, 0.08, 0.90, 0.70],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.6)
                table.scale(1.0, 1.18)
                for (row, col), cell in table.get_celld().items():
                    cell.set_edgecolor("#dee2e6")
                    cell.set_linewidth(0.6)
                    if row == 0:
                        cell.set_facecolor("#f1f5f9")
                        cell.set_text_props(weight="bold", color="#1f2937")
                    else:
                        cell.set_facecolor("white")
                        if col == 1:
                            metric_name = table_rows[row - 1][0]
                            raw_value = table_rows[row - 1][1]
                            if metric_name in {"Cashflow PnL", "Marked PnL", "Position"}:
                                if raw_value.startswith("+"):
                                    cell.get_text().set_color("#16a34a")
                                elif raw_value.startswith("-"):
                                    cell.get_text().set_color("#dc2626")
            else:
                self.summary_ax.text(
                    0.5,
                    0.45,
                    "No official trade summary for this product.",
                    ha="center",
                    va="center",
                    transform=self.summary_ax.transAxes,
                    color="#6b7280",
                )
        else:
            self.summary_ax.axis("off")
            self.summary_ax.set_title("Summary")

            # Fallback to trade volume
            visible_trade_volume = aggregate_trade_volume(trade_points) if not trade_points.empty else pd.DataFrame()
            bucketed_volume, bucket_width = _bucket_trade_volume(visible_trade_volume)
            if bucketed_volume.empty:
                self.pnl_ax.text(0.5, 0.5, "No trades in filter", ha="center", va="center", transform=self.pnl_ax.transAxes)
            else:
                bar_colors = np.where(
                    bucketed_volume["signed_quantity"] > 0,
                    SIDE_COLORS["buy"],
                    np.where(bucketed_volume["signed_quantity"] < 0, SIDE_COLORS["sell"], SIDE_COLORS["neutral"]),
                )
                self.pnl_ax.bar(
                    bucketed_volume["bucket_timestamp"],
                    bucketed_volume["traded_quantity"],
                    color=bar_colors,
                    width=bucket_width * 0.86,
                    alpha=0.9,
                    edgecolor="none",
                )
            self.pnl_ax.set_title("Trade volume")
            self.pnl_ax.set_ylabel("Quantity")
            _style_axis(self.pnl_ax, product_prices)
            self.summary_ax.text(
                0.5,
                0.5,
                "Load official logs to see position and P&L summary.",
                ha="center",
                va="center",
                transform=self.summary_ax.transAxes,
                color="#6b7280",
            )

        self.hover_vline = Line2D([], [], color="#495057", linestyle="--", linewidth=0.8, alpha=0.7, visible=False, zorder=5)
        self.hover_vline.set_transform(self.price_ax.get_xaxis_transform())
        self.price_ax.add_line(self.hover_vline)
        self.hover_hline = Line2D([], [], color="#495057", linestyle="--", linewidth=0.8, alpha=0.7, visible=False, zorder=5)
        self.hover_hline.set_transform(self.price_ax.get_yaxis_transform())
        self.price_ax.add_line(self.hover_hline)
        self.hover_annotation = self.price_ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#adb5bd", "alpha": 0.95},
            fontsize=9,
            color="#212529",
            visible=False,
            zorder=6,
        )
        self.span_selector = None

    def render(self, product: str) -> None:
        self.current_product = product
        self.price_ax.clear()
        self.spread_ax.clear()
        self.pnl_ax.clear()
        self.summary_ax.clear()

        selected_days = self._selected_days()
        product_prices = self.filtered_prices[
            (self.filtered_prices["product"] == product) & (self.filtered_prices["day"].isin(selected_days))
        ].sort_values("global_timestamp")
        product_trades = self.filtered_trades[
            (self.filtered_trades["product"] == product) & (self.filtered_trades["day"].isin(selected_days))
        ].sort_values("global_timestamp")
        product_volume = self.trade_volume[
            (self.trade_volume["product"] == product) & (self.trade_volume["day"].isin(selected_days))
        ].sort_values("global_timestamp")

        self._draw_orderbook_explorer(product, product_prices, product_trades, product_volume)
        abbrev = _abbrev_product(product)
        run_suffix = f" | {_abbrev_run_name(self.selected_run)}" if self.selected_run else ""
        self.figure.suptitle(f"IMC Prosperity 4 | {abbrev}{run_suffix}", fontsize=15, y=0.975)
        self._apply_time_window()
        self.figure.canvas.draw_idle()


def launch_interactive_dashboard(
    prices: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    products: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
    your_trades: pd.DataFrame | None = None,
    pnl_series: pd.DataFrame | None = None,
    final_profit: float | None = None,
    final_positions: dict[str, int] | None = None,
    official_runs: dict[str, dict] | None = None,
    selected_run: str | None = None,
) -> ProductToggleDashboard:
    return ProductToggleDashboard(
        prices,
        trades,
        products=products,
        days=days,
        your_trades=your_trades,
        pnl_series=pnl_series,
        final_profit=final_profit,
        final_positions=final_positions,
        official_runs=official_runs,
        selected_run=selected_run,
    )

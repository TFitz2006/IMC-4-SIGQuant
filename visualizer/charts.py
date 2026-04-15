from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, RadioButtons, RangeSlider, SpanSelector

from visualizer.analytics import PRICE_LEVELS, add_order_book_features, aggregate_trade_volume, infer_trade_aggressor


SIDE_COLORS = {
    "buy": "#2a9d8f",
    "sell": "#e76f51",
    "neutral": "#6c757d",
}
QUOTE_COLORS = {
    "bid": "#2b4c7e",
    "ask": "#c44536",
}
DAY_SHADE = "#f8f9fa"
PRICE_COLOR = "#1f2937"
SPREAD_COLOR = "#6a4c93"
SPREAD_FILL = "#d7c4f0"
QUOTE_FILL = "#a8dadc"
MAX_QUOTE_POINTS_PER_SIDE = 18_000
HOVER_NEIGHBORHOOD = 280


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
    )


def _apply_day_ticks(ax, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    boundaries = _day_boundaries(frame)
    tick_positions = ((boundaries["min_timestamp"] + boundaries["max_timestamp"]) / 2.0).tolist()
    tick_labels = [f"Day {day}" for day in boundaries["day"].tolist()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    for start in boundaries["min_timestamp"].tolist()[1:]:
        ax.axvline(start, color="#adb5bd", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)


def _shade_days(ax, frame: pd.DataFrame) -> None:
    if frame.empty:
        return

    for index, row in _day_boundaries(frame).iterrows():
        if index % 2 == 0:
            ax.axvspan(row["min_timestamp"], row["max_timestamp"], color=DAY_SHADE, alpha=0.85, zorder=0)


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


def _style_axis(ax, frame: pd.DataFrame, xlabel: str = "Round 1 timeline") -> None:
    _shade_days(ax, frame)
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

    for axis in (price_ax, spread_ax, volume_ax):
        _style_axis(axis, product_prices)

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


class ProductToggleDashboard:
    def __init__(
        self,
        prices: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        products: Sequence[str] | None = None,
        days: Sequence[int] | None = None,
    ) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        self.filtered_prices, self.filtered_trades, self.trade_volume, self.product_names = _prepare_dashboard_data(
            prices,
            trades,
            products,
            days,
        )
        self.current_product = self.product_names[0]
        self.available_days = sorted(self.filtered_prices["day"].dropna().astype(int).unique().tolist())

        self.layer_visibility = {
            "Bids": True,
            "Asks": True,
            "Trades": True,
            "Mid": True,
            "Normalize": False,
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
            2,
            left=0.19,
            right=0.985,
            top=0.92,
            bottom=0.10,
            wspace=0.16,
            hspace=0.24,
            height_ratios=[2.3, 1.0],
        )
        self.price_ax = self.figure.add_subplot(grid[0, :])
        self.spread_ax = self.figure.add_subplot(grid[1, 0])
        self.volume_ax = self.figure.add_subplot(grid[1, 1])

        radio_height = max(0.12, 0.055 * len(self.product_names))
        radio_ax = self.figure.add_axes([0.03, 0.74, 0.13, radio_height], facecolor="#f1f3f5")
        self.radio = RadioButtons(radio_ax, self.product_names, active=0)
        _set_checkbox_fontsize(self.radio, 10)
        self.radio.on_clicked(self._on_product_selected)

        layer_ax = self.figure.add_axes([0.03, 0.48, 0.13, 0.17], facecolor="#f1f3f5")
        self.layer_check = CheckButtons(layer_ax, list(self.layer_visibility.keys()), list(self.layer_visibility.values()))
        _set_checkbox_fontsize(self.layer_check, 9.5)
        self.layer_check.on_clicked(self._on_layer_toggled)

        level_ax = self.figure.add_axes([0.03, 0.32, 0.13, 0.12], facecolor="#f1f3f5")
        self.level_check = CheckButtons(level_ax, [f"L{level}" for level in PRICE_LEVELS], [True, True, True])
        _set_checkbox_fontsize(self.level_check, 9.5)
        self.level_check.on_clicked(self._on_level_toggled)

        day_labels = [f"D{day}" for day in self.available_days]
        day_ax = self.figure.add_axes([0.03, 0.18, 0.13, 0.10], facecolor="#f1f3f5")
        self.day_check = CheckButtons(day_ax, day_labels, [True for _ in self.available_days])
        _set_checkbox_fontsize(self.day_check, 9.5)
        self.day_check.on_clicked(self._on_day_toggled)

        slider_ax = self.figure.add_axes([0.035, 0.11, 0.12, 0.035], facecolor="#f1f3f5")
        self.trade_slider = RangeSlider(
            slider_ax,
            "Trade qty",
            self.trade_quantity_bounds[0],
            self.trade_quantity_bounds[1],
            valinit=self.trade_quantity_range,
        )
        self.trade_slider.on_changed(self._on_trade_filter_changed)

        self.figure.text(0.03, 0.69, "Product", fontsize=11, weight="bold", color="#212529")
        self.figure.text(0.03, 0.65, "Layers", fontsize=11, weight="bold", color="#212529")
        self.figure.text(0.03, 0.445, "Depth levels", fontsize=11, weight="bold", color="#212529")
        self.figure.text(0.03, 0.295, "Days", fontsize=11, weight="bold", color="#212529")
        self.figure.text(0.03, 0.08, "Left/right arrows switch products.", fontsize=9, color="#495057")
        self.figure.text(0.03, 0.055, "Normalize centers prices on quoted mid.", fontsize=9, color="#495057")
        self.figure.text(0.23, 0.08, "Drag top chart to zoom time. Scroll zooms time. Shift+scroll zooms price. Double-click or press r to reset.", fontsize=9, color="#495057")

        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.figure.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.figure.canvas.mpl_connect("figure_leave_event", self._on_figure_leave)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("button_press_event", self._on_button_press)

        self.hover_annotation = None
        self.hover_vline = None
        self.hover_hline = None

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

        day = int(label.replace("D", ""))
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
        if self.layer_visibility["Trades"] and not self.current_trade_points.empty:
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

        if self.layer_visibility["Normalize"]:
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

        for axis in (self.price_ax, self.spread_ax, self.volume_ax):
            axis.set_xlim(lower, upper)
        self._autoscale_price_axis()
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes not in {self.price_ax, self.spread_ax, self.volume_ax} or event.xdata is None:
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
        if event.dblclick and event.inaxes in {self.price_ax, self.spread_ax, self.volume_ax}:
            self._reset_time_window_to_selected_days()
            self.price_zoom_scale = 1.0
            self._apply_time_window()

    def _on_span_select(self, xmin: float, xmax: float) -> None:
        if xmax - xmin <= 1.0:
            return
        self.time_window = (min(xmin, xmax), max(xmin, xmax))
        self._apply_time_window()

    def _on_mouse_move(self, event) -> None:
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
        normalize = self.layer_visibility["Normalize"]
        active_levels = {level for level, enabled in self.level_visibility.items() if enabled}
        if not active_levels:
            active_levels = {1}

        bid_points, ask_points = _build_quote_points(product_prices, normalize=normalize, active_levels=active_levels)
        trade_points = _build_trade_points(product_trades, normalize=normalize, quantity_range=self.trade_quantity_range)
        self.hover_points = _build_hover_points(bid_points, ask_points, trade_points)
        self.current_price_frame = product_prices

        _shade_days(self.price_ax, product_prices)
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

        if self.layer_visibility["Trades"] and not trade_points.empty:
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
                        label=f"{side.title()} trades",
                    )
                )

        self.price_ax.set_title(
            f"{product} order book over time{' | normalized to quoted mid' if normalize else ''}",
            fontsize=13,
            pad=10,
        )
        self.price_ax.set_ylabel("Price - quoted mid" if normalize else "Price")
        _apply_day_ticks(self.price_ax, product_prices)
        self.price_ax.set_xlabel("Round 1 timeline")
        self.price_ax.grid(alpha=0.18, color="#dbe2ea")
        self.price_ax.margins(x=0.01)

        valid_mid = clean_mid.dropna()
        valid_spread = product_prices["spread"].dropna()
        visible_volume = trade_points["quantity"].sum() if not trade_points.empty else 0
        summary_lines = [
            f"Last mid: {valid_mid.iloc[-1]:,.1f}" if not valid_mid.empty else "Last mid: n/a",
            f"Median spread: {valid_spread.median():.1f}" if not valid_spread.empty else "Median spread: n/a",
            f"Visible trades: {len(trade_points):,}",
            f"Visible qty: {int(visible_volume):,}",
            f"Levels: {', '.join(f'L{level}' for level in sorted(active_levels))}",
        ]
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

        visible_trade_volume = aggregate_trade_volume(trade_points) if not trade_points.empty else pd.DataFrame()
        bucketed_volume, bucket_width = _bucket_trade_volume(visible_trade_volume)
        if bucketed_volume.empty:
            self.volume_ax.text(0.5, 0.5, "No trades in filter", ha="center", va="center", transform=self.volume_ax.transAxes)
        else:
            bar_colors = np.where(
                bucketed_volume["signed_quantity"] > 0,
                SIDE_COLORS["buy"],
                np.where(bucketed_volume["signed_quantity"] < 0, SIDE_COLORS["sell"], SIDE_COLORS["neutral"]),
            )
            self.volume_ax.bar(
                bucketed_volume["bucket_timestamp"],
                bucketed_volume["traded_quantity"],
                color=bar_colors,
                width=bucket_width * 0.86,
                alpha=0.9,
                edgecolor="none",
            )
        self.volume_ax.set_title("Trade volume")
        self.volume_ax.set_ylabel("Quantity")
        _style_axis(self.volume_ax, product_prices)

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
        self.span_selector = SpanSelector(
            self.price_ax,
            self._on_span_select,
            "horizontal",
            useblit=True,
            props={"facecolor": "#74c0fc", "alpha": 0.18},
            interactive=False,
            drag_from_anywhere=False,
        )

    def render(self, product: str) -> None:
        self.current_product = product
        self.price_ax.clear()
        self.spread_ax.clear()
        self.volume_ax.clear()

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
        self.figure.suptitle(f"IMC Prosperity 4 Round 1 order-book explorer | {product}", fontsize=16, y=0.975)
        self._apply_time_window()
        self.figure.canvas.draw_idle()


def launch_interactive_dashboard(
    prices: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    products: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
) -> ProductToggleDashboard:
    return ProductToggleDashboard(prices, trades, products=products, days=days)

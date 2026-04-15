from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from visualizer.analytics import add_order_book_features


def create_order_book_snapshot(
    prices: pd.DataFrame,
    product: str,
    day: int,
    timestamp: int,
) -> Figure:
    filtered = add_order_book_features(prices)
    filtered = filtered[(filtered["product"] == product) & (filtered["day"] == day)].copy()
    if filtered.empty:
        raise ValueError(f"No price data found for product={product!r} on day={day}.")

    nearest_index = (filtered["timestamp"] - timestamp).abs().idxmin()
    snapshot = filtered.loc[nearest_index]

    figure, ax = plt.subplots(figsize=(10, 5.5))

    bid_rows: list[tuple[float, float, int]] = []
    ask_rows: list[tuple[float, float, int]] = []
    for level in (1, 2, 3):
        bid_price = snapshot.get(f"bid_price_{level}")
        bid_volume = snapshot.get(f"bid_volume_{level}")
        ask_price = snapshot.get(f"ask_price_{level}")
        ask_volume = snapshot.get(f"ask_volume_{level}")

        if pd.notna(bid_price) and pd.notna(bid_volume):
            bid_rows.append((float(bid_price), -level, int(bid_volume)))
        if pd.notna(ask_price) and pd.notna(ask_volume):
            ask_rows.append((float(ask_price), level, int(ask_volume)))

    if not bid_rows and not ask_rows:
        raise ValueError("Selected snapshot has no visible order-book levels.")

    if bid_rows:
        bid_prices, bid_levels, bid_volumes = zip(*bid_rows)
        ax.scatter(
            bid_prices,
            bid_levels,
            s=[max(volume, 1) * 22 for volume in bid_volumes],
            color="#1d3557",
            alpha=0.85,
            label="Bids",
        )
        for price, level, volume in bid_rows:
            ax.annotate(f"L{abs(level)} | {volume}", (price, level), xytext=(4, -2), textcoords="offset points", fontsize=9)

    if ask_rows:
        ask_prices, ask_levels, ask_volumes = zip(*ask_rows)
        ax.scatter(
            ask_prices,
            ask_levels,
            s=[max(volume, 1) * 22 for volume in ask_volumes],
            color="#e63946",
            alpha=0.85,
            label="Asks",
        )
        for price, level, volume in ask_rows:
            ax.annotate(f"L{level} | {volume}", (price, level), xytext=(4, 6), textcoords="offset points", fontsize=9)

    if pd.notna(snapshot["quoted_mid_price"]):
        ax.axvline(snapshot["quoted_mid_price"], color="#2a9d8f", linestyle="--", linewidth=1.0, label="Quoted mid")

    ax.axhline(0, color="#adb5bd", linewidth=1.0)
    ax.set_yticks([-3, -2, -1, 1, 2, 3], ["Bid L3", "Bid L2", "Bid L1", "Ask L1", "Ask L2", "Ask L3"])
    ax.set_xlabel("Price")
    ax.set_ylabel("Order-book level")
    ax.set_title(
        f"{product} order book | requested day {day} ts {timestamp} | nearest ts {int(snapshot['timestamp'])}"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    minimum_price = math.floor(min([row[0] for row in bid_rows + ask_rows])) - 1
    maximum_price = math.ceil(max([row[0] for row in bid_rows + ask_rows])) + 1
    ax.set_xlim(minimum_price, maximum_price)

    figure.tight_layout()
    return figure

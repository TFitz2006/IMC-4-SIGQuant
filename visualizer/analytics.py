from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_LEVELS = (1, 2, 3)


def add_order_book_features(prices: pd.DataFrame) -> pd.DataFrame:
    enriched = prices.copy()
    bid_columns = [f"bid_price_{level}" for level in PRICE_LEVELS if f"bid_price_{level}" in enriched.columns]
    ask_columns = [f"ask_price_{level}" for level in PRICE_LEVELS if f"ask_price_{level}" in enriched.columns]

    enriched["best_bid"] = enriched[bid_columns].max(axis=1, skipna=True)
    enriched["best_ask"] = enriched[ask_columns].min(axis=1, skipna=True)
    enriched["spread"] = enriched["best_ask"] - enriched["best_bid"]
    enriched["quoted_mid_price"] = (enriched["best_bid"] + enriched["best_ask"]) / 2.0

    return enriched


def infer_trade_aggressor(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        inferred = trades.copy()
        inferred["aggressor"] = pd.Series(dtype="string")
        inferred["touch_relation"] = pd.Series(dtype="string")
        return inferred

    reference_book = (
        add_order_book_features(prices)[["product", "day", "global_timestamp", "best_bid", "best_ask", "quoted_mid_price"]]
        .sort_values(["product", "day", "global_timestamp"])
        .reset_index(drop=True)
    )

    aligned_groups: list[pd.DataFrame] = []
    ordered_trades = trades.sort_values(["product", "day", "global_timestamp"]).reset_index(drop=True)

    for (product, day), trade_slice in ordered_trades.groupby(["product", "day"], sort=False):
        book_slice = reference_book[(reference_book["product"] == product) & (reference_book["day"] == day)]
        trade_slice = trade_slice.sort_values("global_timestamp").reset_index(drop=True)

        if book_slice.empty:
            merged_slice = trade_slice.copy()
            merged_slice["best_bid"] = np.nan
            merged_slice["best_ask"] = np.nan
            merged_slice["quoted_mid_price"] = np.nan
        else:
            merged_slice = pd.merge_asof(
                trade_slice,
                book_slice.sort_values("global_timestamp")[["global_timestamp", "best_bid", "best_ask", "quoted_mid_price"]],
                on="global_timestamp",
                direction="backward",
                allow_exact_matches=True,
            )

        aligned_groups.append(merged_slice)

    inferred = pd.concat(aligned_groups, ignore_index=True)

    trade_price = inferred["price"]
    best_bid = inferred["best_bid"]
    best_ask = inferred["best_ask"]

    aggressor = np.select(
        [
            trade_price.ge(best_ask) & best_ask.notna(),
            trade_price.le(best_bid) & best_bid.notna(),
        ],
        [
            "buy",
            "sell",
        ],
        default="neutral",
    )

    touch_relation = np.select(
        [
            trade_price.eq(best_ask) & best_ask.notna(),
            trade_price.eq(best_bid) & best_bid.notna(),
            trade_price.gt(best_ask) & best_ask.notna(),
            trade_price.lt(best_bid) & best_bid.notna(),
        ],
        [
            "at_ask",
            "at_bid",
            "through_ask",
            "through_bid",
        ],
        default="inside_or_unknown",
    )

    inferred["aggressor"] = pd.Series(aggressor, dtype="string")
    inferred["touch_relation"] = pd.Series(touch_relation, dtype="string")
    return inferred


def aggregate_trade_volume(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["product", "day", "global_timestamp", "traded_quantity", "trade_count", "notional", "vwap", "signed_quantity"]
        )

    working = trades.copy()
    working["notional"] = working["price"] * working["quantity"]
    side_map = {"buy": 1, "sell": -1, "neutral": 0}
    if "aggressor" not in working.columns:
        working["aggressor"] = "neutral"
    working["signed_quantity"] = working["aggressor"].map(side_map).fillna(0) * working["quantity"]

    grouped = (
        working.groupby(["product", "day", "global_timestamp"], as_index=False)
        .agg(
            traded_quantity=("quantity", "sum"),
            trade_count=("quantity", "size"),
            notional=("notional", "sum"),
            signed_quantity=("signed_quantity", "sum"),
        )
    )
    grouped["vwap"] = grouped["notional"] / grouped["traded_quantity"]
    return grouped

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_ROUND1_ROOT = Path("DataCapsules/ROUND1")
PRICE_GLOB = "prices_round_1_day_*.csv"
TRADE_GLOB = "trades_round_1_day_*.csv"
DAY_PATTERN = re.compile(r"day_(-?\d+)\.csv$")


def _extract_day(path: Path) -> int:
    match = DAY_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not infer day from file name: {path}")
    return int(match.group(1))


def _sorted_paths(paths: Iterable[Path | str]) -> list[Path]:
    return sorted((Path(path) for path in paths), key=_extract_day)


def _infer_tick_size(frame: pd.DataFrame) -> int:
    deltas = (
        frame.sort_values(["day", "timestamp"])
        .groupby("day")["timestamp"]
        .diff()
        .dropna()
    )
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return 100
    return int(positive_deltas.mode().iloc[0])


def _add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    typed = frame.copy()
    typed["day"] = pd.to_numeric(typed["day"], errors="raise").astype(int)
    typed["timestamp"] = pd.to_numeric(typed["timestamp"], errors="raise").astype(int)

    tick_size = _infer_tick_size(typed)
    day_span = int(typed.groupby("day")["timestamp"].max().max()) + tick_size
    day_values = sorted(typed["day"].unique().tolist())
    day_order = {day: index for index, day in enumerate(day_values)}

    typed["day_sequence"] = typed["day"].map(day_order).astype(int)
    typed["global_timestamp"] = typed["day_sequence"] * day_span + typed["timestamp"]
    typed["day_label"] = typed["day"].map(lambda day: f"Day {day}")

    return typed


def discover_round1_files(root: Path | str = DEFAULT_ROUND1_ROOT) -> tuple[list[Path], list[Path]]:
    root_path = Path(root)
    price_paths = _sorted_paths(root_path.joinpath("Prices").glob(PRICE_GLOB))
    trade_paths = _sorted_paths(root_path.joinpath("Trades").glob(TRADE_GLOB))

    if not price_paths:
        raise FileNotFoundError(f"No price files found under {root_path / 'Prices'}")
    if not trade_paths:
        raise FileNotFoundError(f"No trade files found under {root_path / 'Trades'}")

    return price_paths, trade_paths


def _normalize_text_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            continue
        normalized[column] = normalized[column].replace("", pd.NA).astype("string")
    return normalized


def load_price_data(
    paths: Sequence[Path | str] | None = None,
    root: Path | str = DEFAULT_ROUND1_ROOT,
) -> pd.DataFrame:
    price_paths = _sorted_paths(paths) if paths is not None else discover_round1_files(root)[0]
    frames: list[pd.DataFrame] = []

    for path in price_paths:
        frame = pd.read_csv(path, sep=";")
        frame["source_file"] = path.name
        frame["day"] = pd.to_numeric(frame.get("day", _extract_day(path)), errors="coerce").fillna(_extract_day(path))
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices = _normalize_text_columns(prices, ["product", "source_file"])

    numeric_columns = [column for column in prices.columns if column not in {"product", "source_file"}]
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return _add_time_columns(prices).sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


def load_trade_data(
    paths: Sequence[Path | str] | None = None,
    root: Path | str = DEFAULT_ROUND1_ROOT,
) -> pd.DataFrame:
    trade_paths = _sorted_paths(paths) if paths is not None else discover_round1_files(root)[1]
    frames: list[pd.DataFrame] = []

    for path in trade_paths:
        frame = pd.read_csv(path, sep=";")
        frame["day"] = _extract_day(path)
        frame["source_file"] = path.name
        frames.append(frame)

    trades = pd.concat(frames, ignore_index=True)
    trades = _normalize_text_columns(trades, ["buyer", "seller", "symbol", "currency", "source_file"])
    trades["product"] = trades["symbol"]

    numeric_columns = [column for column in trades.columns if column not in {"buyer", "seller", "symbol", "currency", "product", "source_file"}]
    for column in numeric_columns:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")

    return _add_time_columns(trades).sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


def load_round1_data(root: Path | str = DEFAULT_ROUND1_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_paths, trade_paths = discover_round1_files(root)
    return load_price_data(price_paths), load_trade_data(trade_paths)

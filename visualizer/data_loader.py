from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_ROUND1_ROOT = Path("DataCapsules/ROUND1")
DEFAULT_LOGS_ROOT = Path("OfficialLogs")
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


# -----------------------------------------------------------------------------
# Official IMC Submission Log Loading
# -----------------------------------------------------------------------------


def discover_official_logs(root: Path | str = DEFAULT_LOGS_ROOT) -> list[Path]:
    """Find all run directories containing official submission logs."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    run_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Look for .json or .log files inside
            json_files = list(item.glob("*.json"))
            log_files = list(item.glob("*.log"))
            if json_files or log_files:
                run_dirs.append(item)

    return sorted(run_dirs, key=lambda p: p.name)


def load_official_log(run_dir: Path | str, run_day: int = 1) -> dict:
    """
    Load official submission logs from a run directory.

    Args:
        run_dir: Path to the run directory
        run_day: Day number to assign to this run's data (default 1, after DataCapsules day 0)

    Returns a dict with:
        - run_id: str
        - json_data: dict (from .json file) or None
        - log_data: dict (from .log file) or None
        - your_trades: pd.DataFrame (trades made by your algorithm)
        - run_prices: pd.DataFrame (order book data from the run)
        - run_market_trades: pd.DataFrame (market trades from the run)
        - pnl_series: pd.DataFrame (P&L over time)
        - final_positions: list[dict]
        - final_profit: float
        - status: str
    """
    run_path = Path(run_dir)
    json_files = list(run_path.glob("*.json"))
    log_files = list(run_path.glob("*.log"))

    result = {
        "run_id": run_path.name,
        "run_path": run_path,
        "run_day": run_day,
        "json_data": None,
        "log_data": None,
        "your_trades": pd.DataFrame(),
        "run_prices": pd.DataFrame(),
        "run_market_trades": pd.DataFrame(),
        "pnl_series": pd.DataFrame(),
        "final_positions": [],
        "final_profit": 0.0,
        "status": "UNKNOWN",
        "algorithm_logs": pd.DataFrame(),
    }

    # Load JSON file (results summary)
    if json_files:
        with open(json_files[0], "r") as f:
            result["json_data"] = json.load(f)

        json_data = result["json_data"]
        result["status"] = json_data.get("status", "UNKNOWN")
        result["final_profit"] = float(json_data.get("profit", 0.0))
        result["final_positions"] = json_data.get("positions", [])

        # Parse graphLog (P&L over time)
        graph_log = json_data.get("graphLog", "")
        if graph_log:
            result["pnl_series"] = _parse_pnl_log(graph_log, day=run_day)

        # Parse activitiesLog (order book data from the run)
        activities_log = json_data.get("activitiesLog", "")
        if activities_log:
            result["run_prices"] = _parse_activities_log(activities_log, day=run_day)

    # Load .log file (detailed run data)
    if log_files:
        with open(log_files[0], "r") as f:
            result["log_data"] = json.load(f)

        log_data = result["log_data"]

        # Parse tradeHistory (YOUR trades)
        trade_history = log_data.get("tradeHistory", [])
        if trade_history:
            result["your_trades"] = _parse_trade_history(trade_history, day=run_day)

        # Parse algorithm logs
        logs = log_data.get("logs", [])
        if logs:
            result["algorithm_logs"] = _parse_algorithm_logs(logs, day=run_day)

    return result


def _parse_pnl_log(graph_log: str, day: int = 1) -> pd.DataFrame:
    """Parse the graphLog string into a DataFrame."""
    if not graph_log:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(graph_log), sep=";")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"value": "pnl"})

    df["day"] = day
    df = _add_time_columns(df)

    return df


def _parse_activities_log(activities_log: str, day: int = 1) -> pd.DataFrame:
    """Parse the activitiesLog string into a price DataFrame (same format as DataCapsules)."""
    if not activities_log:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(activities_log), sep=";")

    # Override day with our run_day
    df["day"] = day
    df["source_file"] = f"official_run_day_{day}"

    # Normalize text columns
    df = _normalize_text_columns(df, ["product", "source_file"])

    # Convert numeric columns
    numeric_columns = [col for col in df.columns if col not in {"product", "source_file"}]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _add_time_columns(df)
    return df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


def _parse_trade_history(trade_history: list[dict], day: int = 1) -> pd.DataFrame:
    """Parse the tradeHistory list into a DataFrame with side inference."""
    if not trade_history:
        return pd.DataFrame()

    df = pd.DataFrame(trade_history)

    # Normalize columns
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["product"] = df["symbol"]

    # Determine side: SUBMISSION as buyer = you bought, SUBMISSION as seller = you sold
    df["your_side"] = df.apply(
        lambda row: "buy" if row.get("buyer") == "SUBMISSION" else "sell",
        axis=1
    )

    # Signed quantity for position tracking
    df["signed_quantity"] = df.apply(
        lambda row: row["quantity"] if row["your_side"] == "buy" else -row["quantity"],
        axis=1
    )

    df["day"] = day
    df = _add_time_columns(df)

    return df


def _parse_algorithm_logs(logs: list[dict], day: int = 1) -> pd.DataFrame:
    """Parse algorithm print statements from logs."""
    if not logs:
        return pd.DataFrame()

    records = []
    for entry in logs:
        sandbox_log = entry.get("sandboxLog", "")
        lambda_log = entry.get("lambdaLog", "")
        timestamp = entry.get("timestamp", 0)

        # Only keep entries with actual log content
        if sandbox_log or lambda_log:
            records.append({
                "timestamp": timestamp,
                "sandbox_log": sandbox_log,
                "lambda_log": lambda_log,
                "combined_log": f"{sandbox_log}{lambda_log}".strip(),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["day"] = day
    df = _add_time_columns(df)

    return df


def align_time_to_prices(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Re-index a frame's global_timestamp to match the time scale of a prices DataFrame."""
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


def build_run_payload(base_prices: pd.DataFrame, run_dir: Path | str, run_day: int = 1) -> dict:
    """
    Load a run directory and assemble a complete payload dict for the dashboard.

    Returns a dict with keys: name, prices, your_trades, pnl_series, final_profit,
    final_positions, status, run_prices_count.
    """
    run_path = Path(run_dir)
    log_data = load_official_log(run_path, run_day=run_day)

    merged_prices = base_prices.copy()
    run_prices = log_data.get("run_prices")
    if run_prices is not None and not run_prices.empty:
        merged_prices = pd.concat([merged_prices, run_prices], ignore_index=True, sort=False)
        merged_prices = recalculate_time_columns(merged_prices)
        merged_prices = merged_prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)

    your_trades = log_data.get("your_trades", pd.DataFrame()).copy()
    pnl_series = log_data.get("pnl_series", pd.DataFrame()).copy()
    if not your_trades.empty:
        your_trades = align_time_to_prices(your_trades, merged_prices)
    if not pnl_series.empty:
        pnl_series = align_time_to_prices(pnl_series, merged_prices)

    final_positions = {
        row["symbol"]: int(row["quantity"])
        for row in log_data.get("final_positions", [])
        if row.get("symbol") != "XIRECS"
    }

    return {
        "name": run_path.name,
        "prices": merged_prices,
        "your_trades": your_trades,
        "pnl_series": pnl_series,
        "final_profit": log_data.get("final_profit"),
        "final_positions": final_positions,
        "status": log_data.get("status", "UNKNOWN"),
        "run_prices_count": 0 if run_prices is None else len(run_prices),
    }


def load_all_official_logs(root: Path | str = DEFAULT_LOGS_ROOT) -> list[dict]:
    """Load all official logs from the logs directory."""
    run_dirs = discover_official_logs(root)
    return [load_official_log(run_dir) for run_dir in run_dirs]


def recalculate_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate global_timestamp and related columns after merging multiple datasets.
    Call this after concatenating dataframes with different day values.
    """
    if frame.empty:
        return frame

    # Drop old time columns if they exist
    cols_to_drop = ["day_sequence", "global_timestamp", "day_label"]
    existing_cols = [c for c in cols_to_drop if c in frame.columns]
    if existing_cols:
        frame = frame.drop(columns=existing_cols)

    # Recalculate with full day context
    return _add_time_columns(frame)

"""Binance CSV 数据加载器."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
RESAMPLE_RULES = {
    60: "1min",
    300: "5min",
    900: "15min",
    1800: "30min",
    3600: "1h",
    14400: "4h",
    86400: "1D",
}


def _normalize_timestamp(ts: int | float) -> float:
    if ts > 1e15:
        return float(ts) / 1_000_000
    if ts > 1e12:
        return float(ts) / 1_000
    return float(ts)


def available_symbols(data_dir: Path = DEFAULT_DATA_DIR) -> List[str]:
    if not data_dir.exists():
        return []
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def load_raw(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    sym_dir = data_dir / symbol
    if not sym_dir.exists():
        raise FileNotFoundError(
            f"No data directory for '{symbol}'. Available: {available_symbols(data_dir) or '(none)'}"
        )

    csv_files = sorted(sym_dir.glob(f"{symbol}-1m-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {sym_dir}")

    start_ym = None
    end_ym = None
    if start:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        start_ym = (start_dt.year, start_dt.month)
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        end_ym = (end_dt.year, end_dt.month)

    selected: List[Path] = []
    for path in csv_files:
        parts = path.stem.split("-")
        try:
            year, month = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            continue
        if start_ym and (year, month) < start_ym:
            continue
        if end_ym and (year, month) > end_ym:
            continue
        selected.append(path)

    if not selected:
        raise FileNotFoundError(f"No CSV files for {symbol} in range [{start}, {end})")

    frames = [pd.read_csv(path, header=None, names=BINANCE_COLUMNS) for path in selected]
    df = pd.concat(frames, ignore_index=True)
    df["open_time_sec"] = df["open_time"].apply(_normalize_timestamp)
    df["datetime"] = pd.to_datetime(df["open_time_sec"], unit="s", utc=True)

    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").fillna(0).astype(int)

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]

    return df


def resample(df_1m: pd.DataFrame, bar_seconds: int = 3600) -> pd.DataFrame:
    if bar_seconds == 60:
        return df_1m.copy()

    rule = RESAMPLE_RULES.get(bar_seconds, f"{bar_seconds}s")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "num_trades": "sum",
        "taker_buy_base_vol": "sum",
        "taker_buy_quote_vol": "sum",
    }
    return df_1m.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])


def load(
    symbol: str = "BTCUSDT",
    start: Optional[str] = None,
    end: Optional[str] = None,
    bar_seconds: int = 3600,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    return resample(load_raw(symbol=symbol, start=start, end=end, data_dir=data_dir), bar_seconds=bar_seconds)


def to_bar_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dt, row in df.iterrows():
        rows.append(
            {
                "datetime": dt.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "quote_volume": float(row.get("quote_volume", 0.0)),
                "num_trades": int(row.get("num_trades", 0)),
                "taker_buy_base_vol": float(row.get("taker_buy_base_vol", 0.0)),
                "taker_buy_quote_vol": float(row.get("taker_buy_quote_vol", 0.0)),
            }
        )
    return rows

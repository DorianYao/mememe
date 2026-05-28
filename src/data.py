from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import ExperimentConfig


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _lookback_start_ms(days: int) -> int:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    return int(start.timestamp() * 1000)


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    df = df.loc[:, [c for c in columns if c in df.columns]].copy()
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in columns:
        if column == "timestamp" or column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return (
        df.dropna()
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_klines(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 1000,
    max_retries: int = 3,
) -> pd.DataFrame:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported Binance interval: {interval}")

    load_dotenv()
    rows: list[list[object]] = []
    next_start = start_time_ms
    session = requests.Session()

    while next_start < end_time_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": next_start,
            "endTime": end_time_ms,
            "limit": limit,
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(BINANCE_KLINES_URL, params=params, timeout=20)
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch klines for {symbol} {interval}: {exc}"
                    ) from exc
                time.sleep(1.5 * attempt)
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_start = last_open_time + INTERVAL_MS[interval]
        time.sleep(0.15)
        if len(batch) < limit:
            break

    if not rows:
        raise RuntimeError(
            f"Binance returned no rows for {symbol} {interval} "
            f"between {start_time_ms} and {end_time_ms}."
        )

    raw = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    return _clean_ohlcv(raw)


def download_ohlcv(
    config: ExperimentConfig,
    refresh: bool = False,
    symbol: str | None = None,
) -> Path:
    """下载或复用已缓存的 K 线 CSV（默认使用 config.symbol）。"""
    config.ensure_dirs()
    target_symbol = symbol or config.symbol
    path = config.raw_csv_path_for(target_symbol)
    if path.exists() and not refresh:
        return path

    df = fetch_klines(
        symbol=target_symbol,
        interval=config.interval,
        start_time_ms=_lookback_start_ms(config.lookback_days),
        end_time_ms=_utc_now_ms(),
    )
    df.to_csv(path, index=False)
    return path


def download_multi(
    config: ExperimentConfig,
    symbols: list[str] | None = None,
    refresh: bool = False,
) -> dict[str, Path]:
    """批量下载多个币种。返回 {symbol: csv_path}。"""
    config.ensure_dirs()
    targets = symbols or list(config.symbols)
    saved: dict[str, Path] = {}
    for sym in targets:
        try:
            saved[sym.upper()] = download_ohlcv(config, refresh=refresh, symbol=sym)
        except RuntimeError as exc:
            print(f"[download] WARN: failed to fetch {sym}: {exc}")
    return saved


def load_ohlcv(config: ExperimentConfig, symbol: str | None = None) -> pd.DataFrame:
    target_symbol = symbol or config.symbol
    path = config.raw_csv_path_for(target_symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {path}. Run the download stage first."
        )
    return _clean_ohlcv(pd.read_csv(path))

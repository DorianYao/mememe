#!/usr/bin/env python3
"""Check whether all category-universe OHLCV CSVs are downloaded."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_all_category_data import all_category_symbols  # noqa: E402
from src.config import config_from_yaml  # noqa: E402


def raw_csv_path(data_dir: Path, symbol: str, interval: str, lookback_days: int) -> Path:
    return data_dir / "raw" / f"{symbol}_{interval}_{lookback_days}d.csv"


def check_ready(
    include_legacy: bool = True,
    config_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    config = config_from_yaml(config_path)
    targets = all_category_symbols(include_legacy=include_legacy)
    ready: list[str] = []
    missing: list[str] = []
    for symbol in targets:
        path = raw_csv_path(config.data_dir, symbol, config.interval, config.lookback_days)
        if path.exists() and path.stat().st_size > 0:
            ready.append(symbol)
        else:
            missing.append(symbol)
    return ready, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Exit 0 if ready, 1 if not; minimal stdout.",
    )
    args = parser.parse_args()

    ready, missing = check_ready(
        include_legacy=not args.no_legacy,
        config_path=args.config,
    )
    total = len(ready) + len(missing)
    if args.quiet:
        if missing:
            sys.exit(1)
        sys.exit(0)

    print(f"Data readiness: {len(ready)}/{total} symbols")
    if missing:
        print("\nMissing:")
        for sym in missing:
            print(f"  - {sym}")
    else:
        print("All category symbols ready.")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

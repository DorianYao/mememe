#!/usr/bin/env python3
"""Download Binance OHLCV for all five categories (deduplicated union)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.categories import CATEGORY_IDS, get_legacy_meme8_symbols, get_symbols  # noqa: E402
from src.config import config_from_yaml  # noqa: E402
from src.data import download_multi  # noqa: E402


def all_category_symbols(include_legacy: bool = True) -> list[str]:
    symbols: set[str] = set()
    for cat in CATEGORY_IDS:
        symbols.update(get_symbols(cat))
    if include_legacy:
        symbols.update(get_legacy_meme8_symbols())
    return sorted(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download even if CSV exists.")
    parser.add_argument("--no-legacy", action="store_true", help="Skip legacy meme8 symbols.")
    args = parser.parse_args()

    config = config_from_yaml()
    targets = all_category_symbols(include_legacy=not args.no_legacy)
    print(f"Downloading {len(targets)} unique symbols ({config.interval}, {config.lookback_days}d)...")
    saved = download_multi(config, symbols=targets, refresh=args.refresh)
    failed = set(targets) - set(saved.keys())
    print(f"OK: {len(saved)} / {len(targets)}")
    if failed:
        print("Failed:", ", ".join(sorted(failed)))
        sys.exit(1)


if __name__ == "__main__":
    main()

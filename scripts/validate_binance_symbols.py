#!/usr/bin/env python3
"""Validate category symbol lists against Binance spot USDT markets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES_PATH = ROOT / "config" / "categories.yaml"
RESOLVED_PATH = ROOT / "config" / "categories.resolved.yaml"
EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

# Suggested substitutes when a doc ticker is not listed (same narrative bucket).
SUBSTITUTE_HINTS: dict[str, list[str]] = {
    "SPXUSDT": ["MEMEUSDT", "TURBOUSDT", "PNUTUSDT"],
    "MOGUSDT": ["PEOPLEUSDT", "NEIROUSDT"],
    "POPCATUSDT": ["PNUTUSDT", "1000CATUSDT"],
    "FARTCOINUSDT": ["TURBOUSDT", "ACTUSDT"],
    "BRETTUSDT": ["PENGUUSDT", "MEMEUSDT"],
    "BABYDOGEUSDT": ["1MBABYDOGEUSDT", "BABYUSDT"],
    "GIGAUSDT": ["BOMEUSDT", "1000CHEEMSUSDT"],
    "GOATUSDT": ["PNUTUSDT", "ACTUSDT"],
    "SLERFUSDT": ["BONKUSDT", "WIFUSDT"],
    "MEWUSDT": ["1000CATUSDT", "CATIUSDT"],
    "CHILLGUYUSDT": ["MEMEUSDT", "DOGSUSDT"],
    "MICHIUSDT": ["1000CATUSDT", "PNUTUSDT"],
    "BOOKOFMEMEUSDT": ["MEMEUSDT"],
    "MEMECOINUSDT": ["MEMEUSDT"],
}


def fetch_trading_usdt_symbols() -> set[str]:
    response = requests.get(EXCHANGE_INFO_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return {
        item["symbol"]
        for item in payload["symbols"]
        if item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
        and str(item.get("symbol", "")).endswith("USDT")
    }


def load_categories(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if "categories" not in data:
        raise ValueError(f"No 'categories' key in {path}")
    return data


def validate_category(
    category_id: str,
    entry: dict,
    trading: set[str],
    expected_n: int = 16,
) -> tuple[list[str], list[dict]]:
    symbols = [s.upper() for s in entry.get("symbols", [])]
    invalid = [s for s in symbols if s not in trading]
    issues: list[dict] = []
    for sym in invalid:
        hints = SUBSTITUTE_HINTS.get(sym, [])
        available_hints = [h for h in hints if h in trading]
        issues.append(
            {
                "symbol": sym,
                "category": category_id,
                "substitute_hints": available_hints,
            }
        )
    if len(symbols) != expected_n:
        issues.append(
            {
                "symbol": None,
                "category": category_id,
                "error": f"expected {expected_n} symbols, got {len(symbols)}",
            }
        )
    return symbols, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CATEGORIES_PATH,
        help="Path to categories.yaml",
    )
    parser.add_argument(
        "--write-resolved",
        action="store_true",
        help="Write config/categories.resolved.yaml with validation metadata",
    )
    parser.add_argument(
        "--expected-per-category",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    data = load_categories(args.config)
    trading = fetch_trading_usdt_symbols()

    all_issues: list[dict] = []
    resolved_categories: dict = {}

    for category_id, entry in data["categories"].items():
        symbols, issues = validate_category(
            category_id, entry, trading, expected_n=args.expected_per_category
        )
        all_issues.extend(issues)
        invalid = [i["symbol"] for i in issues if i.get("symbol")]
        resolved_categories[category_id] = {
            **entry,
            "symbols": symbols,
            "validation": {
                "all_trading": len(invalid) == 0 and len(symbols) == args.expected_per_category,
                "invalid_symbols": invalid,
            },
        }
        status = "OK" if not invalid and len(symbols) == args.expected_per_category else "FAIL"
        print(f"[{status}] {category_id}: {len(symbols)} symbols")
        for issue in issues:
            if issue.get("symbol"):
                print(f"  invalid: {issue['symbol']} hints={issue.get('substitute_hints', [])}")
            elif issue.get("error"):
                print(f"  {issue['error']}")

    if "legacy" in data and "meme8" in data["legacy"]:
        legacy_syms = [s.upper() for s in data["legacy"]["meme8"]["symbols"]]
        bad = [s for s in legacy_syms if s not in trading]
        if bad:
            all_issues.append({"category": "legacy.meme8", "invalid_symbols": bad})
            print(f"[FAIL] legacy.meme8 invalid: {bad}")
        else:
            print(f"[OK] legacy.meme8: {len(legacy_syms)} symbols")

    if args.write_resolved:
        out = {
            "categories": resolved_categories,
            "legacy": data.get("legacy", {}),
        }
        RESOLVED_PATH.write_text(
            yaml.safe_dump(out, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"Wrote {RESOLVED_PATH}")

    if all_issues:
        report_path = ROOT / "outputs" / "metrics" / "binance_symbol_validation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(all_issues, indent=2), encoding="utf-8")
        print(f"Issues written to {report_path}")
        return 1

    print("All category symbol lists validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

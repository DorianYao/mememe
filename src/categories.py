from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT, ExperimentConfig

CATEGORIES_PATH = PROJECT_ROOT / "config" / "categories.yaml"
CATEGORY_IDS = ("bluechip", "midcap", "solana_fast", "base_eco", "micro_cap")
SYMBOLS_PER_CATEGORY = 16


def categories_yaml_path(path: Path | None = None) -> Path:
    return path or CATEGORIES_PATH


def load_categories_raw(path: Path | None = None) -> dict[str, Any]:
    yaml_path = categories_yaml_path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Categories config not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if "categories" not in data:
        raise ValueError(f"No 'categories' in {yaml_path}")
    return data


def list_categories() -> list[str]:
    data = load_categories_raw()
    return [cid for cid in CATEGORY_IDS if cid in data["categories"]]


def get_category_entry(category_id: str, path: Path | None = None) -> dict[str, Any]:
    data = load_categories_raw(path)
    key = category_id.lower()
    if key not in data["categories"]:
        available = ", ".join(sorted(data["categories"].keys()))
        raise KeyError(f"Unknown category '{category_id}'. Available: {available}")
    return data["categories"][key]


def get_symbols(category_id: str, path: Path | None = None) -> tuple[str, ...]:
    entry = get_category_entry(category_id, path)
    symbols = tuple(s.upper() for s in entry["symbols"])
    if len(symbols) != SYMBOLS_PER_CATEGORY:
        raise ValueError(
            f"Category {category_id} must have {SYMBOLS_PER_CATEGORY} symbols, "
            f"got {len(symbols)}"
        )
    return symbols


def get_legacy_meme8_symbols(path: Path | None = None) -> tuple[str, ...]:
    data = load_categories_raw(path)
    legacy = data.get("legacy", {}).get("meme8", {})
    symbols = legacy.get("symbols")
    if not symbols:
        raise ValueError("legacy.meme8 symbols missing in categories.yaml")
    return tuple(s.upper() for s in symbols)


def apply_category(config: ExperimentConfig, category_id: str) -> ExperimentConfig:
    """Return a copy of config with category symbols and category_id set."""
    symbols = get_symbols(category_id)
    return replace(
        config,
        category_id=category_id.lower(),
        symbols=symbols,
    )


def load_category_config(
    category_id: str,
    base_config_path: Path | None = None,
    categories_path: Path | None = None,
) -> ExperimentConfig:
    from .config import config_from_yaml

    base = config_from_yaml(base_config_path)
    return apply_category(base, category_id)


def loso_summary_glob_pattern(category_id: str | None = None) -> str:
    """Glob pattern for LOSO summary JSON under outputs/metrics."""
    if category_id:
        return f"{category_id.lower()}16_*_loso_*_summary.json"
    return "meme*_loso_*_summary.json"


def metrics_universe_glob(category_id: str | None = None) -> str:
    """Glob prefix for outputs/metrics artifacts (legacy meme* vs category16_*)."""
    if category_id:
        return f"{category_id.lower()}16_*"
    return "meme*"

"""Tests for five-category × 16-coin configuration."""

from __future__ import annotations

import pytest

from src.categories import (
    CATEGORY_IDS,
    SYMBOLS_PER_CATEGORY,
    apply_category,
    get_symbols,
    list_categories,
    load_categories_raw,
    loso_summary_glob_pattern,
    metrics_universe_glob,
)
from src.config import ExperimentConfig, config_from_yaml


@pytest.mark.parametrize("category_id", CATEGORY_IDS)
def test_each_category_has_sixteen_symbols(category_id: str) -> None:
    symbols = get_symbols(category_id)
    assert len(symbols) == SYMBOLS_PER_CATEGORY
    assert len(set(symbols)) == SYMBOLS_PER_CATEGORY


def test_list_categories_matches_yaml() -> None:
    data = load_categories_raw()
    assert set(list_categories()) == set(data["categories"].keys())


def test_universe_tag_uses_category_prefix() -> None:
    config = apply_category(config_from_yaml(), "bluechip")
    tag = config.universe_tag("loso", "DOGEUSDT")
    assert tag.startswith("bluechip16_15m_w")
    assert "_loso_DOGEUSDT" in tag
    assert "meme8" not in tag


def test_legacy_meme8_tag_unchanged() -> None:
    config = config_from_yaml()
    assert config.category_id is None
    tag = config.universe_tag("loso", "DOGEUSDT")
    assert tag.startswith("meme8_")


def test_category_tags_do_not_collide() -> None:
    tags = set()
    for cid in CATEGORY_IDS:
        cfg = apply_category(ExperimentConfig(), cid)
        tags.add(cfg.universe_tag("loso", "DOGEUSDT"))
    assert len(tags) == len(CATEGORY_IDS)


def test_glob_helpers() -> None:
    assert "bluechip16_" in loso_summary_glob_pattern("bluechip")
    assert loso_summary_glob_pattern(None) == "meme*_loso_*_summary.json"
    assert metrics_universe_glob("midcap") == "midcap16_*"


def test_apply_category_sets_id() -> None:
    cfg = apply_category(ExperimentConfig(), "solana_fast")
    assert cfg.category_id == "solana_fast"
    assert len(cfg.symbols) == 16

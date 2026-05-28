"""Tests for LOSO / walk-forward calendar splits."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import ExperimentConfig
from src.multi import (
    _apply_mask,
    _mask_by_time,
    build_loso_splits,
    compute_calendar_boundaries,
)


def _synthetic_per_symbol() -> dict:
    """Two symbols, 100 hourly-ish timestamps."""
    n = 100
    ts = np.array([f"2024-01-01T{i:02d}:00:00+00:00" for i in range(n)])
    x = np.random.randn(n, 4, 3).astype(np.float32)
    y = (np.random.rand(n) > 0.5).astype(np.float32)
    sy = np.array(["A"] * n, dtype=object)
    sy_b = np.array(["B"] * n, dtype=object)
    return {
        "A": (x, y, ts, sy),
        "B": (x, y, ts.copy(), sy_b),
    }


def test_calendar_boundaries_ordered():
    per = _synthetic_per_symbol()
    config = ExperimentConfig(
        symbols=("A", "B"),
        split_mode="calendar",
        time_train_fraction=0.70,
        time_val_fraction=0.85,
    )
    train_end, val_end = compute_calendar_boundaries(per, ["A", "B"], config)
    assert train_end < val_end


def test_calendar_loso_temporal_invariant(monkeypatch):
    per = _synthetic_per_symbol()
    config = ExperimentConfig(
        symbols=("A", "B"),
        split_mode="calendar",
        window_size=4,
        time_train_fraction=0.70,
        time_val_fraction=0.85,
        label_k=0.1,
    )

    def fake_build(symbols, cfg, cache=None):
        return per

    monkeypatch.setattr("src.multi.build_per_symbol", fake_build)
    splits = build_loso_splits("B", config, symbols=["A", "B"])

    assert len(splits.train.y) > 0
    assert len(splits.val.y) > 0
    assert len(splits.test.y) > 0
    train_max = max(splits.train.timestamps.tolist())
    test_min = min(splits.test.timestamps.tolist())
    assert train_max < test_min


def test_mask_by_time_strict():
    ts = np.array(["2024-01-01T01:00:00+00:00", "2024-01-01T02:00:00+00:00"])
    m = _mask_by_time(ts, lower="2024-01-01T01:00:00+00:00", strict_lower=True)
    assert m.tolist() == [False, True]


def test_label_volatility_window_config():
    config = ExperimentConfig(label_volatility_window=200, volatility_window=50)
    assert config.label_volatility_window != config.volatility_window


def test_apply_mask_empty():
    x = np.zeros((0, 2, 3), dtype=np.float32)
    y = np.zeros(0, dtype=np.float32)
    ts = np.array([])
    sy = np.array([], dtype=object)
    m = np.array([], dtype=bool)
    out = _apply_mask(x, y, ts, sy, m)
    assert len(out[1]) == 0

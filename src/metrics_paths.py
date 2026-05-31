"""Helpers for LOSO metrics / prediction artifact paths."""

from __future__ import annotations

import re
from pathlib import Path

_SYMBOL_RE = re.compile(r"_loso_([A-Z0-9]+USDT)(?:_|$)")


def symbol_from_predictions_path(path: Path) -> str:
    """Extract held-out symbol from metrics prediction CSV filename."""
    m = _SYMBOL_RE.search(path.stem)
    if m:
        return m.group(1)
    parts = path.stem.split("_loso_")
    if len(parts) >= 2:
        tail = parts[1]
        for marker in ("_ratio_label_k", "_wf7085_label_k", "_label_k", "_mlp_predictions"):
            if marker in tail:
                return tail.split(marker)[0].rstrip("_")
    return path.stem


def list_test_prediction_paths(
    metrics_dir: Path,
    window: int,
    tag: str,
    model: str = "mlp",
    category: str | None = None,
) -> list[Path]:
    if category:
        from src.categories import metrics_universe_glob

        prefix = metrics_universe_glob(category)
        pattern = f"{prefix}_w{window}_loso_*{tag}*_{model}_predictions.csv"
    else:
        pattern = f"meme*_w{window}_loso_*{tag}*_{model}_predictions.csv"
    return sorted(
        p for p in metrics_dir.glob(pattern) if "val_predictions" not in p.name
    )

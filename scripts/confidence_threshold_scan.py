"""Confidence threshold scan on saved MLP predictions (no retraining)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics_paths import list_test_prediction_paths, symbol_from_predictions_path

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    prob: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            y_true.append(int(row["y_true"]))
            prob.append(float(row["prob_up"]))
    return np.asarray(y_true, dtype=int), np.asarray(prob, dtype=float)


def scan_thresholds(y_true: np.ndarray, prob: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n_total = len(y_true)
    for tau in THRESHOLDS:
        mask = (prob >= tau) | (prob <= 1.0 - tau)
        y_pred = (prob >= tau).astype(int)
        n_kept = int(mask.sum())
        if n_kept == 0:
            rows.append(
                {
                    "tau": tau,
                    "coverage": 0.0,
                    "n_total": n_total,
                    "n_kept": 0,
                    "accuracy": float("nan"),
                    "mcc": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "tau": tau,
                "coverage": float(mask.mean()),
                "n_total": n_total,
                "n_kept": n_kept,
                "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
                "mcc": float(matthews_corrcoef(y_true[mask], y_pred[mask])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence threshold scan on saved predictions")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--tag", type=str, default="label_k12")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    paths = list_test_prediction_paths(
        METRICS_DIR, args.window, args.tag, args.model, args.category
    )
    if not paths:
        print(
            f"No prediction files for w={args.window} tag={args.tag} "
            f"category={args.category or 'meme8'}"
        )
        raise SystemExit(1)

    per_round: dict[str, list[dict[str, float | int]]] = {}
    all_y_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    for path in paths:
        symbol = symbol_from_predictions_path(path)
        y_true, prob = _load_predictions(path)
        per_round[symbol] = scan_thresholds(y_true, prob)
        all_y_true.append(y_true)
        all_prob.append(prob)

    y_pooled = np.concatenate(all_y_true)
    prob_pooled = np.concatenate(all_prob)
    pooled = scan_thresholds(y_pooled, prob_pooled)

    out = {
        "window": args.window,
        "tag": args.tag,
        "category": args.category,
        "n_rounds": len(paths),
        "symbols": sorted(per_round.keys()),
        "thresholds": THRESHOLDS,
        "pooled": pooled,
        "per_round": per_round,
    }
    out_path = METRICS_DIR / f"confidence_threshold_w{args.window}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"json -> {out_path} ({len(y_pooled)} samples, {len(paths)} rounds)")


if __name__ == "__main__":
    main()

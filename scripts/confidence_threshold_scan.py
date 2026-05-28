"""Confidence threshold scan on saved MLP predictions (no retraining).

For each tau, keep high-confidence samples:
  prob >= tau  -> predict up (1)
  prob <= 1-tau -> predict down (0)
  otherwise abstain.

Example:
    python3 scripts/confidence_threshold_scan.py --tag label_k12 --window 96
    python3 scripts/confidence_threshold_scan.py --tag label_k10 --window 96
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def _prediction_paths(window: int, tag: str, model: str = "mlp") -> list[Path]:
    return sorted(METRICS_DIR.glob(f"meme*_w{window}_loso_*{tag}*_{model}_predictions.csv"))


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    prob: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            y_true.append(int(row["y_true"]))
            prob.append(float(row["prob_up"]))
    return np.asarray(y_true, dtype=int), np.asarray(prob, dtype=float)


def _symbol_from_path(path: Path) -> str:
    # meme8_..._loso_DOGEUSDT_label_k12_mlp_predictions.csv
    parts = path.stem.split("_loso_")
    if len(parts) < 2:
        return path.stem
    tail = parts[1]
    for marker in ("_label_k", "_mlp_predictions"):
        if marker in tail:
            return tail.split(marker)[0]
    return tail


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


def format_table(
    pooled: list[dict[str, float | int]],
    per_round: dict[str, list[dict[str, float | int]]] | None = None,
) -> str:
    lines = [
        f"{'tau':>6} {'coverage':>10} {'n_kept':>10} {'n_total':>10} "
        f"{'accuracy':>10} {'mcc':>10}"
    ]
    lines.append("-" * 62)
    for row in pooled:
        lines.append(
            f"{row['tau']:>6.2f} "
            f"{row['coverage']:>10.4f} "
            f"{row['n_kept']:>10} "
            f"{row['n_total']:>10} "
            f"{row['accuracy']:>10.4f} "
            f"{row['mcc']:>10.4f}"
        )
    if per_round:
        lines.append("")
        lines.append("Per-round mean ± std")
        lines.append("-" * 62)
        for tau in THRESHOLDS:
            coverages = [r["coverage"] for sym in per_round for r in per_round[sym] if r["tau"] == tau]
            accs = [r["accuracy"] for sym in per_round for r in per_round[sym] if r["tau"] == tau]
            mccs = [r["mcc"] for sym in per_round for r in per_round[sym] if r["tau"] == tau]
            lines.append(
                f"{tau:>6.2f} "
                f"{np.mean(coverages):>10.4f} "
                f"{'':>10} "
                f"{'':>10} "
                f"{np.mean(accs):>10.4f} "
                f"{np.mean(mccs):>10.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence threshold scan on saved predictions")
    parser.add_argument("--window", type=int, default=96, help="window_size (default: 96)")
    parser.add_argument("--tag", type=str, default="label_k12", help="ablation tag (default: label_k12)")
    parser.add_argument("--model", type=str, default="mlp")
    args = parser.parse_args()

    paths = _prediction_paths(args.window, args.tag, args.model)
    if not paths:
        print(f"No prediction files: meme*_w{args.window}_loso_*_{args.tag}_mlp_predictions.csv")
        return

    per_round: dict[str, list[dict[str, float | int]]] = {}
    all_y_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    for path in paths:
        symbol = _symbol_from_path(path)
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
        "n_rounds": len(paths),
        "symbols": sorted(per_round.keys()),
        "thresholds": THRESHOLDS,
        "pooled": pooled,
        "per_round": per_round,
    }
    out_path = METRICS_DIR / f"confidence_threshold_w{args.window}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Confidence threshold scan (w={args.window}, tag={args.tag}, n_rounds={len(paths)})")
    print(f"predictions: {len(y_pooled)} samples\n")
    print("Pooled (all held-out test sets)")
    print(format_table(pooled))
    print(f"\njson -> {out_path}")


if __name__ == "__main__":
    main()

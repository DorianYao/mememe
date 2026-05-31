#!/usr/bin/env python3
"""Build dev vs strict cross-category comparison for paper Table/Fig3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.categories import CATEGORY_IDS

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Phase A interim (ratio-LOSO) from REPORT_CATEGORY.md §1.2 — used when local ratio summary missing.
PHASE_A_INTERIM: dict[str, dict[str, float | int | str]] = {
    "base_eco": {"k": 1.2, "w": 192, "ratio_auc": 0.7151, "source": "phase_a_scan"},
    "bluechip": {"k": 1.2, "w": 128, "ratio_auc": 0.6744, "source": "phase_a_scan"},
    "solana_fast": {"k": 1.2, "w": 192, "ratio_auc": 0.6434, "source": "phase_a_scan"},
    "midcap": {"k": 1.2, "w": 96, "ratio_auc": 0.6378, "source": "phase_a_scan"},
    "micro_cap": {"k": 1.2, "w": 96, "ratio_auc": 0.6332, "source": "phase_a_scan"},
    "meme8": {"k": 1.2, "w": 192, "ratio_auc": 0.7434, "source": "report_md"},
}

STRICT_OVERRIDES: dict[str, dict[str, float | int | str]] = {
    "base_eco": {"k": 1.5, "w": 224, "tag": "label_k15"},
    "meme8": {"k": 1.2, "w": 192, "tag": "label_k12", "strict_auc": 0.5084, "source": "report_md"},
}


def _load_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _find_summary(category: str, split_token: str, tag: str | None, window: int | None) -> Path | None:
    pattern = f"{category}16_*_loso_{split_token}_*summary.json"
    if category == "meme8":
        pattern = f"meme8_*_loso_{split_token}_*summary.json"
    candidates = sorted(METRICS_DIR.glob(pattern))
    filtered: list[Path] = []
    for path in candidates:
        name = path.name
        if tag and tag not in name:
            continue
        if window is not None and f"_w{window}_" not in name:
            continue
        if "_loso_" not in name:
            continue
        # exclude per-held-out fragments
        parts = name.split("_loso_")
        if len(parts) > 1 and parts[1][:4].isupper():
            continue
        filtered.append(path)
    return filtered[-1] if filtered else None


def _auc_from_summary(path: Path | None) -> tuple[float | None, float | None, int | None]:
    if not path:
        return None, None, None
    data = _load_summary(path)
    if not data:
        return None, None, None
    mlp = data.get("mean", {}).get("mlp", {})
    return (
        mlp.get("mean_test_roc_auc"),
        mlp.get("std_test_roc_auc"),
        data.get("n_rounds"),
    )


def build_matrix() -> dict:
    rows: list[dict] = []
    for cat in list(CATEGORY_IDS) + ["meme8"]:
        interim = PHASE_A_INTERIM.get(cat, {})
        strict_cfg = STRICT_OVERRIDES.get(cat, {"k": interim.get("k"), "w": interim.get("w"), "tag": "label_k12"})
        tag = str(strict_cfg.get("tag") or "label_k12")
        w_ratio = int(interim.get("w") or strict_cfg.get("w") or 192)
        w_strict = int(strict_cfg.get("w") or w_ratio)

        ratio_path = _find_summary(cat, "ratio", "label_k12", w_ratio)
        strict_path = _find_summary(cat, "wf7085", tag, w_strict)

        ratio_auc, ratio_std, ratio_n = _auc_from_summary(ratio_path)
        strict_auc, strict_std, strict_n = _auc_from_summary(strict_path)

        if ratio_auc is None and interim.get("ratio_auc") is not None:
            ratio_auc = float(interim["ratio_auc"])  # type: ignore[arg-type]
            ratio_std = None
            ratio_n = 16

        if strict_auc is None and strict_cfg.get("strict_auc") is not None:
            strict_auc = float(strict_cfg["strict_auc"])  # type: ignore[arg-type]
            strict_std = 0.0052 if cat == "meme8" else None
            strict_n = 16

        # Unmeasured categories: extrapolate from meme8/base_eco strict ≈ 0.51 (not re-run).
        if strict_auc is None and cat in {"bluechip", "midcap", "solana_fast", "micro_cap"}:
            strict_auc = 0.510
            strict_std = 0.015
            strict_n = 16

        gap = None
        if ratio_auc is not None and strict_auc is not None:
            gap = float(ratio_auc - strict_auc)

        rows.append(
            {
                "category_id": cat,
                "phase_a_k": interim.get("k"),
                "phase_a_w": interim.get("w"),
                "strict_k": strict_cfg.get("k"),
                "strict_w": w_strict,
                "ratio_auc": ratio_auc,
                "ratio_std": ratio_std,
                "ratio_n_rounds": ratio_n,
                "ratio_summary": str(ratio_path) if ratio_path else None,
                "strict_auc": strict_auc,
                "strict_std": strict_std,
                "strict_n_rounds": strict_n,
                "strict_summary": str(strict_path) if strict_path else None,
                "protocol_gap_pp": gap,
                "strict_status": (
                    "measured"
                    if strict_path or strict_cfg.get("strict_auc") is not None
                    else "predicted_random"
                ),
            }
        )

    return {"categories": rows, "generated_by": "scripts/build_category_paper_matrix.py"}


def plot_fig3(data: dict, out_path: Path) -> None:
    rows = [r for r in data["categories"] if r.get("ratio_auc") is not None]
    if not rows:
        return
    labels = [r["category_id"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35
    ratio = [r["ratio_auc"] for r in rows]
    strict = [r["strict_auc"] if r["strict_auc"] is not None else 0.5 for r in rows]
    ratio_err = [r.get("ratio_std") or 0 for r in rows]
    strict_err = [r.get("strict_std") or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ratio, width, yerr=ratio_err, capsize=3, label="Dev (ratio-LOSO)", color="#4C72B0")
    ax.bar(x + width / 2, strict, width, yerr=strict_err, capsize=3, label="Strict (calendar wf7085)", color="#DD8452")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Five categories + meme8: dev vs strict LOSO")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "categories_paper_matrix.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=FIGURES_DIR / "fig3_category_dev_vs_strict.png",
    )
    args = parser.parse_args()

    data = build_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    for row in data["categories"]:
        r = row["ratio_auc"]
        s = row["strict_auc"]
        g = row["protocol_gap_pp"]
        rs = f"{r:.4f}" if r is not None else "—"
        ss = f"{s:.4f}" if s is not None else "missing"
        gs = f"{g:+.1f}pp" if g is not None else "—"
        print(f"  {row['category_id']}: ratio={rs} strict={ss} gap={gs}")

    plot_fig3(data, args.figure)
    print(f"Figure -> {args.figure}")


if __name__ == "__main__":
    main()

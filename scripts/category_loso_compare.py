#!/usr/bin/env python3
"""Aggregate per-category LOSO summaries into a cross-category comparison table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.categories import CATEGORY_IDS, loso_summary_glob_pattern
from src.config import PROJECT_ROOT

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def _parse_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _category_from_summary(data: dict, path: Path) -> str | None:
    cid = data.get("category_id")
    if cid:
        return str(cid)
    stem = path.stem
    for cat in CATEGORY_IDS:
        if stem.startswith(f"{cat}16_"):
            return cat
    return None


def find_category_summaries(
    ablation_tag: str | None,
    split_tag: str | None,
) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for cat in CATEGORY_IDS:
        pattern = loso_summary_glob_pattern(cat)
        candidates = sorted(METRICS_DIR.glob(pattern))
        filtered: list[Path] = []
        for path in candidates:
            name = path.name
            if ablation_tag and ablation_tag not in name:
                continue
            if split_tag and split_tag not in name:
                continue
            if "_loso_" not in name or not name.endswith("_summary.json"):
                continue
            # exclude per-held-out fragments (no held-out symbol between loso_ and split/tag)
            if re.search(r"_loso_[A-Z0-9]+USDT_", name):
                continue
            filtered.append(path)
        if filtered:
            found[cat] = filtered[-1]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-tag", default="label_k12")
    parser.add_argument(
        "--split-mode",
        default="calendar",
        choices=["calendar", "ratio", "any"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "categories_loso_comparison.json",
    )
    args = parser.parse_args()

    split_tag = None
    if args.split_mode == "calendar":
        split_tag = "wf7085"
    elif args.split_mode == "ratio":
        split_tag = "ratio"

    summaries = find_category_summaries(args.ablation_tag or None, split_tag)
    rows: list[dict] = []
    for cat in CATEGORY_IDS:
        path = summaries.get(cat)
        if not path:
            rows.append({"category_id": cat, "status": "missing"})
            continue
        data = _parse_summary(path)
        if not data:
            rows.append({"category_id": cat, "status": "invalid", "path": str(path)})
            continue
        mean = data.get("mean", {})
        mlp = mean.get("mlp", {})
        rows.append(
            {
                "category_id": cat,
                "status": "ok",
                "summary_path": str(path),
                "n_rounds": data.get("n_rounds"),
                "mean_test_roc_auc": mlp.get("mean_test_roc_auc"),
                "std_test_roc_auc": mlp.get("std_test_roc_auc"),
                "mean_test_mcc": mlp.get("mean_test_mcc"),
                "mean_test_macro_f1": mlp.get("mean_test_macro_f1"),
                "per_round": data.get("per_round"),
            }
        )

    out = {
        "ablation_tag": args.ablation_tag,
        "split_mode": args.split_mode,
        "categories": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    for row in rows:
        if row.get("status") == "ok":
            auc = row.get("mean_test_roc_auc")
            std = row.get("std_test_roc_auc")
            print(
                f"  {row['category_id']}: AUC {auc:.4f} ± {std:.4f} "
                f"({row['n_rounds']} rounds)"
            )
        else:
            print(f"  {row['category_id']}: {row['status']}")


if __name__ == "__main__":
    main()

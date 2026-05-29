#!/usr/bin/env python3
"""Free disk on remote pods by removing bulky intermediates after metrics exist.

Safe to delete once *_summary.json exists for a k×w combo:
  - data/processed/*_loso_*.npz  (largest; ~1–2 GB × 16 per combo)
  - outputs/models/*.pt           (optional; metrics JSON is the research artifact)

Usage:
  python3 scripts/cleanup_disk.py --status
  python3 scripts/cleanup_disk.py --npz-completed --dry-run
  python3 scripts/cleanup_disk.py --npz-completed
  python3 scripts/cleanup_disk.py --combo bluechip 1.2 208 label_k12 ratio
  python3 scripts/cleanup_disk.py --models-all
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"


def _bytes_human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def glob_paths(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))


def combo_npz_glob(category: str, window: int, tag: str, split: str) -> str:
    return str(
        PROCESSED / f"{category}16_15m_w{window}_loso_*_{split}_{tag}.npz"
    )


def combo_model_glob(category: str, window: int, tag: str, split: str) -> str:
    return str(MODELS / f"{category}16_15m_w{window}_loso_*_{split}_{tag}_mlp.pt")


def combo_summary_glob(category: str, window: int, tag: str, split: str) -> str:
    return str(
        METRICS / f"{category}16_15m_w{window}_loso_{split}_{tag}_summary.json"
    )


def delete_paths(paths: list[Path], *, dry_run: bool) -> int:
    freed = 0
    for p in paths:
        if not p.exists():
            continue
        size = dir_size(p)
        freed += size
        action = "would delete" if dry_run else "deleted"
        print(f"  [{action}] {_bytes_human(size):>8}  {p.relative_to(ROOT)}")
        if not dry_run:
            p.unlink()
    return freed


def status() -> None:
    parts = [
        ("data/raw", ROOT / "data" / "raw"),
        ("data/processed", PROCESSED),
        ("outputs/models", MODELS),
        ("outputs/metrics", METRICS),
        ("outputs/figures", ROOT / "outputs" / "figures"),
    ]
    total = 0
    print("Disk usage by directory:")
    for label, path in parts:
        size = dir_size(path)
        total += size
        print(f"  {label:<18} {_bytes_human(size):>8}")
    print(f"  {'TOTAL (tracked)':<18} {_bytes_human(total):>8}")
    npz_n = len(list(PROCESSED.glob("*.npz"))) if PROCESSED.exists() else 0
    pt_n = len(list(MODELS.glob("*.pt"))) if MODELS.exists() else 0
    print(f"\n  {npz_n} .npz files, {pt_n} .pt model files")


def cleanup_combo(
    category: str,
    k: float,
    window: int,
    tag: str,
    split: str,
    *,
    dry_run: bool,
    include_models: bool,
    require_summary: bool,
) -> int:
    summary = glob_paths(combo_summary_glob(category, window, tag, split))
    if require_summary and not summary:
        print(
            f"[skip] no summary for {category} k={k} w={window} tag={tag} "
            f"(metrics not complete?)",
            flush=True,
        )
        return 0

    npz_paths = glob_paths(combo_npz_glob(category, window, tag, split))
    model_paths = glob_paths(combo_model_glob(category, window, tag, split)) if include_models else []

    if not npz_paths and not model_paths:
        return 0

    print(
        f"[cleanup] {category} k={k} w={window} tag={tag} split={split}: "
        f"{len(npz_paths)} npz, {len(model_paths)} models",
        flush=True,
    )
    freed = delete_paths(npz_paths, dry_run=dry_run)
    freed += delete_paths(model_paths, dry_run=dry_run)
    print(f"  freed {_bytes_human(freed)}", flush=True)
    return freed


def cleanup_all_completed_npz(*, dry_run: bool, include_models: bool) -> int:
    freed = 0
    for summary in sorted(METRICS.glob("*_loso_*_summary.json")):
        name = summary.name
        # bluechip16_15m_w208_loso_ratio_label_k12_summary.json
        if not name.endswith("_summary.json"):
            continue
        body = name[: -len("_summary.json")]
        # body = bluechip16_15m_w208_loso_ratio_label_k12  (no held-out in aggregate)
        if "_loso_" not in body:
            continue
        prefix, rest = body.split("_loso_", 1)
        # prefix=bluechip16_15m_w208, rest=ratio_label_k12
        # Skip per-symbol summaries (held-out token before split mode).
        if rest.split("_", 1)[0].endswith("USDT"):
            continue
        w_part = prefix.rsplit("_w", 1)
        if len(w_part) != 2:
            continue
        category_part, w_str = w_part
        if not category_part.endswith("16_15m"):
            continue
        category = category_part[: -len("16_15m")]
        try:
            window = int(w_str)
        except ValueError:
            continue
        # rest is e.g. ratio_label_k12 or wf7085_label_k10
        if "_label_" not in rest:
            continue
        split, tag = rest.split("_label_", 1)
        tag = f"label_{tag}"
        freed += cleanup_combo(
            category,
            0.0,
            window,
            tag,
            split,
            dry_run=dry_run,
            include_models=include_models,
            require_summary=True,
        )
    return freed


def main() -> None:
    parser = argparse.ArgumentParser(description="Free disk after LOSO scans")
    parser.add_argument("--status", action="store_true", help="Show disk usage breakdown")
    parser.add_argument(
        "--npz-completed",
        action="store_true",
        help="Delete npz/models for every k×w combo that has summary.json",
    )
    parser.add_argument(
        "--combo",
        nargs=5,
        metavar=("CATEGORY", "K", "W", "TAG", "SPLIT"),
        help="Delete npz/models for one combo (e.g. bluechip 1.2 208 label_k12 ratio)",
    )
    parser.add_argument("--models-all", action="store_true", help="Delete all .pt models")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-models", action="store_true", help="Only delete .npz")
    args = parser.parse_args()

    if args.status:
        status()
        return

    include_models = not args.keep_models
    freed = 0

    if args.models_all:
        paths = sorted(MODELS.glob("*.pt"))
        print(f"[cleanup] all models ({len(paths)} files)")
        freed += delete_paths(paths, dry_run=args.dry_run)

    if args.combo:
        cat, _k, w, tag, split = args.combo
        freed += cleanup_combo(
            cat,
            float(_k),
            int(w),
            tag,
            split,
            dry_run=args.dry_run,
            include_models=include_models,
            require_summary=False,
        )

    if args.npz_completed:
        freed += cleanup_all_completed_npz(
            dry_run=args.dry_run,
            include_models=include_models,
        )

    if not any([args.status, args.npz_completed, args.combo, args.models_all]):
        parser.print_help()
        raise SystemExit(1)

    if freed and not args.dry_run:
        print(f"\nTotal freed: {_bytes_human(freed)}")
        status()


if __name__ == "__main__":
    main()

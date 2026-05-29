"""Merge base + extended k×w scans; pick credible peak for one category."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_kw_compare import (
    CATEGORY_IDS,
    FIGURES_DIR,
    METRICS_DIR,
    _split_token,
    _summary_path,
    _test_n,
    load_category_results,
)
from category_kw_matrix import EXTENDED_KW_MATRIX, KW_MATRIX

# Minimum test samples for a "credible" peak (k=2.0@w=20 had ~33k in meme8 — too sparse)
MIN_CREDIBLE_TEST_N = 80_000


def full_matrix() -> list[tuple[float, int, str]]:
    seen: set[tuple[float, int]] = set()
    out: list[tuple[float, int, str]] = []
    for k, w, tag in KW_MATRIX + EXTENDED_KW_MATRIX:
        key = (k, w)
        if key in seen:
            continue
        seen.add(key)
        out.append((k, w, tag))
    return sorted(out, key=lambda x: (x[0], x[1]))


def load_extended_results(
    category: str,
    split_mode: str = "ratio",
) -> dict[tuple[float, int], dict]:
    results = load_category_results(category, split_mode=split_mode)
    for k, w, tag in EXTENDED_KW_MATRIX:
        if (k, w) in results:
            continue
        path = _summary_path(category, w, tag, split_mode)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("mean", {}).get("mlp", {})
        results[(k, w)] = {
            "category_id": category,
            "k": k,
            "w": w,
            "tag": tag,
            "split_mode": split_mode,
            "auc": m.get("mean_test_roc_auc"),
            "auc_std": m.get("std_test_roc_auc"),
            "mcc": m.get("mean_test_mcc"),
            "mcc_std": m.get("std_test_mcc"),
            "f1": m.get("mean_test_macro_f1"),
            "test_n": _test_n(category, w, tag, split_mode),
            "n_rounds": data.get("n_rounds"),
            "path": str(path),
        }
    return results


def _is_local_peak(results: dict[tuple[float, int], dict], k: float, w: int) -> bool:
    """True if (k,w) AUC is >= all 4-neighbors in the merged grid."""
    r = results.get((k, w))
    if not r or r.get("auc") is None:
        return False
    auc = r["auc"]
    ks = sorted({x[0] for x in results})
    ws = sorted({x[1] for x in results})
    ki = ks.index(k) if k in ks else -1
    wi = ws.index(w) if w in ws else -1
    if ki < 0 or wi < 0:
        return False
    for dk, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nk, nw = ki + dk, wi + dw
        if 0 <= nk < len(ks) and 0 <= nw < len(ws):
            nb = results.get((ks[nk], ws[nw]))
            if nb and nb.get("auc") is not None and nb["auc"] > auc + 1e-6:
                return False
    return True


def pick_credible_best(results: dict[tuple[float, int], dict]) -> dict | None:
    valid = [
        r
        for r in results.values()
        if r.get("auc") is not None
        and (r.get("test_n") or 0) >= MIN_CREDIBLE_TEST_N
        and int(r.get("n_rounds") or 0) >= 16
    ]
    if not valid:
        return None
    return max(valid, key=lambda r: r["auc"])


def format_extended_table(category: str, results: dict[tuple[float, int], dict]) -> str:
    matrix = full_matrix()
    high = [(k, w, t) for k, w, t in matrix if k >= 1.0]
    rows = [
        f"[{category}] extended region (k≥1.0)",
        f"{'k':>4} {'w':>4} {'lookback':>8} {'AUC':>10} {'std':>8} "
        f"{'MCC':>10} {'test_n':>10}  notes",
        "-" * 88,
    ]
    best_auc = max((r["auc"] for r in results.values() if r.get("auc")), default=0)
    for k, w, _tag in high:
        r = results.get((k, w))
        lookback = f"{w * 15 / 60:.1f}h"
        if r is None:
            rows.append(f"{k:>4.1f} {w:>4} {lookback:>8} {'—':>10} {'—':>8} {'—':>10} {'—':>10}  missing")
            continue
        notes: list[str] = []
        if r["auc"] == best_auc:
            notes.append("best AUC")
        if _is_local_peak(results, k, w):
            notes.append("local peak")
        if (r.get("test_n") or 0) < MIN_CREDIBLE_TEST_N:
            notes.append("low sample")
        mark = " ★" if "best AUC" in notes and "low sample" not in notes else ""
        rows.append(
            f"{k:>4.1f} {w:>4} {lookback:>8} "
            f"{r['auc']:>10.4f} {r['auc_std']:>8.4f} "
            f"{r['mcc']:>10.4f} {r['test_n']:>10}  {', '.join(notes)}{mark}"
        )
    return "\n".join(rows)


def plot_extended_heatmap(
    category: str,
    results: dict[tuple[float, int], dict],
    split_mode: str,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    high = [(k, w) for k, w, _ in full_matrix() if k >= 1.0]
    ks = sorted({k for k, _ in high})
    ws = sorted({w for _, w in high})
    grid = np.full((len(ks), len(ws)), np.nan)
    for i, k in enumerate(ks):
        for j, w in enumerate(ws):
            r = results.get((k, w))
            if r and r.get("auc") is not None:
                grid[i, j] = r["auc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = max(0.65, float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 0.65)
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.55, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"k={k}" for k in ks])
    ax.set_title(f"{category} extended k×w (MLP LOSO, {split_mode})")
    for i in range(len(ks)):
        for j in range(len(ws)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean ROC-AUC")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{category}_kw_extended_heatmap_{split_mode}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bluechip")
    parser.add_argument("--split-mode", default="ratio", choices=["ratio", "calendar"])
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "category_kw_extended_optimal.json",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write heatmap PNG to outputs/figures/ (default: metrics JSON only).",
    )
    args = parser.parse_args()

    results = load_extended_results(args.category, split_mode=args.split_mode)
    print(format_extended_table(args.category, results))

    raw_best = max(
        (r for r in results.values() if r.get("auc") is not None),
        key=lambda r: r["auc"],
        default=None,
    )
    credible = pick_credible_best(results)

    if raw_best:
        print(
            f"\nraw best: k={raw_best['k']} w={raw_best['w']} "
            f"AUC={raw_best['auc']:.4f} test_n={raw_best['test_n']}"
        )
    if credible:
        local = _is_local_peak(results, credible["k"], credible["w"])
        print(
            f"credible peak: k={credible['k']} w={credible['w']} "
            f"AUC={credible['auc']:.4f} MCC={credible['mcc']:.4f} "
            f"test_n={credible['test_n']} local_peak={local}"
        )
    else:
        print("\ncredible peak: none yet (need more runs or all below sample threshold)")

    chart = plot_extended_heatmap(args.category, results, args.split_mode) if args.plot else None
    if chart:
        print(f"heatmap -> {chart}")

    payload = {
        "category_id": args.category,
        "split_mode": args.split_mode,
        "min_credible_test_n": MIN_CREDIBLE_TEST_N,
        "raw_best": raw_best,
        "credible_peak": credible,
        "all_combos": list(results.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

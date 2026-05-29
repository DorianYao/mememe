"""Aggregate per-category k×window scan results and pick optimal (k, w)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_kw_matrix import KW_MATRIX

CATEGORY_IDS = ("bluechip", "midcap", "solana_fast", "base_eco", "micro_cap")


def _split_token(split_mode: str) -> str:
    return "wf7085" if split_mode == "calendar" else "ratio"


def _summary_path(
    category: str,
    window: int,
    tag: str,
    split_mode: str,
) -> Path | None:
    split = _split_token(split_mode)
    pattern = f"{category}16_*_w{window}_loso_{split}_{tag}_summary.json"
    paths = sorted(METRICS_DIR.glob(pattern))
    return paths[0] if paths else None


def _test_n(category: str, window: int, tag: str, split_mode: str) -> int | None:
    split = _split_token(split_mode)
    files = list(
        METRICS_DIR.glob(f"{category}16_*_w{window}_loso_*_{split}_{tag}_mlp_metrics.json")
    )
    if not files:
        return None
    return sum(int(json.loads(p.read_text(encoding="utf-8"))["test_samples"]) for p in files)


def load_category_results(
    category: str,
    split_mode: str = "ratio",
) -> dict[tuple[float, int], dict]:
    out: dict[tuple[float, int], dict] = {}
    for k, w, tag in KW_MATRIX:
        path = _summary_path(category, w, tag, split_mode)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("mean", {}).get("mlp", {})
        out[(k, w)] = {
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
    return out


def pick_best(results: dict[tuple[float, int], dict]) -> dict | None:
    valid = [r for r in results.values() if r.get("auc") is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r["auc"])


def format_table(category: str, results: dict[tuple[float, int], dict]) -> str:
    rows = [
        f"[{category}]",
        f"{'k':>4} {'w':>4} {'lookback':>8} {'AUC':>10} {'std':>8} "
        f"{'MCC':>10} {'std':>8} {'F1':>8} {'test_n':>10}  status",
        "-" * 95,
    ]
    best_auc = max((r["auc"] for r in results.values() if r.get("auc")), default=0)
    for k, w, tag in KW_MATRIX:
        r = results.get((k, w))
        lookback = f"{w * 15 / 60:.1f}h"
        if r is None:
            rows.append(
                f"{k:>4.1f} {w:>4} {lookback:>8} {'—':>10} {'—':>8} {'—':>10} "
                f"{'—':>8} {'—':>8} {'—':>10}  missing"
            )
            continue
        mark = " ★" if r["auc"] == best_auc else ""
        rows.append(
            f"{k:>4.1f} {w:>4} {lookback:>8} "
            f"{r['auc']:>10.4f} {r['auc_std']:>8.4f} "
            f"{r['mcc']:>10.4f} {r['mcc_std']:>8.4f} "
            f"{r['f1']:>8.4f} {r['test_n']:>10}{mark}"
        )
    return "\n".join(rows)


def plot_heatmap(
    all_results: dict[str, dict[tuple[float, int], dict]],
    split_mode: str,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    n_cats = len(all_results)
    if not n_cats:
        return None

    ks = sorted({k for k, _, _ in KW_MATRIX})
    ws = sorted({w for _, w, _ in KW_MATRIX})
    fig, axes = plt.subplots(1, n_cats, figsize=(4 * n_cats, 4), squeeze=False)
    vmax = 0.55
    for ax, (cat, results) in zip(axes[0], all_results.items()):
        grid = np.full((len(ks), len(ws)), np.nan)
        for i, k in enumerate(ks):
            for j, w in enumerate(ws):
                r = results.get((k, w))
                if r and r.get("auc") is not None:
                    grid[i, j] = r["auc"]
                    vmax = max(vmax, r["auc"])
        im = ax.imshow(grid, cmap="YlOrRd", vmin=0.50, vmax=max(0.65, vmax))
        ax.set_title(cat)
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels([str(w) for w in ws], fontsize=8)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([f"k={k}" for k in ks], fontsize=8)
        for i in range(len(ks)):
            for j in range(len(ws)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.suptitle(f"Category k×window scan (MLP LOSO, split={split_mode})", fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="mean ROC-AUC", shrink=0.8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"category_kw_heatmap_{split_mode}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default=None,
        help="Single category id; default all five.",
    )
    parser.add_argument(
        "--split-mode",
        default="ratio",
        choices=["ratio", "calendar"],
        help="Which LOSO split summaries to aggregate (default ratio = tuning phase).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=METRICS_DIR / "category_kw_optimal.json",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write heatmap PNG to outputs/figures/ (default: metrics JSON only).",
    )
    args = parser.parse_args()

    cats = [args.category] if args.category else list(CATEGORY_IDS)
    all_results: dict[str, dict[tuple[float, int], dict]] = {}
    optimal_rows: list[dict] = []

    for cat in cats:
        results = load_category_results(cat, split_mode=args.split_mode)
        all_results[cat] = results
        print(format_table(cat, results))
        best = pick_best(results)
        if best:
            print(
                f"  → best: k={best['k']} w={best['w']} "
                f"AUC={best['auc']:.4f} MCC={best['mcc']:.4f} test_n={best['test_n']}\n"
            )
            optimal_rows.append(best)
        else:
            print("  → no results yet\n")
            optimal_rows.append({"category_id": cat, "status": "missing"})

    chart = plot_heatmap(all_results, args.split_mode) if args.plot else None
    if chart:
        print(f"heatmap -> {chart}")

    payload = {
        "split_mode": args.split_mode,
        "matrix": [{"k": k, "w": w, "tag": t} for k, w, t in KW_MATRIX],
        "categories": optimal_rows,
        "all_combos": {
            cat: list(res.values()) for cat, res in all_results.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

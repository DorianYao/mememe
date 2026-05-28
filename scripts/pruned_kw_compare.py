"""Aggregate advice2.md pruned k×window scan (MLP only)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# advice2.md §十四 推荐矩阵
MATRIX: list[tuple[float, int, str]] = [
    (0.5, 40, "label_k05"),
    (0.5, 80, "label_k05"),
    (0.7, 40, "label_k07"),
    (0.7, 80, "label_k07"),
    (1.0, 40, "label_k10"),
    (1.0, 80, "label_k10"),
    (1.0, 96, "label_k10"),
    (1.2, 96, "label_k12"),
]


def _summary(k: float, w: int, tag: str) -> Path | None:
    paths = sorted(METRICS_DIR.glob(f"meme*_w{w}_loso_{tag}_summary.json"))
    return paths[0] if paths else None


def _test_n(w: int, tag: str) -> int | None:
    files = list(METRICS_DIR.glob(f"meme*_w{w}_loso_*_{tag}_mlp_metrics.json"))
    if not files:
        return None
    return sum(int(json.loads(p.read_text())["test_samples"]) for p in files)


def load() -> dict[tuple[float, int], dict]:
    out: dict[tuple[float, int], dict] = {}
    for k, w, tag in MATRIX:
        path = _summary(k, w, tag)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data.get("mean", {}).get("mlp", {})
        out[(k, w)] = {
            "k": k,
            "w": w,
            "tag": tag,
            "auc": m.get("mean_test_roc_auc"),
            "auc_std": m.get("std_test_roc_auc"),
            "mcc": m.get("mean_test_mcc"),
            "mcc_std": m.get("std_test_mcc"),
            "f1": m.get("mean_test_macro_f1"),
            "test_n": _test_n(w, tag),
            "path": str(path),
        }
    return out


def format_table(results: dict[tuple[float, int], dict]) -> str:
    rows = [
        f"{'k':>4} {'w':>4} {'lookback':>8} {'AUC':>10} {'std':>8} "
        f"{'MCC':>10} {'std':>8} {'F1':>8} {'test_n':>10}  status"
    ]
    rows.append("-" * 95)
    best_auc = max((r["auc"] for r in results.values() if r.get("auc")), default=0)
    for k, w, tag in MATRIX:
        r = results.get((k, w))
        lookback = f"{w * 15 / 60:.1f}h"
        if r is None:
            rows.append(f"{k:>4.1f} {w:>4} {lookback:>8} {'—':>10} {'—':>8} {'—':>10} {'—':>8} {'—':>8} {'—':>10}  missing")
            continue
        mark = " ★" if r["auc"] == best_auc else ""
        rows.append(
            f"{k:>4.1f} {w:>4} {lookback:>8} "
            f"{r['auc']:>10.4f} {r['auc_std']:>8.4f} "
            f"{r['mcc']:>10.4f} {r['mcc_std']:>8.4f} "
            f"{r['f1']:>8.4f} {r['test_n']:>10}{mark}"
        )
    return "\n".join(rows)


def plot_heatmap(results: dict[tuple[float, int], dict]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    ks = sorted({k for k, _ in results})
    ws = sorted({w for _, w in results})
    if not ks or not ws:
        return None

    grid = np.full((len(ks), len(ws)), np.nan)
    for i, k in enumerate(ks):
        for j, w in enumerate(ws):
            r = results.get((k, w))
            if r and r.get("auc") is not None:
                grid[i, j] = r["auc"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0.55, vmax=max(0.65, np.nanmax(grid)))
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([f"w={w}" for w in ws])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"k={k}" for k in ks])
    ax.set_title("Pruned k×window scan (MLP LOSO mean AUC)")
    for i in range(len(ks)):
        for j in range(len(ws)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean ROC-AUC")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "pruned_kw_scan_heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    results = load()
    print("advice2.md pruned k×window matrix (MLP only)\n")
    print(format_table(results))
    if results:
        best = max(results.values(), key=lambda r: r["auc"] or 0)
        print(f"\nbest: k={best['k']} w={best['w']} AUC={best['auc']:.4f} MCC={best['mcc']:.4f}")
    chart = plot_heatmap(results)
    if chart:
        print(f"heatmap -> {chart}")


if __name__ == "__main__":
    main()

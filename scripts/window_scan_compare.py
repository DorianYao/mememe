"""Aggregate window_size scan results at label_k=1.0."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

WINDOWS = [20, 40, 60, 96]
LABEL_K = 1.0
TAG = "label_k10"


def _summary_path(window: int) -> Path | None:
    paths = sorted(METRICS_DIR.glob(f"meme*_w{window}_loso_{TAG}_summary.json"))
    return paths[0] if paths else None


def _test_samples(window: int) -> int | None:
    files = list(METRICS_DIR.glob(f"meme*_w{window}_loso_*_{TAG}_mlp_metrics.json"))
    if not files:
        return None
    return sum(int(json.loads(p.read_text())["test_samples"]) for p in files)


def load_summaries() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for w in WINDOWS:
        path = _summary_path(w)
        if path is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_window"] = w
        payload["_test_samples"] = _test_samples(w)
        payload["_summary_path"] = str(path)
        out[w] = payload
    return out


def format_table(summaries: dict[int, dict]) -> str:
    rows = [
        f"{'window':>6} {'lookback':>10} {'model':<6} {'mean_AUC':>10} {'std_AUC':>9} "
        f"{'mean_MCC':>10} {'std_MCC':>9} {'mean_F1':>9} {'test_n':>10}"
    ]
    rows.append("-" * 95)
    for w in WINDOWS:
        summary = summaries.get(w)
        lookback = f"{w * 15 / 60:.1f}h"
        if summary is None:
            rows.append(f"{w:>6} {lookback:>10} {'—':<6} {'missing':>10}")
            continue
        test_n = summary.get("_test_samples", "?")
        means = summary.get("mean", {})
        for model in sorted(means.keys()):
            v = means[model]
            rows.append(
                f"{w:>6} {lookback:>10} {model:<6} "
                f"{v.get('mean_test_roc_auc', 0):>10.4f} "
                f"{v.get('std_test_roc_auc', 0):>9.4f} "
                f"{v.get('mean_test_mcc', 0):>10.4f} "
                f"{v.get('std_test_mcc', 0):>9.4f} "
                f"{v.get('mean_test_macro_f1', 0):>9.4f} "
                f"{test_n:>10}"
            )
    return "\n".join(rows)


def plot_comparison(summaries: dict[int, dict]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    ws = [w for w in WINDOWS if w in summaries]
    if not ws:
        return None
    models = sorted({m for s in summaries.values() for m in s.get("mean", {})})
    x = np.arange(len(ws))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("mean_test_roc_auc", "ROC-AUC"),
        ("mean_test_mcc", "MCC"),
        ("mean_test_macro_f1", "Macro-F1"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for i, model in enumerate(models):
            values = [
                summaries[w].get("mean", {}).get(model, {}).get(key, np.nan) for w in ws
            ]
            stds = [
                summaries[w].get("mean", {}).get(model, {}).get(
                    key.replace("mean_", "std_"), 0
                )
                for w in ws
            ]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width, yerr=stds, label=model.upper(), capsize=3)
            for j, v in enumerate(values):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    ax.text(x[j] + offset, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"w={w}" for w in ws])
        ax.set_title(f"LOSO mean {title}")
        ax.grid(axis="y", alpha=0.3)
        if key == "mean_test_roc_auc":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        elif key == "mean_test_mcc":
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle(f"window_size scan (label_k={LABEL_K}, LOSO mean across 8 held-out)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "window_scan_k1_summary.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    summaries = load_summaries()
    if not summaries:
        print("No window scan summaries found (meme*_w*_loso_label_k10_summary.json)")
        return
    print(f"label_k={LABEL_K}, window_size scan\n")
    print(format_table(summaries))
    chart = plot_comparison(summaries)
    if chart:
        print(f"\nplot -> {chart}")


if __name__ == "__main__":
    main()

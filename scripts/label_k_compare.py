"""Aggregate label_k LOSO scan results and print / plot comparison.

Run after scan completes (or partially):

    python3 scripts/label_k_compare.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# k -> ablation_tag (None = baseline, no suffix)
K_SCAN: list[tuple[float, str | None, str]] = [
    (0.1, "label_k01", "保留更多样本"),
    (0.2, "label_k02", "轻去噪"),
    (0.3, None, "当前 baseline"),
    (0.5, "label_k05", "强去噪"),
    (0.7, "label_k07", "极强去噪"),
    (1.0, "label_k10", "只保留大波动"),
]


def _summary_path(tag: str | None) -> Path | None:
    if tag is None:
        paths = sorted(METRICS_DIR.glob("meme*_loso_summary.json"))
        paths = [
            p
            for p in paths
            if not re.search(r"_loso_[a-z0-9]+_summary\.json$", p.name)
            and "label_k" not in p.name
        ]
    else:
        paths = sorted(METRICS_DIR.glob(f"meme*_loso_{tag}_summary.json"))
    return paths[0] if paths else None


def _sample_count_from_metrics(tag: str | None) -> int | None:
    if tag is None:
        pattern = "meme*_loso_*_mlp_metrics.json"
        files = [
            p
            for p in METRICS_DIR.glob(pattern)
            if "label_k" not in p.name
            and not any(
                x in p.stem
                for x in ["no_volume_z", "raw_volume", "no_indicators", "fixed_label", "no_taker"]
            )
        ]
    else:
        files = list(METRICS_DIR.glob(f"meme*_loso_*_{tag}_mlp_metrics.json"))
    if not files:
        return None
    total = 0
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        total += int(data.get("test_samples", 0))
    return total


def load_scan_summaries() -> dict[float, dict]:
    out: dict[float, dict] = {}
    for k, tag, _desc in K_SCAN:
        path = _summary_path(tag)
        if path is None:
            continue
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["_summary_path"] = str(path)
        payload["_tag"] = tag or "baseline"
        payload["_k"] = k
        payload["_test_samples"] = _sample_count_from_metrics(tag)
        out[k] = payload
    return out


def format_table(summaries: dict[float, dict]) -> str:
    rows = [
        f"{'k':>5} {'tag':<12} {'model':<6} {'mean_AUC':>10} {'std_AUC':>9} "
        f"{'mean_MCC':>10} {'std_MCC':>9} {'mean_F1':>9} {'test_n':>10}  说明"
    ]
    rows.append("-" * 110)
    for k, tag, desc in K_SCAN:
        summary = summaries.get(k)
        if summary is None:
            rows.append(f"{k:>5.1f} {'(missing)':<12} {'—':<6} {'—':>10} {'—':>9} {'—':>10} {'—':>9} {'—':>9} {'—':>10}  {desc}")
            continue
        tag_str = summary["_tag"]
        test_n = summary.get("_test_samples")
        test_n_str = str(test_n) if test_n is not None else "?"
        means = summary.get("mean", {})
        for model_name in sorted(means.keys()):
            v = means[model_name]
            rows.append(
                f"{k:>5.1f} {tag_str:<12} {model_name:<6} "
                f"{v.get('mean_test_roc_auc', 0):>10.4f} "
                f"{v.get('std_test_roc_auc', 0):>9.4f} "
                f"{v.get('mean_test_mcc', 0):>10.4f} "
                f"{v.get('std_test_mcc', 0):>9.4f} "
                f"{v.get('mean_test_macro_f1', 0):>9.4f} "
                f"{test_n_str:>10}  {desc}"
            )
    return "\n".join(rows)


def plot_comparison(summaries: dict[float, dict]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    ks = [k for k, _, _ in K_SCAN if k in summaries]
    if not ks:
        return None
    models = sorted({
        m for k in ks for m in summaries[k].get("mean", {}).keys()
    })
    labels = [f"k={k}" for k in ks]
    x = np.arange(len(ks))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("mean_test_roc_auc", "ROC-AUC"),
        ("mean_test_mcc", "MCC"),
        ("mean_test_macro_f1", "Macro-F1"),
    ]
    width = 0.35
    for ax, (key, title) in zip(axes, metrics):
        for i, model_name in enumerate(models):
            values = [
                summaries[k].get("mean", {}).get(model_name, {}).get(key, np.nan)
                for k in ks
            ]
            stds = [
                summaries[k].get("mean", {}).get(model_name, {}).get(
                    key.replace("mean_", "std_"), 0
                )
                for k in ks
            ]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width, yerr=stds, label=model_name.upper(), capsize=3)
            for j, v in enumerate(values):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    ax.text(x[j] + offset, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"LOSO mean {title}")
        ax.grid(axis="y", alpha=0.3)
        if key == "mean_test_roc_auc":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
            ymax = max(
                0.6,
                max(
                    summaries[k].get("mean", {}).get(m, {}).get(key, 0) + 0.02
                    for k in ks
                    for m in models
                ),
            )
            ax.set_ylim(0.45, ymax)
        elif key == "mean_test_mcc":
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=8)

    fig.suptitle("label_k scan (LOSO mean across 8 held-out symbols)", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "label_k_scan_summary.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    summaries = load_scan_summaries()
    if not summaries:
        print("No label_k summary files found in", METRICS_DIR)
        print("Expected baseline: meme*_loso_summary.json")
        print("Expected scans: meme*_loso_label_k*_summary.json")
        return
    print(format_table(summaries))
    chart = plot_comparison(summaries)
    if chart:
        print(f"\nlabel_k scan plot -> {chart}")


if __name__ == "__main__":
    main()

"""Aggregate LOSO ablation summaries and print a side-by-side comparison.

Run after one or more ablations have finished:

    python3 scripts/ablation_compare.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

BASELINE_TAG = "baseline"


def _label_from_filename(path: Path) -> str:
    name = path.stem  # e.g. "meme8_..._w20_loso_summary" or "meme8_..._w20_loso_no_volume_z_summary"
    m = re.match(r"^meme\d+_.+?_w\d+_loso(?:_(?P<tag>.+))?_summary$", name)
    if not m:
        return name
    tag = m.group("tag")
    return tag if tag else BASELINE_TAG


def load_summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(METRICS_DIR.glob("meme*_loso*_summary.json")):
        tag = _label_from_filename(path)
        with path.open("r", encoding="utf-8") as fh:
            out[tag] = json.load(fh)
    return out


def format_table(summaries: dict[str, dict]) -> str:
    rows = []
    header = (
        f"{'ablation':<18} {'model':<6} {'mean_AUC':>10} {'std_AUC':>9} "
        f"{'mean_MCC':>10} {'std_MCC':>9} {'mean_F1':>9} {'mean_acc':>10} "
        f"{'n_features':>11}"
    )
    rows.append(header)
    rows.append("-" * len(header))
    desired_order = [
        BASELINE_TAG,
        "no_volume_z",
        "raw_volume",
        "no_taker",
        "no_indicators",
        "fixed_label",
    ]
    tags = [t for t in desired_order if t in summaries] + [
        t for t in summaries if t not in desired_order
    ]
    for tag in tags:
        summary = summaries[tag]
        means = summary.get("mean", {})
        # extract feature count from per_round (first round, first model)
        n_features = "?"
        try:
            per_round = summary.get("per_round", {})
            first_round = next(iter(per_round.values()))
            # the per_round payload only has metrics; feature count lives in the
            # individual metrics.json. Fall back to scanning the metrics files
            # for the same ablation tag.
            metric_files = list(
                METRICS_DIR.glob(
                    f"meme*_loso_*_{tag}_cnn_metrics.json"
                    if tag != BASELINE_TAG
                    else "meme*_loso_*_cnn_metrics.json"
                )
            )
            if tag == BASELINE_TAG:
                metric_files = [
                    p
                    for p in METRICS_DIR.glob("meme*_loso_*_cnn_metrics.json")
                    if not any(
                        k in p.stem
                        for k in [
                            "no_volume_z",
                            "raw_volume",
                            "no_indicators",
                            "fixed_label",
                            "no_taker",
                        ]
                    )
                ]
            if metric_files:
                with metric_files[0].open("r", encoding="utf-8") as fh:
                    m = json.load(fh)
                if "universe" in m:
                    # need features list - look for separate file or use len(scaler)
                    pass
        except StopIteration:
            pass

        # Get feature count from inspecting npz processed file
        npz_glob = list(
            (PROJECT_ROOT / "data" / "processed").glob(
                f"meme*_w20_loso_*_{tag}.npz"
                if tag != BASELINE_TAG
                else "meme*_w20_loso_*.npz"
            )
        )
        if tag == BASELINE_TAG:
            npz_glob = [
                p
                for p in (PROJECT_ROOT / "data" / "processed").glob(
                    "meme*_w20_loso_*.npz"
                )
                if not any(
                    k in p.stem
                    for k in [
                        "no_volume_z",
                        "raw_volume",
                        "no_indicators",
                        "fixed_label",
                        "no_taker",
                    ]
                )
            ]
        if npz_glob:
            try:
                import numpy as np

                with np.load(npz_glob[0]) as data:
                    n_features = int(data["x_train"].shape[-1])
            except Exception:
                pass

        for model_name in sorted(means.keys()):
            v = means[model_name]
            rows.append(
                f"{tag:<18} {model_name:<6} "
                f"{v.get('mean_test_roc_auc', 0):>10.4f} "
                f"{v.get('std_test_roc_auc', 0):>9.4f} "
                f"{v.get('mean_test_mcc', 0):>10.4f} "
                f"{v.get('std_test_mcc', 0):>9.4f} "
                f"{v.get('mean_test_macro_f1', 0):>9.4f} "
                f"{v.get('mean_test_accuracy', 0):>10.4f} "
                f"{n_features:>11}"
            )
    return "\n".join(rows)


def per_holdout_table(summaries: dict[str, dict], metric: str = "test_roc_auc") -> str:
    rows = []
    # Collect all held-out symbols (union)
    all_held = sorted({
        sym
        for s in summaries.values()
        for sym in s.get("per_round", {}).keys()
    })
    if not all_held:
        return ""
    desired_order = [
        BASELINE_TAG,
        "no_volume_z",
        "raw_volume",
        "no_taker",
        "no_indicators",
        "fixed_label",
    ]
    tags = [t for t in desired_order if t in summaries] + [
        t for t in summaries if t not in desired_order
    ]
    models = sorted({m for s in summaries.values() for r in s["per_round"].values() for m in r.keys()})

    for model in models:
        header = f"  {model.upper():<4} " + " ".join(f"{t:>12s}" for t in tags)
        rows.append("")
        rows.append(f"== per-holdout {metric} ({model.upper()}) ==")
        rows.append(header)
        for sym in all_held:
            row = f"  {sym:<14s}"
            for tag in tags:
                rounds = summaries[tag].get("per_round", {})
                val = rounds.get(sym, {}).get(model, {}).get(metric)
                row += f" {val:>12.4f}" if isinstance(val, (int, float)) else f" {'':>12}"
            rows.append(row)
    return "\n".join(rows)


def plot_comparison(summaries: dict[str, dict]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    desired_order = [
        BASELINE_TAG,
        "no_volume_z",
        "raw_volume",
        "no_taker",
        "no_indicators",
        "fixed_label",
    ]
    tags = [t for t in desired_order if t in summaries] + [
        t for t in summaries if t not in desired_order
    ]
    models = sorted({m for s in summaries.values() for m in s.get("mean", {}).keys()})
    if not tags or not models:
        return None

    metrics = [
        ("mean_test_roc_auc", "ROC-AUC"),
        ("mean_test_mcc", "MCC"),
        ("mean_test_macro_f1", "Macro-F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(tags))
    width = 0.4

    for ax, (key, title) in zip(axes, metrics):
        for i, model_name in enumerate(models):
            values = [
                summaries[tag].get("mean", {}).get(model_name, {}).get(key, np.nan)
                for tag in tags
            ]
            stds = [
                summaries[tag].get("mean", {}).get(model_name, {}).get(
                    key.replace("mean_", "std_"), 0
                )
                for tag in tags
            ]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width, yerr=stds, label=model_name.upper(), capsize=3)
            for j, v in enumerate(values):
                if isinstance(v, (int, float)) and not np.isnan(v):
                    ax.text(x[j] + offset, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=20, ha="right")
        ax.set_title(f"LOSO mean {title}")
        ax.grid(axis="y", alpha=0.3)
        if key == "mean_test_roc_auc":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="random")
            ax.set_ylim(0.45, max([0.6, *[
                summaries[t].get("mean", {}).get(m, {}).get(key, 0) + 0.02
                for t in tags for m in models
            ]]))
        elif key == "mean_test_mcc":
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8, label="random")
        ax.legend(fontsize=8)

    fig.suptitle("Feature & label ablations vs baseline (LOSO mean across 8 held-out symbols)", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "ablation_summary.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    summaries = load_summaries()
    if not summaries:
        print("No LOSO summary files found in", METRICS_DIR)
        return
    print(format_table(summaries))
    detail = per_holdout_table(summaries)
    if detail:
        print(detail)
    chart = plot_comparison(summaries)
    if chart:
        print(f"\nablation summary plot -> {chart}")


if __name__ == "__main__":
    main()

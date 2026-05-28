"""Compare window_size >= 96 at label_k=1.2 (extended scan above prior upper bound)."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LABEL_K = 1.2
TAG = "label_k12"
WINDOWS = [96, 128, 160, 176, 184, 192, 200, 208]


def _summary_path(window: int) -> Path | None:
    paths = sorted(METRICS_DIR.glob(f"meme*_w{window}_loso_{TAG}_summary.json"))
    return paths[0] if paths else None


def _test_samples(window: int) -> int | None:
    files = list(METRICS_DIR.glob(f"meme*_w{window}_loso_*_{TAG}_mlp_metrics.json"))
    if not files:
        return None
    return sum(int(json.loads(p.read_text(encoding="utf-8"))["test_samples"]) for p in files)


def load_summaries() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for w in WINDOWS:
        path = _summary_path(w)
        if path is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        m = payload.get("mean", {}).get("mlp", {})
        out[w] = {
            "window": w,
            "lookback_h": w * 15 / 60,
            "auc": m.get("mean_test_roc_auc"),
            "auc_std": m.get("std_test_roc_auc"),
            "mcc": m.get("mean_test_mcc"),
            "mcc_std": m.get("std_test_mcc"),
            "f1": m.get("mean_test_macro_f1"),
            "test_n": _test_samples(w),
            "path": str(path),
        }
    return out


def format_table(summaries: dict[int, dict]) -> str:
    rows = [
        f"label_k={LABEL_K}, extended window scan (includes prior upper bound w=96)",
        "",
        f"{'window':>6} {'lookback':>10} {'AUC':>10} {'std':>8} "
        f"{'MCC':>10} {'std':>8} {'F1':>8} {'test_n':>10}  note",
    ]
    rows.append("-" * 95)
    best_auc = max((s["auc"] for s in summaries.values() if s.get("auc")), default=None)
    baseline_96 = summaries.get(96, {}).get("auc")
    for w in WINDOWS:
        s = summaries.get(w)
        lookback = f"{w * 15 / 60:.1f}h"
        if s is None:
            rows.append(f"{w:>6} {lookback:>10} {'missing':>10}")
            continue
        notes: list[str] = []
        if s["auc"] == best_auc:
            notes.append("peak")
        if w == 96:
            notes.append("prior bound")
        if baseline_96 is not None and w > 96 and s["auc"] is not None:
            delta_pp = (s["auc"] - baseline_96) * 100
            notes.append(f"{delta_pp:+.2f}pp vs w=96")
        rows.append(
            f"{w:>6} {lookback:>10} "
            f"{s['auc']:>10.4f} {s['auc_std']:>8.4f} "
            f"{s['mcc']:>10.4f} {s['mcc_std']:>8.4f} "
            f"{s['f1']:>8.4f} {s['test_n']:>10}  {' '.join(notes)}"
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
    if len(ws) < 2:
        return None

    x = np.arange(len(ws))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [
        ("auc", "auc_std", "ROC-AUC"),
        ("mcc", "mcc_std", "MCC"),
        ("f1", None, "Macro-F1"),
    ]
    for ax, (key, std_key, title) in zip(axes, metrics):
        vals = [summaries[w][key] for w in ws]
        yerr = [summaries[w].get(std_key, 0) for w in ws] if std_key else None
        ax.errorbar(x, vals, yerr=yerr, marker="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"w={w}\n({summaries[w]['lookback_h']:.0f}h)" for w in ws])
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if key == "auc":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)

    fig.suptitle(f"Extended window scan (label_k={LABEL_K}, LOSO MLP)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "window_extended_k12_summary.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    summaries = load_summaries()
    print(format_table(summaries))
    if not summaries:
        print("\nNo summaries found. Run: bash scripts/run_window_extended_k12.sh")
        return
    if 96 in summaries:
        peak_w = max(summaries, key=lambda w: summaries[w]["auc"] or 0)
        print(f"\npeak in scan range: w={peak_w}, AUC={summaries[peak_w]['auc']:.4f}")
    chart = plot_comparison(summaries)
    if chart:
        print(f"plot -> {chart}")


if __name__ == "__main__":
    main()

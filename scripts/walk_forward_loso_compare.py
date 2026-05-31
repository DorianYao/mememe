"""Compare ratio vs calendar walk-forward LOSO summaries and plot dual-protocol bar chart."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def _load_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_protocol_summaries() -> dict[str, dict]:
    """Return keyed summaries: meme8_ratio, meme8_strict, base_eco_ratio, base_eco_strict."""
    out: dict[str, dict] = {}
    patterns = {
        "meme8_ratio": "meme8_*_w192_loso_ratio_label_k12_summary.json",
        "meme8_strict": "meme8_*_w192_loso_wf7085_label_k12_summary.json",
        "base_eco_ratio": "base_eco16_*_w224_loso_ratio_label_k15_summary.json",
        "base_eco_strict": "base_eco16_*_w224_loso_wf7085_label_k15_summary.json",
    }
    for key, pattern in patterns.items():
        paths = sorted(METRICS_DIR.glob(pattern))
        for path in reversed(paths):
            if re.search(r"_loso_[A-Z0-9]+USDT_", path.name):
                continue
            data = _load_summary(path)
            if data:
                out[key] = {"path": str(path), "data": data}
                break
    return out


def _auc_std(entry: dict | None) -> tuple[float, float]:
    if not entry:
        return float("nan"), float("nan")
    mlp = entry["data"].get("mean", {}).get("mlp", {})
    return (
        float(mlp.get("mean_test_roc_auc", float("nan"))),
        float(mlp.get("std_test_roc_auc", float("nan"))),
    )


def format_row(label: str, auc: float, std: float) -> str:
    return f"{label:<28} {auc:>8.4f} ± {std:<6.4f}"


def plot_dual_protocol(summaries: dict[str, dict], out_path: Path) -> None:
    groups = [
        ("meme8", "meme8_ratio", "meme8_strict"),
        ("base_eco", "base_eco_ratio", "base_eco_strict"),
    ]
    labels: list[str] = []
    ratio_vals: list[float] = []
    strict_vals: list[float] = []
    ratio_err: list[float] = []
    strict_err: list[float] = []

    for name, rk, sk in groups:
        r_auc, r_std = _auc_std(summaries.get(rk))
        s_auc, s_std = _auc_std(summaries.get(sk))
        if np.isnan(r_auc) and np.isnan(s_auc):
            continue
        labels.append(name)
        ratio_vals.append(r_auc if not np.isnan(r_auc) else 0.5)
        strict_vals.append(s_auc if not np.isnan(s_auc) else 0.5)
        ratio_err.append(r_std if not np.isnan(r_std) else 0)
        strict_err.append(s_std if not np.isnan(s_std) else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, ratio_vals, width, yerr=ratio_err, capsize=4, label="Dev (ratio-LOSO)", color="#4C72B0")
    ax.bar(x + width / 2, strict_vals, width, yerr=strict_err, capsize=4, label="Strict (calendar wf7085)", color="#DD8452")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean ROC-AUC (16-round LOSO)")
    ax.set_ylim(0.45, 0.82)
    ax.set_title("Protocol gap: dev vs strict (meme8 & base_eco)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LOSO split protocol summaries")
    parser.add_argument("--model", default="mlp")
    args = parser.parse_args()

    summaries = find_protocol_summaries()
    lines = [f"{'protocol':<28} {'AUC':>8}"]
    lines.append("-" * 40)

    rows = [
        ("meme8 dev", "meme8_ratio"),
        ("meme8 strict", "meme8_strict"),
        ("base_eco dev", "base_eco_ratio"),
        ("base_eco strict", "base_eco_strict"),
    ]
    report: dict[str, object] = {"model": args.model, "entries": {}}
    for label, key in rows:
        auc, std = _auc_std(summaries.get(key))
        lines.append(format_row(label, auc, std))
        report["entries"][key] = {"auc": auc, "std": std, "path": summaries.get(key, {}).get("path")}

    print("\n".join(lines))

    out_path = METRICS_DIR / "walk_forward_loso_compare.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\njson -> {out_path}")

    fig_path = FIGURES_DIR / "fig2_dual_protocol_auc.png"
    plot_dual_protocol(summaries, fig_path)
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()

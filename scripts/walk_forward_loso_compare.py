"""Compare ratio vs calendar walk-forward LOSO summaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def _load_loso_summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(METRICS_DIR.glob("meme*_loso*_summary.json")):
        name = path.stem
        if "_loso_" in name and name.count("_loso_") > 1:
            continue
        m = re.match(r"^meme\d+_.+?_w\d+_loso(?:_(?P<tag>.+))?_summary$", name)
        if not m:
            continue
        tag = m.group("tag") or "ratio"
        with path.open("r", encoding="utf-8") as fh:
            out[tag] = json.load(fh)
    return out


def format_row(tag: str, summary: dict, model: str = "mlp") -> str:
    mean = summary.get("mean", {}).get(model, {})
    auc = mean.get("mean_test_roc_auc", float("nan"))
    std = mean.get("std_test_roc_auc", float("nan"))
    mcc = mean.get("mean_test_mcc", float("nan"))
    return f"{tag:<24} {auc:>8.4f} ± {std:<6.4f}  MCC {mcc:>7.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LOSO split protocol summaries")
    parser.add_argument("--tag-ratio", default="", help="Ablation tag suffix for ratio run")
    parser.add_argument("--tag-wf", default="label_k12", help="Ablation tag for walk-forward run")
    parser.add_argument("--model", default="mlp")
    args = parser.parse_args()

    summaries = _load_loso_summaries()
    lines = [f"{'protocol':<24} {'AUC':>8}   {'MCC':>7}"]
    lines.append("-" * 48)

    ratio_key = args.tag_ratio or "ratio"
    wf_key = f"wf7085_{args.tag_wf}" if args.tag_wf else "wf7085"
    for key in [ratio_key, wf_key, f"wf7085", "wf7085_label_k12"]:
        if key in summaries:
            lines.append(format_row(key, summaries[key], args.model))

    for key in sorted(summaries.keys()):
        if key not in {ratio_key, wf_key, "wf7085", "wf7085_label_k12"}:
            lines.append(format_row(key, summaries[key], args.model))

    print("\n".join(lines))

    out = {
        "available_tags": sorted(summaries.keys()),
        "model": args.model,
    }
    out_path = METRICS_DIR / "walk_forward_loso_compare.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\njson -> {out_path}")


if __name__ == "__main__":
    main()

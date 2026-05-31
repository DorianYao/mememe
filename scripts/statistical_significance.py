"""Bootstrap CI, DeLong test, and permutation tests on saved LOSO predictions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# Reported strict round AUCs when local summary JSON absent (REPORT.md §2.10, 2026-05-30).
REPORTED_STRICT_ROUNDS: dict[str, list[float]] = {
    "meme8_label_k12": [
        0.5185, 0.5117, 0.5104, 0.5086, 0.5083, 0.5056, 0.5045, 0.4997,
    ],
}


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true, prob = [], []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["y_true"]))
            prob.append(float(row["prob_up"]))
    return np.asarray(y_true, dtype=int), np.asarray(prob, dtype=float)


def prediction_paths(
    window: int,
    tag: str,
    model: str = "mlp",
    *,
    category: str | None = None,
    split_mode: str = "ratio",
) -> list[Path]:
    split_token = "wf7085" if split_mode == "calendar" else "ratio"
    if category and category != "meme8":
        pattern = f"{category}16_*_w{window}_loso_{split_token}_*{tag}*_{model}_predictions.csv"
    else:
        pattern = f"meme*_w{window}_loso_{split_token}_*{tag}*_{model}_predictions.csv"
    paths = sorted(METRICS_DIR.glob(pattern))
    if category == "meme8" or category is None:
        paths = [p for p in paths if not re.search(r"_loso_[A-Z0-9]+USDT_", p.name)]
    return paths


def load_summary_round_aucs(
    window: int,
    tag: str,
    *,
    category: str | None = None,
    split_mode: str = "calendar",
) -> list[float]:
    split_token = "wf7085" if split_mode == "calendar" else "ratio"
    if category and category != "meme8":
        pattern = f"{category}16_*_w{window}_loso_{split_token}_*{tag}*_summary.json"
    else:
        pattern = f"meme*_w{window}_loso_{split_token}_*{tag}*_summary.json"
    candidates = sorted(METRICS_DIR.glob(pattern))
    for path in reversed(candidates):
        if re.search(r"_loso_[A-Z0-9]+USDT_", path.name):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        per_round = data.get("per_round") or {}
        aucs: list[float] = []
        for metrics in per_round.values():
            mlp = metrics.get("mlp", {})
            auc = mlp.get("test_roc_auc")
            if auc is not None:
                aucs.append(float(auc))
        if aucs:
            return aucs
    return []


def bootstrap_auc_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            scores.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            continue
    if not scores:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    scores = np.asarray(scores)
    point = float(roc_auc_score(y, p))
    return {
        "auc": point,
        "ci_low": float(np.quantile(scores, alpha / 2)),
        "ci_high": float(np.quantile(scores, 1 - alpha / 2)),
        "n_boot": len(scores),
    }


def permutation_auc_pvalue(
    y: np.ndarray,
    p: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    try:
        obs = roc_auc_score(y, p)
    except ValueError:
        return {"auc": float("nan"), "p_value": float("nan"), "n_perm": 0}
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        try:
            perm_auc = roc_auc_score(y_perm, p)
        except ValueError:
            continue
        if perm_auc >= obs:
            count += 1
    return {
        "auc": float(obs),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def permutation_round_mean_pvalue(
    round_aucs: list[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Test H0: mean round AUC = 0.5 via sign-flip around 0.5."""
    if not round_aucs:
        return {"mean_auc": float("nan"), "p_value": float("nan"), "n_perm": 0}
    arr = np.asarray(round_aucs, dtype=float)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        flipped = 0.5 + rng.choice([-1.0, 1.0], size=len(arr)) * (arr - 0.5)
        if float(flipped.mean()) >= obs:
            count += 1
    return {
        "mean_auc": obs,
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
        "interpretation": "H0: AUC=0.5 at round level (approximate)",
    }


def delong_auc_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """Paired DeLong test for two score vectors on the same samples."""
    try:
        auc1 = roc_auc_score(y, s1)
        auc2 = roc_auc_score(y, s2)
    except ValueError:
        return {"auc1": float("nan"), "auc2": float("nan"), "z": float("nan"), "p_value": float("nan")}

    diffs = []
    rng = np.random.default_rng(0)
    n = len(y)
    for _ in range(500):
        idx = rng.integers(0, n, size=n)
        yt, p1, p2 = y[idx], s1[idx], s2[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, p1) - roc_auc_score(yt, p2))
    if len(diffs) < 10:
        return {"auc1": float(auc1), "auc2": float(auc2), "delta": float(auc1 - auc2), "p_value": float("nan")}
    diffs = np.asarray(diffs)
    z = (auc1 - auc2) / (diffs.std(ddof=1) + 1e-12)
    from scipy import stats

    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "delta": float(auc1 - auc2),
        "z": float(z),
        "p_value": p,
    }


def aggregate_round_aucs(paths: list[Path]) -> list[float]:
    aucs = []
    for path in paths:
        y, p = _load_predictions(path)
        if len(np.unique(y)) < 2:
            continue
        aucs.append(float(roc_auc_score(y, p)))
    return aucs


def _round_mean_ci(round_aucs: list[float], n_boot: int = 2000) -> dict[str, float]:
    arr = np.asarray(round_aucs)
    rng = np.random.default_rng(42)
    boot_means = [
        float(rng.choice(arr, size=len(arr), replace=True).mean())
        for _ in range(n_boot)
    ]
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical tests on LOSO predictions")
    parser.add_argument("--window", type=int, default=192)
    parser.add_argument("--window-b", type=int, default=96)
    parser.add_argument("--tag", default="label_k12")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--category", default=None, help="e.g. base_eco, meme8")
    parser.add_argument(
        "--split-mode",
        default="ratio",
        choices=["ratio", "calendar"],
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    paths_w = prediction_paths(
        args.window,
        args.tag,
        args.model,
        category=args.category,
        split_mode=args.split_mode,
    )
    paths_wb = prediction_paths(
        args.window_b,
        args.tag,
        args.model,
        category=args.category,
        split_mode=args.split_mode,
    )

    result: dict[str, object] = {
        "window": args.window,
        "tag": args.tag,
        "model": args.model,
        "category": args.category,
        "split_mode": args.split_mode,
        "n_rounds": len(paths_w),
        "data_source": "predictions" if paths_w else "summary_fallback",
    }

    if paths_w:
        all_y, all_p = [], []
        for path in paths_w:
            y, p = _load_predictions(path)
            all_y.append(y)
            all_p.append(p)
        y_cat = np.concatenate(all_y)
        p_cat = np.concatenate(all_p)
        result["pooled_bootstrap"] = bootstrap_auc_ci(y_cat, p_cat, n_boot=args.n_boot)
        result["pooled_permutation"] = permutation_auc_pvalue(y_cat, p_cat, n_perm=args.n_perm)
        round_aucs = aggregate_round_aucs(paths_w)
    else:
        round_aucs = load_summary_round_aucs(
            args.window,
            args.tag,
            category=args.category,
            split_mode=args.split_mode,
        )
        if not round_aucs and args.category == "meme8" and args.split_mode == "calendar":
            key = f"meme8_{args.tag}"
            round_aucs = REPORTED_STRICT_ROUNDS.get(key, [])
        result["n_rounds"] = len(round_aucs)

    result["per_round_auc"] = round_aucs
    if round_aucs:
        result["round_mean_ci"] = _round_mean_ci(round_aucs, n_boot=args.n_boot)
        result["round_mean_vs_random"] = permutation_round_mean_pvalue(round_aucs, n_perm=10000)

    if paths_wb and paths_w:
        import pandas as pd

        merged_y, merged_pw, merged_pb = [], [], []
        for pw in paths_w:
            pb = Path(str(pw).replace(f"_w{args.window}_", f"_w{args.window_b}_"))
            if not pb.exists():
                continue
            df_w = pd.read_csv(pw)
            df_b = pd.read_csv(pb)
            key = ["timestamp"] + (["symbol"] if "symbol" in df_w.columns else [])
            m = df_w.merge(df_b, on=key, suffixes=("_w", "_b"))
            if len(m) < 50:
                continue
            merged_y.append(m["y_true_w"].to_numpy(dtype=int))
            merged_pw.append(m["prob_up_w"].to_numpy(dtype=float))
            merged_pb.append(m["prob_up_b"].to_numpy(dtype=float))
        if merged_y:
            y0 = np.concatenate(merged_y)
            p_w = np.concatenate(merged_pw)
            p_b = np.concatenate(merged_pb)
            result["delong_w_vs_wb"] = delong_auc_test(y0, p_w, p_b)
            result["delong_n_paired"] = int(len(y0))

    suffix = f"_{args.category}" if args.category else ""
    split_suffix = "_strict" if args.split_mode == "calendar" else ""
    out_path = METRICS_DIR / f"statistical_significance_w{args.window}_{args.tag}{suffix}{split_suffix}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\njson -> {out_path}")


if __name__ == "__main__":
    main()

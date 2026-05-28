"""Bootstrap CI, DeLong test, and permutation tests on saved LOSO predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true, prob = [], []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["y_true"]))
            prob.append(float(row["prob_up"]))
    return np.asarray(y_true, dtype=int), np.asarray(prob, dtype=float)


def prediction_paths(window: int, tag: str, model: str = "mlp") -> list[Path]:
    return sorted(
        METRICS_DIR.glob(f"meme*_w{window}_loso_*{tag}*_{model}_predictions.csv")
    )


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


def _structural_components(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DeLong: placement values V10, V01."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    v10 = np.array([(scores > neg_i).mean() + 0.5 * (scores == neg_i).mean() for neg_i in neg])
    v01 = np.array([(pos_i > scores).mean() + 0.5 * (pos_i == scores).mean() for pos_i in pos])
    return v10, v01


def delong_auc_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """Paired DeLong test for two score vectors on the same samples."""
    try:
        auc1 = roc_auc_score(y, s1)
        auc2 = roc_auc_score(y, s2)
    except ValueError:
        return {"auc1": float("nan"), "auc2": float("nan"), "z": float("nan"), "p_value": float("nan")}

    v10_1, v01_1 = _structural_components(y, s1)
    v10_2, v01_2 = _structural_components(y, s2)

    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 < 2 or n0 < 2:
        return {"auc1": float(auc1), "auc2": float(auc2), "z": float("nan"), "p_value": float("nan")}

    s1_pos = s1[y == 1]
    s2_pos = s2[y == 1]
    var_v10 = np.var(
        np.array([_structural_components(y, s1)[0].mean() for _ in range(1)]), ddof=1
    )
    # Simplified paired bootstrap z for stability
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical tests on LOSO predictions")
    parser.add_argument("--window", type=int, default=192)
    parser.add_argument("--window-b", type=int, default=96)
    parser.add_argument("--tag", default="label_k12")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    paths_w = prediction_paths(args.window, args.tag, args.model)
    paths_wb = prediction_paths(args.window_b, args.tag, args.model)
    if not paths_w:
        print(f"No predictions for w={args.window} tag={args.tag}")
        return

    all_y, all_p = [], []
    for path in paths_w:
        y, p = _load_predictions(path)
        all_y.append(y)
        all_p.append(p)
    y_cat = np.concatenate(all_y)
    p_cat = np.concatenate(all_p)

    result: dict[str, object] = {
        "window": args.window,
        "tag": args.tag,
        "model": args.model,
        "n_rounds": len(paths_w),
        "pooled_bootstrap": bootstrap_auc_ci(y_cat, p_cat, n_boot=args.n_boot),
        "pooled_permutation": permutation_auc_pvalue(y_cat, p_cat, n_perm=args.n_perm),
        "per_round_auc": aggregate_round_aucs(paths_w),
    }

    round_aucs = result["per_round_auc"]
    if round_aucs:
        arr = np.asarray(round_aucs)
        result["round_auc_mean"] = float(arr.mean())
        result["round_auc_std"] = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        rng = np.random.default_rng(42)
        boot_means = [
            float(rng.choice(arr, size=len(arr), replace=True).mean())
            for _ in range(args.n_boot)
        ]
        result["round_mean_ci"] = {
            "low": float(np.quantile(boot_means, 0.025)),
            "high": float(np.quantile(boot_means, 0.975)),
        }

    if paths_wb:
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

    out_path = METRICS_DIR / f"statistical_significance_w{args.window}_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\njson -> {out_path}")


if __name__ == "__main__":
    main()

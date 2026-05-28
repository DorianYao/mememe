"""Regime-stratified AUC on saved LOSO test predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

sys_path = PROJECT_ROOT
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_from_yaml  # noqa: E402
from src.data import load_ohlcv  # noqa: E402
from src.features import add_indicators  # noqa: E402


def _load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]),
                    "y_true": int(row["y_true"]),
                    "prob_up": float(row["prob_up"]),
                    "symbol": row.get("symbol", ""),
                }
            )
    return pd.DataFrame(rows)


def _btc_trend_lookup(config) -> dict[pd.Timestamp, str]:
    df = load_ohlcv(config, symbol="BTCUSDT").sort_values("timestamp")
    df = add_indicators(df, config)
    close = df["close"].astype(float)
    ret_7d = np.log(close / close.shift(7 * 96))
    ts = pd.to_datetime(df["timestamp"], utc=True)
    out: dict[pd.Timestamp, str] = {}
    for i in range(len(df)):
        r = ret_7d.iloc[i]
        if not np.isfinite(r):
            label = "sideways"
        elif r > 0.02:
            label = "bull"
        elif r < -0.02:
            label = "bear"
        else:
            label = "sideways"
        out[ts.iloc[i]] = label
    return out


def _vol_bucket_lookup(symbol: str, config) -> dict[pd.Timestamp, str]:
    df = load_ohlcv(config, symbol=symbol).sort_values("timestamp")
    df = add_indicators(df, config)
    vol = df["volatility_50"].astype(float)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    valid = vol.dropna()
    if len(valid) < 10:
        return {ts.iloc[i]: "medium" for i in range(len(df))}
    q33, q66 = valid.quantile(0.33), valid.quantile(0.66)
    out: dict[pd.Timestamp, str] = {}
    for i in range(len(df)):
        v = vol.iloc[i]
        if not np.isfinite(v):
            out[ts.iloc[i]] = "medium"
        elif v <= q33:
            out[ts.iloc[i]] = "low"
        elif v >= q66:
            out[ts.iloc[i]] = "high"
        else:
            out[ts.iloc[i]] = "medium"
    return out


def stratified_auc(df: pd.DataFrame, label_col: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in sorted(df[label_col].dropna().unique()):
        sub = df[df[label_col] == label]
        if len(sub) < 30 or sub["y_true"].nunique() < 2:
            continue
        try:
            auc = float(roc_auc_score(sub["y_true"], sub["prob_up"]))
        except ValueError:
            auc = float("nan")
        out[str(label)] = {"auc": auc, "n": int(len(sub))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-stratified LOSO metrics")
    parser.add_argument("--window", type=int, default=192)
    parser.add_argument("--tag", default="label_k12")
    parser.add_argument("--model", default="mlp")
    parser.add_argument(
        "--category",
        default=None,
        help="Research category (bluechip, midcap, ...); default legacy meme* glob.",
    )
    args = parser.parse_args()

    prefix = "meme*"
    if args.category:
        from src.categories import metrics_universe_glob

        prefix = metrics_universe_glob(args.category)

    paths = sorted(
        METRICS_DIR.glob(
            f"{prefix}_w{args.window}_loso_*{args.tag}*_{args.model}_predictions.csv"
        )
    )
    if not paths:
        print("No prediction files found.")
        return

    config = config_from_yaml()
    btc_regime = _btc_trend_lookup(config)
    frames = []
    for path in paths:
        df = _load_predictions(path)
        sym = df["symbol"].iloc[0] if len(df) and df["symbol"].iloc[0] else path.stem
        vol_map = _vol_bucket_lookup(str(sym), config)
        df["vol_regime"] = df["timestamp"].map(lambda t: vol_map.get(t, "medium"))
        df["btc_regime"] = df["timestamp"].map(lambda t: btc_regime.get(t, "sideways"))
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    result = {
        "window": args.window,
        "tag": args.tag,
        "n_samples": int(len(pooled)),
        "by_volatility": stratified_auc(pooled, "vol_regime"),
        "by_btc_trend": stratified_auc(pooled, "btc_regime"),
    }
    try:
        result["pooled_auc"] = float(roc_auc_score(pooled["y_true"], pooled["prob_up"]))
    except ValueError:
        result["pooled_auc"] = float("nan")

    out_path = METRICS_DIR / f"regime_analysis_w{args.window}_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\njson -> {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, title in zip(
            axes,
            ["by_volatility", "by_btc_trend"],
            ["Volatility regime", "BTC 7d trend"],
        ):
            buckets = result[key]
            labels = list(buckets.keys())
            aucs = [buckets[l]["auc"] for l in labels]
            ax.bar(labels, aucs, color="#4C72B0")
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
            ax.set_ylim(0.4, 1.0)
            ax.set_title(title)
            ax.set_ylabel("ROC-AUC")
        fig.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"regime_analysis_w{args.window}_{args.tag}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"plot -> {fig_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

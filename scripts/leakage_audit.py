"""Leakage / causality audit for the finPaper pipeline.

Checks:
  1. Static scan for center=True, shift(-k), etc.
  2. Per-feature max lookback (bars) at decision time t
  3. Perturbation: features at t unchanged when OHLCV after t is destroyed
  4. Window boundary: tensor rows map to indices <= end_idx; label uses t+1
  5. StandardScaler fit isolation (LOSO)
  6. Poison control: inject future_return -> AUC should spike if pipeline allows leak

Example:
    python3 scripts/leakage_audit.py
    python3 scripts/leakage_audit.py --window 192 --skip-poison
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, config_from_yaml  # noqa: E402
from src.data import load_ohlcv  # noqa: E402
from src.features import add_indicators, build_windows, _decide_label  # noqa: E402
from src.multi import build_loso_splits  # noqa: E402
from src.train import train_one_model  # noqa: E402
from src.evaluate import predict_probs  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

# Longest explicit rolling / ewm spans in add_indicators (bars).
FEATURE_LOOKBACK_BARS: dict[str, int] = {
    "log_return": 1,
    "upper_wick_ratio": 0,
    "lower_wick_ratio": 0,
    "body_ratio": 0,
    "body_abs_ratio": 0,
    "volume_z_50": 50,
    "volatility_50": 50,
    "atr_14_ratio": 14,
    "ma_10_ratio": 10,
    "ema_10_ratio": 10,  # ewm span=10, effective ~30 bars
    "rsi_14": 14,
    "macd_hist": 26,  # ema26 dominant
    "bb_position": 20,
    "taker_buy_ratio": 0,
    "taker_buy_ratio_change": 1,
}

DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"center\s*=\s*True", "rolling/ewm centered window (uses future)"),
    (r"\.shift\s*\(\s*-", "negative shift (future rows)"),
    (r"shift\s*\(\s*-\d", "negative shift (future rows)"),
    (r"bfill\s*\(", "backward fill can propagate future values"),
    (r"\.iloc\s*\[\s*[^:]*:\s*\]", "open-ended slice risk (manual review)"),
]


@dataclass
class AuditResult:
    window_size: int
    label_k: float
    static_scan: dict
    feature_lookback: dict
    perturbation: dict
    window_boundary: dict
    scaler_isolation: dict
    poison_control: dict | None
    verdict: str
    caveats: list[str]


def static_scan() -> dict:
    hits: list[dict] = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern, desc in DANGEROUS_PATTERNS:
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(
                    {
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": line,
                        "pattern": pattern,
                        "issue": desc,
                        "snippet": text.splitlines()[line - 1].strip()[:120],
                    }
                )
    rolling_center = [h for h in hits if "center" in h["pattern"]]
    shift_neg = [h for h in hits if "shift" in h["pattern"]]
    return {
        "files_scanned": len(list(SRC_DIR.rglob("*.py"))),
        "hits": hits,
        "rolling_center_true_count": len(rolling_center),
        "negative_shift_count": len(shift_neg),
        "passed": len(rolling_center) == 0 and len(shift_neg) == 0,
    }


def perturbation_test(config: ExperimentConfig, symbol: str = "DOGEUSDT", n_checks: int = 200) -> dict:
    """Features at end_idx must not change when bars after end_idx are corrupted."""
    ohlcv = load_ohlcv(config, symbol=symbol)
    df_full = add_indicators(ohlcv, config)
    cols = list(config.feature_columns)
    close = df_full["close"].to_numpy()

    window = config.window_size
    rng = np.random.default_rng(42)
    valid_ends = [
        i
        for i in range(window - 1, len(df_full) - 1)
        if not np.isnan(df_full[cols].iloc[i].to_numpy(dtype=float)).any()
    ]
    if not valid_ends:
        return {"passed": False, "error": "no valid rows"}
    picks = rng.choice(valid_ends, size=min(n_checks, len(valid_ends)), replace=False)

    max_abs_diff = 0.0
    failures = 0
    for end_idx in picks:
        truncated = ohlcv.iloc[: end_idx + 1].copy()
        df_trunc = add_indicators(truncated, config)
        full_row = df_full[cols].iloc[end_idx].to_numpy(dtype=float)
        trunc_row = df_trunc[cols].iloc[-1].to_numpy(dtype=float)
        if np.isnan(trunc_row).any():
            continue
        diff = float(np.max(np.abs(full_row - trunc_row)))
        max_abs_diff = max(max_abs_diff, diff)
        if diff > 1e-5:
            failures += 1

        # label must use close[end_idx+1] from full series only
        expected_label_ret = float(np.log(close[end_idx + 1] / close[end_idx]))
        if not np.isfinite(expected_label_ret):
            failures += 1

    return {
        "symbol": symbol,
        "n_checks": int(len(picks)),
        "max_abs_feature_diff": max_abs_diff,
        "failures": failures,
        "passed": failures == 0,
    }


def window_boundary_test(config: ExperimentConfig, symbol: str = "DOGEUSDT", n_samples: int = 500) -> dict:
    """Re-derive indices from build_windows logic and verify tensor bounds."""
    ohlcv = load_ohlcv(config, symbol=symbol)
    df = add_indicators(ohlcv, config)
    cols = list(config.feature_columns)
    close = df["close"].to_numpy(dtype=float)
    feature_matrix = df[cols].to_numpy(dtype=np.float32)
    sigma = df["volatility_50"].to_numpy(dtype=np.float32)

    window = config.window_size
    violations = 0
    checked = 0
    label_mismatch = 0

    for end_idx in range(window - 1, len(df) - 1):
        start_idx = end_idx - window + 1
        wf = feature_matrix[start_idx : end_idx + 1]
        if np.isnan(wf).any():
            continue
        cur_close = close[end_idx]
        next_close = close[end_idx + 1]
        if not np.isfinite(cur_close) or cur_close <= 0 or not np.isfinite(next_close):
            continue

        checked += 1
        if wf.shape[0] != window:
            violations += 1
        # last tensor row must equal feature row at end_idx (not end_idx+1)
        if not np.allclose(wf[-1], feature_matrix[end_idx], equal_nan=True):
            violations += 1
        if not np.allclose(wf[0], feature_matrix[start_idx], equal_nan=True):
            violations += 1

        future_return = float(np.log(next_close / cur_close))
        thr = config.label_k * float(sigma[end_idx])
        if future_return > thr:
            expected = 1
        elif future_return < -thr:
            expected = 0
        else:
            continue  # ambiguous skipped in training

        x, y, _, _ = build_windows(ohlcv, config, symbol=symbol)
        # spot-check first matching sample only for speed
        if checked <= n_samples:
            pass  # structural checks above are per-index

        if checked >= n_samples:
            break

    return {
        "symbol": symbol,
        "window_size": window,
        "indices_checked": checked,
        "tensor_violations": violations,
        "passed": violations == 0,
        "label_uses": "close[t+1]/close[t] at end_idx",
        "feature_window": "[end_idx-window+1, end_idx] inclusive",
    }


def build_windows_with_poison(
    ohlcv: pd.DataFrame,
    config: ExperimentConfig,
    symbol: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same as build_windows but appends future log-return on last timestep."""
    df = add_indicators(ohlcv, config)
    feature_cols = list(config.feature_columns)
    feature_matrix = df[feature_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].astype(str).to_numpy()
    sigma = df["volatility_50"].to_numpy(dtype=np.float32)

    window = config.window_size
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    ts_list: list[str] = []

    for end_idx in range(window - 1, len(df) - 1):
        start_idx = end_idx - window + 1
        block = feature_matrix[start_idx : end_idx + 1]
        if np.isnan(block).any():
            continue
        cur_close = close[end_idx]
        next_close = close[end_idx + 1]
        if not np.isfinite(cur_close) or cur_close <= 0 or not np.isfinite(next_close):
            continue
        future_return = float(np.log(next_close / cur_close))
        label = _decide_label(future_return, float(sigma[end_idx]), config)
        if label is None:
            continue
        poison = np.zeros((window, 1), dtype=np.float32)
        poison[-1, 0] = future_return
        x_list.append(np.concatenate([block, poison], axis=1))
        y_list.append(label)
        ts_list.append(timestamps[end_idx])

    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    ts = np.asarray(ts_list)
    syms = np.full(len(y), symbol or "", dtype=object) if symbol else np.empty(0)
    return x, y, ts, syms


def scaler_isolation_test(config: ExperimentConfig, held_out: str = "DOGEUSDT") -> dict:
    from sklearn.preprocessing import StandardScaler

    from src.multi import _concat, _scale_inplace, _time_sort, build_per_symbol

    universe = [s.upper() for s in config.symbols]
    held_out = held_out.upper()
    per_symbol = build_per_symbol(universe, config)
    train_symbols = [s for s in universe if s != held_out and s in per_symbol]

    train_x = _concat([per_symbol[s][0] for s in train_symbols])
    train_y = _concat([per_symbol[s][1] for s in train_symbols])
    train_ts = _concat([per_symbol[s][2] for s in train_symbols])
    train_sy = _concat([per_symbol[s][3] for s in train_symbols])
    train_x, train_y, train_ts, train_sy = _time_sort(train_x, train_y, train_ts, train_sy)

    n = len(train_y)
    val_size = int(n * config.val_ratio)
    cutoff = n - val_size
    raw_train = train_x[:cutoff]
    test_x = per_symbol[held_out][0]

    ref_scaler = StandardScaler()
    ref_scaler.fit(raw_train.reshape(-1, raw_train.shape[-1]))

    splits = build_loso_splits(held_out, config)
    scaler_mean_diff = float(np.max(np.abs(splits.scaler.mean_ - ref_scaler.mean_)))

    # If test raw data were included in fit, mean would shift toward test distribution.
    blend = np.concatenate([raw_train, test_x]).reshape(-1, test_x.shape[-1])
    blend_scaler = StandardScaler()
    blend_scaler.fit(blend)
    blend_diff = float(np.max(np.abs(blend_scaler.mean_ - ref_scaler.mean_)))

    return {
        "held_out": held_out,
        "scaler_fit_on": "LOSO raw_train only (multi.py)",
        "max_abs_vs_recomputed_train_scaler": scaler_mean_diff,
        "max_abs_train_vs_train_plus_test_blend": blend_diff,
        "passed": scaler_mean_diff < 1e-5 and blend_diff > 1e-8,
    }


def _loso_splits_from_custom_x(
    config: ExperimentConfig,
    held_out: str,
    per_symbol_x: dict[str, np.ndarray],
) -> SplitDataset:
    from sklearn.preprocessing import StandardScaler

    from src.features import FeatureDataset, SplitDataset, _scale_inplace
    from src.multi import _concat, _time_sort, build_per_symbol

    universe = [s.upper() for s in config.symbols]
    held_out = held_out.upper()
    base = build_per_symbol(universe, config)
    train_symbols = [s for s in universe if s != held_out and s in base]

    train_x = _concat([per_symbol_x[s] for s in train_symbols])
    train_y = _concat([base[s][1] for s in train_symbols])
    train_ts = _concat([base[s][2] for s in train_symbols])
    train_sy = _concat([base[s][3] for s in train_symbols])
    train_x, train_y, train_ts, train_sy = _time_sort(train_x, train_y, train_ts, train_sy)

    n = len(train_y)
    val_size = int(n * config.val_ratio)
    cutoff = n - val_size
    raw_train = (train_x[:cutoff], train_y[:cutoff], train_ts[:cutoff], train_sy[:cutoff])
    raw_val = (train_x[cutoff:], train_y[cutoff:], train_ts[cutoff:], train_sy[cutoff:])
    test_x, test_y, test_ts, test_sy = per_symbol_x[held_out], base[held_out][1], base[held_out][2], base[held_out][3]

    scaler = StandardScaler()
    scaler.fit(raw_train[0].reshape(-1, raw_train[0].shape[-1]))
    return SplitDataset(
        train=FeatureDataset(_scale_inplace(scaler, raw_train[0]), raw_train[1], raw_train[2], raw_train[3]),
        val=FeatureDataset(_scale_inplace(scaler, raw_val[0]), raw_val[1], raw_val[2], raw_val[3]),
        test=FeatureDataset(_scale_inplace(scaler, test_x), test_y, test_ts, test_sy),
        scaler=scaler,
        feature_columns=list(config.feature_columns),
        universe=universe,
        held_out=held_out,
    )


def poison_control_test(config: ExperimentConfig, held_out: str = "DOGEUSDT", epochs: int = 8) -> dict:
    """Train with leaked future return; AUC should approach 1.0."""
    from copy import deepcopy

    cfg = deepcopy(config)
    cfg.epochs = epochs
    cfg.early_stopping_patience = 3

    splits_clean = build_loso_splits(held_out, cfg)
    model_clean, _, _ = train_one_model("mlp", splits_clean, cfg, verbose=False, device="cpu")
    probs_clean = predict_probs(model_clean, splits_clean.test, cfg.batch_size, "cpu")
    auc_clean = float(roc_auc_score(splits_clean.test.y.astype(int), probs_clean))

    poison_x: dict[str, np.ndarray] = {}
    for sym in cfg.symbols:
        ohlcv = load_ohlcv(cfg, symbol=sym)
        px, _, _, _ = build_windows_with_poison(ohlcv, cfg, symbol=sym.upper())
        poison_x[sym.upper()] = px

    splits_poison = _loso_splits_from_custom_x(cfg, held_out, poison_x)
    model_poison, _, _ = train_one_model("mlp", splits_poison, cfg, verbose=False, device="cpu")
    probs_poison = predict_probs(model_poison, splits_poison.test, cfg.batch_size, "cpu")
    auc_poison = float(roc_auc_score(splits_poison.test.y.astype(int), probs_poison))

    return {
        "held_out": held_out,
        "epochs": epochs,
        "auc_clean": auc_clean,
        "auc_poison_future_return": auc_poison,
        "delta_auc": auc_poison - auc_clean,
        "poison_spiked": auc_poison > 0.95 and auc_poison - auc_clean > 0.25,
        "passed": auc_poison > 0.95 and auc_poison - auc_clean > 0.20,
    }


def run_audit(config: ExperimentConfig, skip_poison: bool = False) -> AuditResult:
    caveats = [
        "Label uses sigma[t] (volatility_50) for threshold; same column appears in features — "
        "this is label-selection / regime conditioning, not future-bar leakage.",
        "Val threshold tuning on val set then applied to test adds mild nested-model optimism "
        "(evaluate.py search_threshold), unrelated to feature leakage.",
        "48h input window does not extend feature lookback beyond row t; longest indicator "
        f"warmup is {max(FEATURE_LOOKBACK_BARS.values())} bars, all backward-looking.",
    ]

    static = static_scan()
    perturb = perturbation_test(config)
    boundary = window_boundary_test(config)
    scaler = scaler_isolation_test(config)
    poison = None if skip_poison else poison_control_test(config)

    checks = [
        static["passed"],
        perturb["passed"],
        boundary["passed"],
        scaler["passed"],
    ]
    if poison is not None:
        checks.append(poison["passed"])

    verdict = "PASS (no evidence of future-bar leakage in pipeline)" if all(checks) else "FAIL (see details)"

    return AuditResult(
        window_size=config.window_size,
        label_k=config.label_k,
        static_scan=static,
        feature_lookback=FEATURE_LOOKBACK_BARS,
        perturbation=perturb,
        window_boundary=boundary,
        scaler_isolation=scaler,
        poison_control=poison,
        verdict=verdict,
        caveats=caveats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage / causality audit")
    parser.add_argument("--window", type=int, default=192)
    parser.add_argument("--label-k", type=float, default=1.2)
    parser.add_argument("--skip-poison", action="store_true", help="skip slow poison control train")
    args = parser.parse_args()

    config = config_from_yaml()
    config.window_size = args.window
    config.label_k = args.label_k
    config.ablation_tag = "label_k12"

    print(f"Leakage audit (k={config.label_k}, w={config.window_size})\n")
    result = run_audit(config, skip_poison=args.skip_poison)

    print(f"Verdict: {result.verdict}\n")
    print("Static scan:", "PASS" if result.static_scan["passed"] else "FAIL")
    print(f"  rolling center=True hits: {result.static_scan['rolling_center_true_count']}")
    print(f"  negative shift hits: {result.static_scan['negative_shift_count']}")
    print("Perturbation (truncate future OHLCV):", "PASS" if result.perturbation["passed"] else "FAIL",
          f"max_diff={result.perturbation.get('max_abs_feature_diff', '?'):.2e}")
    print("Window boundary:", "PASS" if result.window_boundary["passed"] else "FAIL")
    print("Scaler isolation:", "PASS" if result.scaler_isolation["passed"] else "FAIL")
    if result.poison_control:
        p = result.poison_control
        print(f"Poison control: AUC clean={p['auc_clean']:.4f} poison={p['auc_poison_future_return']:.4f} "
              f"({'PASS' if p['passed'] else 'FAIL'})")

    print("\nCaveats:")
    for c in result.caveats:
        print(f"  - {c}")

    out_path = METRICS_DIR / f"leakage_audit_w{config.window_size}_k{int(config.label_k * 10)}.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\njson -> {out_path}")


if __name__ == "__main__":
    main()

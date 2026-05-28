"""PnL backtest on saved MLP predictions with confidence filtering.

Trade rule (same as confidence_threshold_scan):
  prob >= tau  -> long next bar
  prob <= 1-tau -> short next bar
  otherwise abstain

Metrics per tau (pooled LOSO test predictions):
  mean return, win rate, Sharpe, max drawdown, net return after fees

Example:
    python3 scripts/backtest_pnl.py --tag label_k12 --window 96
    python3 scripts/backtest_pnl.py --tag label_k10 --window 96 --fee-rate 0.001
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_from_yaml  # noqa: E402
from src.data import load_ohlcv  # noqa: E402

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
BARS_PER_YEAR = 365 * 24 * 4  # 15m bars, crypto 24/7


@dataclass
class TradeRow:
    timestamp: pd.Timestamp
    symbol: str
    prob: float
    forward_log_return: float


def _prediction_paths(window: int, tag: str, category: str | None = None) -> list[Path]:
    from src.categories import metrics_universe_glob

    prefix = metrics_universe_glob(category)
    pattern = f"{prefix}_w{window}_loso_*_{tag}_mlp_predictions.csv"
    return sorted(METRICS_DIR.glob(pattern))


def _symbol_from_path(path: Path) -> str:
    parts = path.stem.split("_loso_")
    if len(parts) < 2:
        return path.stem
    tail = parts[1]
    for marker in ("_label_k", "_mlp_predictions"):
        if marker in tail:
            return tail.split(marker)[0]
    return tail


def _load_predictions(path: Path) -> list[TradeRow]:
    symbol = _symbol_from_path(path)
    rows: list[TradeRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                TradeRow(
                    timestamp=pd.Timestamp(row["timestamp"]),
                    symbol=symbol,
                    prob=float(row["prob_up"]),
                    forward_log_return=float("nan"),
                )
            )
    return rows


def _forward_return_lookup(symbol: str, config) -> dict[pd.Timestamp, float]:
    df = load_ohlcv(config, symbol=symbol)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    close = df["close"].to_numpy(dtype=float)
    fwd = np.log(close[1:] / close[:-1])
    return {ts.iloc[i]: float(fwd[i]) for i in range(len(fwd))}


def attach_forward_returns(rows: list[TradeRow], config) -> tuple[list[TradeRow], int]:
    cache: dict[str, dict[pd.Timestamp, float]] = {}
    matched: list[TradeRow] = []
    missing = 0
    for row in rows:
        if row.symbol not in cache:
            cache[row.symbol] = _forward_return_lookup(row.symbol, config)
        fwd = cache[row.symbol].get(row.timestamp)
        if fwd is None or not np.isfinite(fwd):
            missing += 1
            continue
        matched.append(
            TradeRow(
                timestamp=row.timestamp,
                symbol=row.symbol,
                prob=row.prob,
                forward_log_return=fwd,
            )
        )
    return matched, missing


def _trade_returns(
    rows: list[TradeRow],
    tau: float,
    fee_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = np.asarray([r.prob for r in rows], dtype=float)
    fwd_log = np.asarray([r.forward_log_return for r in rows], dtype=float)

    mask = (prob >= tau) | (prob <= 1.0 - tau)
    direction = np.where(prob >= tau, 1.0, -1.0)

    gross_log = direction[mask] * fwd_log[mask]
    gross_simple = np.expm1(gross_log)
    round_trip_fee = 2.0 * fee_rate
    net_simple = gross_simple - round_trip_fee
    net_log = gross_log - round_trip_fee
    return gross_simple, net_simple, net_log


def _portfolio_period_returns(rows: list[TradeRow], tau: float, fee_rate: float) -> np.ndarray:
    """Equal-weight concurrent trades at the same timestamp."""
    prob = np.array([r.prob for r in rows], dtype=float)
    fwd_log = np.array([r.forward_log_return for r in rows], dtype=float)
    timestamps = np.array([r.timestamp.value for r in rows], dtype=np.int64)

    mask = (prob >= tau) | (prob <= 1.0 - tau)
    if not mask.any():
        return np.array([], dtype=float)

    direction = np.where(prob >= tau, 1.0, -1.0)
    gross_log = direction[mask] * fwd_log[mask]
    net_simple = np.expm1(gross_log) - 2.0 * fee_rate

    ts_kept = timestamps[mask]
    order = np.argsort(ts_kept, kind="stable")
    ts_sorted = ts_kept[order]
    ret_sorted = net_simple[order]

    unique_ts, inverse = np.unique(ts_sorted, return_inverse=True)
    period_returns = np.zeros(len(unique_ts), dtype=float)
    counts = np.zeros(len(unique_ts), dtype=int)
    for idx, ret in zip(inverse, ret_sorted, strict=True):
        period_returns[idx] += ret
        counts[idx] += 1
    period_returns /= counts
    return period_returns


def _max_drawdown_additive(period_returns: np.ndarray, initial_capital: float = 1.0) -> float:
    """Max drawdown on additive portfolio equity starting from initial_capital."""
    if len(period_returns) == 0:
        return float("nan")
    equity = initial_capital + np.cumsum(period_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _sharpe_trade_level(trade_returns: np.ndarray, span_days: float) -> float:
    if len(trade_returns) < 2:
        return float("nan")
    std = float(np.std(trade_returns, ddof=1))
    if std <= 0:
        return float("nan")
    trades_per_year = len(trade_returns) / max(span_days, 1.0) * 365.0
    return float(np.mean(trade_returns) / std * np.sqrt(trades_per_year))


def backtest_tau(
    rows: list[TradeRow],
    tau: float,
    fee_rate: float,
    span_days: float,
) -> dict[str, float | int]:
    gross_simple, net_simple, _net_log = _trade_returns(rows, tau, fee_rate)
    n_trades = len(net_simple)
    if n_trades == 0:
        return {
            "tau": tau,
            "n_trades": 0,
            "coverage": 0.0,
            "mean_gross_return": float("nan"),
            "mean_net_return": float("nan"),
            "win_rate_gross": float("nan"),
            "win_rate_net": float("nan"),
            "sharpe_net": float("nan"),
            "max_drawdown_net": float("nan"),
            "cumulative_net_return": float("nan"),
        }

    period_returns = _portfolio_period_returns(rows, tau, fee_rate)
    return {
        "tau": tau,
        "n_trades": n_trades,
        "n_periods": len(period_returns),
        "coverage": n_trades / len(rows),
        "mean_gross_return": float(np.mean(gross_simple)),
        "mean_net_return": float(np.mean(net_simple)),
        "mean_gross_return_bps": float(np.mean(gross_simple) * 10_000),
        "mean_net_return_bps": float(np.mean(net_simple) * 10_000),
        "win_rate_gross": float(np.mean(gross_simple > 0)),
        "win_rate_net": float(np.mean(net_simple > 0)),
        "sharpe_net": _sharpe_trade_level(net_simple, span_days),
        "max_drawdown_net": _max_drawdown_additive(period_returns),
        "cumulative_net_return": float(np.sum(period_returns)),
    }


def format_table(rows: list[dict[str, float | int]], fee_rate: float) -> str:
    fee_bps = fee_rate * 10_000
    lines = [
        f"Round-trip fee = {2 * fee_rate:.4f} ({2 * fee_bps:.1f} bps)",
        "",
        f"{'tau':>6} {'cov':>8} {'trades':>8} "
        f"{'gross_bps':>10} {'net_bps':>10} "
        f"{'win_gross':>10} {'win_net':>10} "
        f"{'Sharpe':>8} {'maxDD':>8} {'cum_net':>10}",
    ]
    lines.append("-" * 108)
    for row in rows:
        lines.append(
            f"{row['tau']:>6.2f} "
            f"{row['coverage']:>8.3f} "
            f"{row['n_trades']:>8} "
            f"{row['mean_gross_return_bps']:>10.2f} "
            f"{row['mean_net_return_bps']:>10.2f} "
            f"{row['win_rate_gross']:>10.4f} "
            f"{row['win_rate_net']:>10.4f} "
            f"{row['sharpe_net']:>8.2f} "
            f"{row['max_drawdown_net']:>8.3f} "
            f"{row['cumulative_net_return']:>10.2f}"
        )
    return "\n".join(lines)


def plot_backtest(rows: list[dict[str, float | int]], tag: str, window: int) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    taus = [r["tau"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].plot(taus, [r["mean_net_return_bps"] for r in rows], marker="o")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_title("Mean net return (bps/trade)")
    axes[0].set_xlabel("tau")
    axes[0].grid(alpha=0.3)

    axes[1].plot(taus, [r["sharpe_net"] for r in rows], marker="o", color="green")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("Sharpe (net, annualized)")
    axes[1].set_xlabel("tau")
    axes[1].grid(alpha=0.3)

    axes[2].plot(taus, [abs(r["max_drawdown_net"]) * 100 for r in rows], marker="o", color="orange")
    axes[2].set_title("Max drawdown (net, %)")
    axes[2].set_xlabel("tau")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"PnL backtest (w={window}, {tag}, pooled LOSO)", fontsize=12)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"backtest_pnl_w{window}_{tag}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PnL backtest with confidence filtering")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--tag", type=str, default="label_k12")
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="one-way fee rate (default 0.001 = 10 bps, round-trip 20 bps)",
    )
    parser.add_argument("--category", default=None, help="Research category glob prefix.")
    args = parser.parse_args()

    paths = _prediction_paths(args.window, args.tag, category=args.category)
    if not paths:
        print(
            f"No prediction files for window={args.window} tag={args.tag} "
            f"category={args.category or 'legacy'}"
        )
        return

    config = config_from_yaml()
    all_rows: list[TradeRow] = []
    for path in paths:
        all_rows.extend(_load_predictions(path))

    matched_rows, missing = attach_forward_returns(all_rows, config)
    if not matched_rows:
        print("No predictions matched to OHLCV forward returns.")
        return

    timestamps = pd.to_datetime([r.timestamp for r in matched_rows], utc=True)
    span_days = max((timestamps.max() - timestamps.min()).total_seconds() / 86400.0, 1.0)

    results = [
        backtest_tau(matched_rows, tau, args.fee_rate, span_days) for tau in THRESHOLDS
    ]

    out = {
        "window": args.window,
        "tag": args.tag,
        "fee_rate_one_way": args.fee_rate,
        "fee_rate_round_trip": 2 * args.fee_rate,
        "n_predictions": len(all_rows),
        "n_matched": len(matched_rows),
        "n_missing_returns": missing,
        "span_days": span_days,
        "thresholds": THRESHOLDS,
        "pooled": results,
    }
    out_path = METRICS_DIR / f"backtest_pnl_w{args.window}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PnL backtest (w={args.window}, tag={args.tag}, fee={args.fee_rate:.4f}/side)")
    print(f"predictions={len(all_rows)}, matched={len(matched_rows)}, span={span_days:.1f}d\n")
    print(format_table(results, args.fee_rate))
    chart = plot_backtest(results, args.tag, args.window)
    if chart:
        print(f"\nplot -> {chart}")
    print(f"json -> {out_path}")


if __name__ == "__main__":
    main()

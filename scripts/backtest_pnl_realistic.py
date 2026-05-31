"""Realistic PnL backtest on saved MLP predictions.

Closer to live trading vs scripts/backtest_pnl.py:
  - Execution: signal at bar t close -> enter t+1 open, exit t+1 close
  - Costs: fee + slippage per side (round-trip deducted per position)
  - Position cap: max concurrent positions, ranked by |prob - 0.5|
  - Capital: equal-weight among selected positions, compound equity curve
  - Sharpe: daily portfolio returns (15m bars aggregated to calendar days)

Example:
    python3 scripts/backtest_pnl_realistic.py --tag label_k12 --window 96
    python3 scripts/backtest_pnl_realistic.py --tag label_k12 --max-positions 3 --slippage 0.0005
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


@dataclass
class TradeRow:
    timestamp: pd.Timestamp
    symbol: str
    prob: float
    exec_log_return: float


@dataclass
class RealisticConfig:
    fee_rate: float = 0.001
    slippage: float = 0.0005
    spread_bps: float = 0.0
    entry_lag: int = 1
    random_slippage: bool = False
    slippage_mc_trials: int = 50
    max_positions: int = 3
    initial_capital: float = 1.0

    @property
    def one_way_cost(self) -> float:
        return self.fee_rate + self.slippage + self.spread_bps / 10_000.0

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * self.one_way_cost

    @property
    def notional_per_position(self) -> float:
        return self.initial_capital / max(self.max_positions, 1)


def _prediction_paths(
    window: int,
    tag: str,
    model: str = "mlp",
    category: str | None = None,
) -> list[Path]:
    from src.metrics_paths import list_test_prediction_paths

    return list_test_prediction_paths(METRICS_DIR, window, tag, model, category)


def _load_predictions(path: Path) -> list[tuple[pd.Timestamp, str, float]]:
    from src.metrics_paths import symbol_from_predictions_path

    symbol = symbol_from_predictions_path(path)
    rows: list[tuple[pd.Timestamp, str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append((pd.Timestamp(row["timestamp"]), symbol, float(row["prob_up"])))
    return rows


def _execution_return_lookup(
    symbol: str,
    config,
    entry_lag: int = 1,
) -> dict[pd.Timestamp, float]:
    """Log return from bar (t+entry_lag) open to same bar close; signal at bar t close."""
    df = load_ohlcv(config, symbol=symbol).sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    open_ = df["open"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    lag = max(1, int(entry_lag))
    out: dict[pd.Timestamp, float] = {}
    for i in range(len(df) - lag):
        o, c = open_[i + lag], close[i + lag]
        if not np.isfinite(o) or o <= 0 or not np.isfinite(c) or c <= 0:
            continue
        out[ts.iloc[i]] = float(np.log(c / o))
    return out


def load_matched_rows(
    config,
    paths: list[Path],
    entry_lag: int = 1,
) -> tuple[list[TradeRow], int]:
    cache: dict[str, dict[pd.Timestamp, float]] = {}
    matched: list[TradeRow] = []
    missing = 0
    for path in paths:
        for ts, symbol, prob in _load_predictions(path):
            if symbol not in cache:
                cache[symbol] = _execution_return_lookup(symbol, config, entry_lag=entry_lag)
            exec_log = cache[symbol].get(ts)
            if exec_log is None or not np.isfinite(exec_log):
                missing += 1
                continue
            matched.append(TradeRow(timestamp=ts, symbol=symbol, prob=prob, exec_log_return=exec_log))
    return matched, missing


def _select_signals(
    batch: list[TradeRow],
    tau: float,
    max_positions: int,
) -> list[tuple[TradeRow, int, float]]:
    """Return (row, direction, confidence) for accepted signals."""
    selected: list[tuple[TradeRow, int, float]] = []
    for row in batch:
        if row.prob >= tau:
            selected.append((row, 1, abs(row.prob - 0.5)))
        elif row.prob <= 1.0 - tau:
            selected.append((row, -1, abs(row.prob - 0.5)))
    selected.sort(key=lambda x: x[2], reverse=True)
    return selected[:max_positions]


def _net_position_return(
    direction: int,
    exec_log: float,
    cfg: RealisticConfig,
    rng: np.random.Generator | None = None,
) -> float:
    gross_simple = float(np.expm1(direction * exec_log))
    if cfg.random_slippage and rng is not None:
        slip = float(rng.uniform(0.0, 2.0 * cfg.slippage))
        cost = 2.0 * (cfg.fee_rate + slip) + 2.0 * (cfg.spread_bps / 10_000.0)
    else:
        cost = cfg.round_trip_cost
    return gross_simple - cost


def simulate_portfolio(
    rows: list[TradeRow],
    tau: float,
    cfg: RealisticConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int | list[float]]:
    if not rows:
        return {
            "n_trades": 0,
            "n_periods": 0,
            "coverage": 0.0,
            "period_pnl": [],
            "trade_returns": [],
            "timestamps": [],
        }

    by_ts: dict[pd.Timestamp, list[TradeRow]] = {}
    for row in rows:
        by_ts.setdefault(row.timestamp, []).append(row)

    period_pnl: list[float] = []
    trade_returns: list[float] = []
    timestamps: list[pd.Timestamp] = []
    n_trades = 0
    notional = cfg.notional_per_position

    for ts in sorted(by_ts.keys()):
        selected = _select_signals(by_ts[ts], tau, cfg.max_positions)
        if not selected:
            continue

        period_profit = 0.0
        for row, direction, _conf in selected:
            pos_ret = _net_position_return(direction, row.exec_log_return, cfg, rng=rng)
            trade_returns.append(pos_ret)
            period_profit += notional * pos_ret
            n_trades += 1

        period_pnl.append(period_profit)
        timestamps.append(ts)

    return {
        "n_trades": n_trades,
        "n_periods": len(period_pnl),
        "coverage": n_trades / len(rows),
        "period_pnl": period_pnl,
        "trade_returns": trade_returns,
        "timestamps": timestamps,
    }


def _equity_curve(period_pnl: list[float], cfg: RealisticConfig) -> np.ndarray:
    return cfg.initial_capital + np.cumsum(np.asarray(period_pnl, dtype=float))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity_curve)
    return float((equity_curve / peak - 1.0).min())


def _sharpe_daily(timestamps: list[pd.Timestamp], period_pnl: list[float], initial_capital: float) -> float:
    if len(period_pnl) < 2:
        return float("nan")
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamps, utc=True), "pnl": period_pnl}
    )
    daily_pnl = df.set_index("timestamp")["pnl"].resample("1D").sum()
    daily_ret = daily_pnl / initial_capital
    daily_ret = daily_ret[daily_ret != 0]
    if len(daily_ret) < 2:
        return float("nan")
    std = float(daily_ret.std(ddof=1))
    if std <= 0:
        return float("nan")
    return float(daily_ret.mean() / std * np.sqrt(365))


def backtest_tau(
    rows: list[TradeRow],
    tau: float,
    cfg: RealisticConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, float | int]:
    sim = simulate_portfolio(rows, tau, cfg, rng=rng)
    trade_returns = np.asarray(sim["trade_returns"], dtype=float)
    period_pnl = sim["period_pnl"]

    if len(trade_returns) == 0:
        return {
            "tau": tau,
            "n_trades": 0,
            "n_periods": 0,
            "coverage": 0.0,
            "mean_net_return_bps": float("nan"),
            "win_rate_net": float("nan"),
            "sharpe_daily": float("nan"),
            "max_drawdown_net": float("nan"),
            "total_return_net": float("nan"),
            "final_equity": float("nan"),
        }

    equity_curve = _equity_curve(period_pnl, cfg)
    final_equity = float(equity_curve[-1])

    return {
        "tau": tau,
        "n_trades": int(sim["n_trades"]),
        "n_periods": int(sim["n_periods"]),
        "coverage": float(sim["coverage"]),
        "mean_net_return_bps": float(np.mean(trade_returns) * 10_000),
        "win_rate_net": float(np.mean(trade_returns > 0)),
        "sharpe_daily": _sharpe_daily(sim["timestamps"], period_pnl, cfg.initial_capital),
        "max_drawdown_net": _max_drawdown(equity_curve),
        "total_return_net": float(final_equity / cfg.initial_capital - 1.0),
        "final_equity": final_equity,
    }


def format_table(rows: list[dict[str, float | int]], cfg: RealisticConfig) -> str:
    lines = [
        "Realistic assumptions:",
        f"  execution: signal@t close -> enter t+{cfg.entry_lag} open, exit same bar close",
        f"  fee={cfg.fee_rate:.4f}/side, slippage={cfg.slippage:.4f}/side, "
        f"round-trip={(cfg.round_trip_cost * 10_000):.1f} bps",
        f"  max concurrent positions={cfg.max_positions}, notional={cfg.notional_per_position:.3f}/pos",
        f"  capital model=fixed-notional (no compounding, max deploy={cfg.max_positions * cfg.notional_per_position:.2f})",
        "",
        f"{'tau':>6} {'cov':>8} {'trades':>8} {'net_bps':>10} "
        f"{'win_rate':>10} {'Sharpe_d':>9} {'maxDD':>8} {'total_ret':>10}",
    ]
    lines.append("-" * 88)
    for row in rows:
        lines.append(
            f"{row['tau']:>6.2f} "
            f"{row['coverage']:>8.3f} "
            f"{row['n_trades']:>8} "
            f"{row['mean_net_return_bps']:>10.2f} "
            f"{row['win_rate_net']:>10.4f} "
            f"{row['sharpe_daily']:>9.2f} "
            f"{row['max_drawdown_net']:>8.3f} "
            f"{row['total_return_net']:>10.3f}"
        )
    return "\n".join(lines)


def plot_backtest(rows: list[dict[str, float | int]], tag: str, window: int, cfg: RealisticConfig) -> Path | None:
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

    axes[1].plot(taus, [r["total_return_net"] * 100 for r in rows], marker="o", color="green")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title("Total return (%)")
    axes[1].set_xlabel("tau")
    axes[1].grid(alpha=0.3)

    axes[2].plot(taus, [abs(r["max_drawdown_net"]) * 100 for r in rows], marker="o", color="orange")
    axes[2].set_title("Max drawdown (%)")
    axes[2].set_xlabel("tau")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Realistic backtest (w={window}, {tag}, max_pos={cfg.max_positions})",
        fontsize=12,
    )
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"backtest_realistic_w{window}_{tag}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _tau_from_val_predictions(val_path: Path) -> float:
    from sklearn.metrics import matthews_corrcoef

    if not val_path.exists():
        return 0.70
    y_true, prob = [], []
    with val_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["y_true"]))
            prob.append(float(row["prob_up"]))
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    best_tau, best_score = 0.70, -1.0
    for tau in THRESHOLDS:
        mask = (p >= tau) | (p <= 1.0 - tau)
        if mask.sum() < 10:
            continue
        pred = (p >= tau).astype(int)
        try:
            score = matthews_corrcoef(y[mask], pred[mask])
        except ValueError:
            score = 0.0
        if score > best_score:
            best_score = float(score)
            best_tau = float(tau)
    return best_tau


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic PnL backtest")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--tag", type=str, default="label_k12")
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="one-way fee (default 10 bps)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="one-way slippage (default 5 bps)")
    parser.add_argument("--spread-bps", type=float, default=0.0, help="extra one-way spread (bps)")
    parser.add_argument("--entry-lag", type=int, default=1, help="bars after signal to enter (1=t+1)")
    parser.add_argument("--random-slippage", action="store_true")
    parser.add_argument("--slippage-mc-trials", type=int, default=50)
    parser.add_argument("--per-round-tau", action="store_true", help="pick tau per held-out val set")
    parser.add_argument("--max-positions", type=int, default=3, help="max concurrent positions")
    parser.add_argument("--category", default=None, help="Research category glob prefix.")
    args = parser.parse_args()

    cfg = RealisticConfig(
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        spread_bps=args.spread_bps,
        entry_lag=args.entry_lag,
        random_slippage=args.random_slippage,
        slippage_mc_trials=args.slippage_mc_trials,
        max_positions=args.max_positions,
    )

    paths = _prediction_paths(
        args.window, args.tag, args.model, category=args.category
    )
    if not paths:
        print(f"No prediction files for w={args.window} tag={args.tag} model={args.model}")
        return

    config = config_from_yaml()
    matched_rows, missing = load_matched_rows(config, paths, entry_lag=cfg.entry_lag)
    if not matched_rows:
        print("No predictions matched to execution returns.")
        return

    timestamps = pd.to_datetime([r.timestamp for r in matched_rows], utc=True)
    span_days = max((timestamps.max() - timestamps.min()).total_seconds() / 86400.0, 1.0)

    results = [backtest_tau(matched_rows, tau, cfg) for tau in THRESHOLDS]

    mc_sharpes: dict[float, list[float]] = {tau: [] for tau in THRESHOLDS}
    if cfg.random_slippage:
        for trial in range(cfg.slippage_mc_trials):
            rng = np.random.default_rng(trial)
            for tau in THRESHOLDS:
                row = backtest_tau(matched_rows, tau, cfg, rng=rng)
                if np.isfinite(row["sharpe_daily"]):
                    mc_sharpes[tau].append(float(row["sharpe_daily"]))

    per_round = []
    if args.per_round_tau:
        for path in paths:
            val_path = path.with_name(path.name.replace("_predictions.csv", "_val_predictions.csv"))
            tau = _tau_from_val_predictions(val_path)
            matched, _ = load_matched_rows(config, [path], entry_lag=cfg.entry_lag)
            if matched:
                row = backtest_tau(matched, tau, cfg)
                from src.metrics_paths import symbol_from_predictions_path

                row["symbol"] = symbol_from_predictions_path(path)
                row["tau_used"] = tau
                per_round.append(row)

    exec_desc = f"signal@t close -> enter t+{cfg.entry_lag} open, exit same bar close"
    out = {
        "mode": "realistic",
        "window": args.window,
        "tag": args.tag,
        "model": args.model,
        "assumptions": {
            "execution": exec_desc,
            "fee_rate_one_way": cfg.fee_rate,
            "slippage_one_way": cfg.slippage,
            "spread_bps_one_way": cfg.spread_bps,
            "entry_lag": cfg.entry_lag,
            "random_slippage": cfg.random_slippage,
            "round_trip_cost_bps": cfg.round_trip_cost * 10_000,
            "max_positions": cfg.max_positions,
            "notional_per_position": cfg.notional_per_position,
            "position_selection": "top |prob-0.5| when over cap",
            "capital": "fixed notional per position, additive equity curve (no compounding)",
            "sharpe": "daily PnL / initial_capital, sqrt(365)",
        },
        "n_predictions": len(matched_rows) + missing,
        "n_matched": len(matched_rows),
        "n_missing_returns": missing,
        "span_days": span_days,
        "thresholds": THRESHOLDS,
        "pooled": results,
        "per_round": per_round,
    }
    if cfg.random_slippage:
        out["sharpe_mc"] = {
            str(tau): {
                "median": float(np.median(v)) if v else float("nan"),
                "p05": float(np.quantile(v, 0.05)) if v else float("nan"),
                "p95": float(np.quantile(v, 0.95)) if v else float("nan"),
            }
            for tau, v in mc_sharpes.items()
            if tau == 0.70
        }
    suffix = f"_lag{cfg.entry_lag}" if cfg.entry_lag != 1 else ""
    if cfg.spread_bps > 0:
        suffix += f"_spr{int(cfg.spread_bps)}"
    out_path = METRICS_DIR / f"backtest_realistic_w{args.window}_{args.tag}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Realistic PnL backtest (w={args.window}, tag={args.tag})")
    print(f"predictions={len(matched_rows)}, span={span_days:.1f}d, missing={missing}\n")
    print(format_table(results, cfg))
    chart = plot_backtest(results, args.tag, args.window, cfg)
    if chart:
        print(f"\nplot -> {chart}")
    print(f"json -> {out_path}")


if __name__ == "__main__":
    main()

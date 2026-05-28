"""Figures for shared-regime hypothesis: correlation heatmaps and protocol timeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_from_yaml  # noqa: E402
from src.data import load_ohlcv  # noqa: E402
from src.features import add_indicators  # noqa: E402
from src.multi import build_per_symbol, compute_calendar_boundaries  # noqa: E402

PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
SYMBOLS = [
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "BOMEUSDT",
    "1000SATSUSDT",
]
SHORT = {
    "DOGEUSDT": "DOGE",
    "SHIBUSDT": "SHIB",
    "PEPEUSDT": "PEPE",
    "WIFUSDT": "WIF",
    "BONKUSDT": "BONK",
    "FLOKIUSDT": "FLOKI",
    "BOMEUSDT": "BOME",
    "1000SATSUSDT": "1000SATS",
    "BTCUSDT": "BTC",
}

plt.rcParams["font.sans-serif"] = [
    "AR PL UMing CN",
    "Droid Sans Fallback",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _aligned_returns(config) -> pd.DataFrame:
    """Inner-join 15m log returns across meme symbols on timestamp."""
    series: dict[str, pd.Series] = {}
    for sym in SYMBOLS:
        df = load_ohlcv(config, symbol=sym).sort_values("timestamp")
        close = df["close"].astype(float)
        ret = np.log(close / close.shift(1))
        ts = pd.to_datetime(df["timestamp"], utc=True)
        series[SHORT[sym]] = pd.Series(ret.values, index=ts, name=SHORT[sym])
    panel = pd.concat(series, axis=1, join="inner").dropna(how="any")
    return panel


def _btc_vol_regime(config, panel_index: pd.DatetimeIndex) -> pd.Series:
    df = load_ohlcv(config, symbol="BTCUSDT").sort_values("timestamp")
    df = add_indicators(df, config)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    vol = pd.Series(df["volatility_50"].astype(float).values, index=ts)
    vol = vol.reindex(panel_index)
    valid = vol.dropna()
    q33, q66 = valid.quantile(0.33), valid.quantile(0.66)
    regime = pd.Series("medium", index=panel_index, dtype=object)
    regime[vol <= q33] = "low"
    regime[vol >= q66] = "high"
    regime[vol.isna()] = "medium"
    return regime


def plot_correlation_heatmaps(config) -> dict:
    panel = _aligned_returns(config)
    regime = _btc_vol_regime(config, panel.index)

    corr_all = panel.corr()
    high_mask = regime == "high"
    low_mask = regime == "low"
    corr_high = panel.loc[high_mask].corr() if high_mask.sum() > 100 else corr_all * np.nan
    corr_low = panel.loc[low_mask].corr() if low_mask.sum() > 100 else corr_all * np.nan

    # off-diagonal mean correlation
    def _mean_offdiag(c: pd.DataFrame) -> float:
        m = c.values.copy()
        np.fill_diagonal(m, np.nan)
        return float(np.nanmean(m))

    stats = {
        "n_bars": int(len(panel)),
        "date_start": str(panel.index.min().date()),
        "date_end": str(panel.index.max().date()),
        "mean_pairwise_corr_all": round(_mean_offdiag(corr_all), 4),
        "mean_pairwise_corr_high_vol": round(_mean_offdiag(corr_high), 4),
        "mean_pairwise_corr_low_vol": round(_mean_offdiag(corr_low), 4),
        "n_high_vol": int(high_mask.sum()),
        "n_low_vol": int(low_mask.sum()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), dpi=150)
    vmin, vmax = 0.0, 1.0
    titles = [
        f"全样本（$n$={len(panel):,}）\n平均配对相关={stats['mean_pairwise_corr_all']:.3f}",
        f"BTC 高波动（上 33\\%）\n平均配对相关={stats['mean_pairwise_corr_high_vol']:.3f}",
        f"BTC 低波动（下 33\\%）\n平均配对相关={stats['mean_pairwise_corr_low_vol']:.3f}",
    ]
    mats = [corr_all, corr_high, corr_low]
    labels = list(corr_all.columns)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat.values, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{mat.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.35 < mat.values[i, j] < 0.85 else "white",
                )
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Meme 币 15 分钟对数收益横截面相关（时间对齐内连接）", fontsize=11, y=1.02)
    fig.tight_layout()
    out = PAPER_FIGURES / "meme_return_correlation.png"
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_METRICS / "meme_return_correlation.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out}")
    return stats


def _calendar_dates(config) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    per = build_per_symbol(SYMBOLS, config)
    train_end_s, val_end_s = compute_calendar_boundaries(per, SYMBOLS, config)
    t_train = pd.Timestamp(train_end_s)
    t_val = pd.Timestamp(val_end_s)
    all_ts = []
    for sym in SYMBOLS:
        if sym in per:
            all_ts.extend(pd.to_datetime(per[sym][2]).tolist())
    t_min = pd.Timestamp(min(all_ts))
    t_max = pd.Timestamp(max(all_ts))
    return t_min, t_train, t_val, t_max


def plot_protocol_timeline(config) -> None:
    t_min, t_train, t_val, t_max = _calendar_dates(config)
    held = "PEPE"

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 4.8), dpi=150, sharex=True)
    y_labels = ["开发协议（ratio-LOSO）", "严格协议（calendar WF）"]
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B", "overlap": "#E45756"}

    # Panel 0: ratio-LOSO
    ax = axes[0]
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([y_labels[0]])
    # train pool (7 coins) 85% of pooled time ~ use full span as train visual
    ax.barh(0.5, (t_max - t_min).total_seconds(), left=t_min, height=0.35, color=colors["train"], alpha=0.85, label="训练池（7 币，池化前 85% 时间）")
    ax.barh(0.5, (t_max - t_min).total_seconds(), left=t_min, height=0.18, color=colors["overlap"], alpha=0.9, label=f"测试：held-out {held}（全时段，与训练池日历重叠）")
    ax.annotate(
        "日历重叠 → 共享市场冲击同时作用于训练池与 held-out",
        xy=(t_min + (t_max - t_min) * 0.5, 0.82),
        ha="center",
        fontsize=9,
        color=colors["overlap"],
    )

    # Panel 1: strict WF
    ax = axes[1]
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([y_labels[1]])
    ax.barh(0.5, (t_train - t_min).total_seconds(), left=t_min, height=0.35, color=colors["train"], alpha=0.85)
    ax.barh(0.5, (t_val - t_train).total_seconds(), left=t_train, height=0.35, color=colors["val"], alpha=0.85, label="验证（7 币，70%--85%）")
    ax.barh(0.5, (t_max - t_val).total_seconds(), left=t_val, height=0.35, color=colors["test"], alpha=0.85, label=f"测试：仅 {held}（$>$85%，$\max train < \min test$）")
    ax.axvline(t_train, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(t_val, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.text(t_train, 0.92, r"$T_{\mathrm{train}}$ (70\%)", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.text(t_val, 0.92, r"$T_{\mathrm{val}}$ (85\%)", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.annotate(
        "无日历重叠 → 切断共享时段排序通道",
        xy=(t_val + (t_max - t_val) * 0.5, 0.82),
        ha="center",
        fontsize=9,
        color=colors["test"],
    )

    for ax in axes:
        ax.set_xlim(t_min, t_max)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)

    import matplotlib.dates as mdates

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=25)

    handles = [
        mpatches.Patch(color=colors["train"], label="训练（非 held-out 币）"),
        mpatches.Patch(color=colors["val"], label="验证"),
        mpatches.Patch(color=colors["test"], label="测试（held-out）"),
        mpatches.Patch(color=colors["overlap"], label="held-out 全时段（开发协议重叠）"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"LOSO 时间切分示意（样本期：{t_min.date()} — {t_max.date()}，held-out 示例：{held}）",
        fontsize=11,
        y=1.14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PAPER_FIGURES / "loso_protocol_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    config = config_from_yaml(PROJECT_ROOT / "config" / "default.yaml")
    stats = plot_correlation_heatmaps(config)
    plot_protocol_timeline(config)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

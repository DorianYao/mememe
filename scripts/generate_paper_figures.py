"""Generate figures for the LaTeX paper under paper/figures/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

plt.rcParams["font.sans-serif"] = [
    "AR PL UMing CN",
    "Droid Sans Fallback",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir() -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)


def _load_summary(name: str) -> dict:
    path = METRICS_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def plot_data_exploration() -> None:
    """Missing values, label distribution, feature histograms."""
    symbols = [
        "DOGEUSDT",
        "SHIBUSDT",
        "PEPEUSDT",
        "WIFUSDT",
        "BONKUSDT",
        "FLOKIUSDT",
        "BOMEUSDT",
        "1000SATSUSDT",
    ]
    frames = []
    for sym in symbols:
        path = RAW_DIR / f"{sym}_15m_720d.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        print("No raw CSV found; skip data exploration.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    missing_pct = all_df[numeric_cols].isna().mean() * 100
    completeness_pct = 100.0 - missing_pct

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # 缺失率为 0 时横条不可见，改绘完整率（100%）并标注缺失率
    axes[0].barh(numeric_cols, completeness_pct, color="#4C72B0", height=0.65)
    axes[0].set_xlabel("完整率 (%)")
    axes[0].set_title("原始 K 线字段缺失值检测\n（8 币合计，缺失率均为 0%）")
    axes[0].set_xlim(95, 100.5)
    axes[0].axvline(100, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    for i, comp in enumerate(completeness_pct):
        axes[0].text(comp + 0.05, i, f"{comp:.0f}%", va="center", ha="left", fontsize=7)

    k = 0.3
    up = down = skip = 0
    for sym in symbols:
        sub = all_df[all_df["symbol"] == sym].copy()
        sub["log_return"] = np.log(sub["close"] / sub["close"].shift(1))
        sub["volatility_50"] = sub["log_return"].rolling(50).std()
        sub["next_return"] = sub["log_return"].shift(-1)
        valid = sub[["log_return", "volatility_50", "next_return"]].dropna()
        up += int((valid["next_return"] > k * valid["volatility_50"]).sum())
        down += int((valid["next_return"] < -k * valid["volatility_50"]).sum())
        skip += int(
            (
                (valid["next_return"].abs() <= k * valid["volatility_50"])
            ).sum()
        )
    labels = ["涨 (1)", "跌 (0)", "丢弃 (横盘)"]
    counts = [int(up), int(down), int(skip)]
    colors = ["#55A868", "#C44E52", "#BBBBBB"]
    axes[1].bar(labels, counts, color=colors)
    axes[1].set_title("动态波动率标签分布 (k=0.3, 8 币合计)")
    axes[1].set_ylabel("K 线根数")
    for i, c in enumerate(counts):
        axes[1].text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)

    sample = all_df[all_df["symbol"] == "DOGEUSDT"].copy()
    sample["log_return"] = np.log(sample["close"] / sample["close"].shift(1))
    sample = sample.dropna(subset=["log_return"])
    axes[2].hist(sample["log_return"], bins=80, color="#8172B3", alpha=0.85, edgecolor="white")
    axes[2].set_title("DOGEUSDT 对数收益分布")
    axes[2].set_xlabel("log_return")
    axes[2].set_ylabel("频数")
    axes[2].axvline(0, color="red", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out = PAPER_FIGURES / "data_exploration.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_training_curve() -> None:
    history_path = METRICS_DIR / (
        "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w192_loso_"
        "PEPEUSDT_label_k12_mlp_history.json"
    )
    if not history_path.exists():
        history_path = METRICS_DIR / (
            "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w96_loso_"
            "PEPEUSDT_label_k10_mlp_history.json"
        )
    if not history_path.exists():
        history_path = next(METRICS_DIR.glob("*_mlp_history.json"))
    tag = "w192" if "w192" in history_path.name else "w96"
    with history_path.open("r", encoding="utf-8") as fh:
        history = json.load(fh)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="训练集", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"], label="验证集", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title(f"MLP 训练/验证损失 ({tag}, PEPE held-out)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="训练 Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="验证 Accuracy")
    axes[1].plot(epochs, history["train_macro_f1"], label="训练 Macro-F1", linestyle="--")
    axes[1].plot(epochs, history["val_macro_f1"], label="验证 Macro-F1", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("指标值")
    axes[1].set_ylim(0.45, 0.80)
    axes[1].set_title("MLP 准确率与 Macro-F1")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = PAPER_FIGURES / "training_curve.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_confusion_matrix() -> None:
    metrics_path = METRICS_DIR / (
        "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w192_loso_"
        "PEPEUSDT_label_k12_mlp_metrics.json"
    )
    if not metrics_path.exists():
        metrics_path = METRICS_DIR / (
            "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w96_loso_"
            "PEPEUSDT_label_k10_mlp_metrics.json"
        )
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    cm = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测跌 (0)", "预测涨 (1)"])
    ax.set_yticklabels(["真实跌 (0)", "真实涨 (1)"])
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1f}%)", ha="center", va="center", fontsize=10)
    ax.set_title(
        f"混淆矩阵 (PEPE held-out, AUC={metrics['test_roc_auc']:.3f}, "
        f"MCC={metrics['test_mcc']:.3f})"
    )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = PAPER_FIGURES / "confusion_matrix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _load_summary_by_pattern(w: int, tag: str = "") -> dict | None:
    pattern = f"meme*_w{w}_loso"
    if tag:
        pattern += f"_{tag}"
    paths = sorted(METRICS_DIR.glob(f"{pattern}_summary.json"))
    if not paths:
        return None
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _held_out_auc(summary: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, row in summary.get("per_round", {}).items():
        out[sym.replace("USDT", "")] = row["mlp"]["test_roc_auc"]
    return out


def _mean_mlp_auc(summary: dict) -> float:
    return summary["mean"]["mlp"]["mean_test_roc_auc"]


def _mean_mlp_f1(summary: dict) -> float:
    return summary["mean"]["mlp"]["mean_test_macro_f1"]


def plot_label_k_sensitivity() -> None:
    scans = [
        (0.1, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k01_summary.json"),
        (0.2, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k02_summary.json"),
        (0.3, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_summary.json"),
        (0.5, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k05_summary.json"),
        (0.7, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k07_summary.json"),
        (1.0, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k10_summary.json"),
    ]
    ks, aucs, f1s = [], [], []
    for k, fname in scans:
        summary = _load_summary(fname)
        ks.append(k)
        aucs.append(_mean_mlp_auc(summary))
        f1s.append(_mean_mlp_f1(summary))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, aucs, "o-", linewidth=2, label="ROC-AUC", color="#4C72B0")
    ax.plot(ks, f1s, "s--", linewidth=2, label="Macro-F1", color="#C44E52")
    best_idx = int(np.argmax(aucs))
    ax.scatter([ks[best_idx]], [aucs[best_idx]], s=120, c="gold", edgecolors="black", zorder=5)
    ax.annotate(
        f"最优 k={ks[best_idx]}",
        (ks[best_idx], aucs[best_idx]),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=9,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, label="随机 AUC=0.5")
    ax.set_xlabel("标签阈值系数 k")
    ax.set_ylabel("LOSO 8 轮平均")
    ax.set_title("超参数敏感性：动态标签阈值 k (MLP, w=20)")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "sensitivity_label_k.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_window_sensitivity() -> None:
    scans = [
        (20, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_label_k10_summary.json"),
        (40, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w40_loso_label_k10_summary.json"),
        (60, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w60_loso_label_k10_summary.json"),
        (96, "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w96_loso_label_k10_summary.json"),
    ]
    ws, aucs, f1s = [], [], []
    for w, fname in scans:
        summary = _load_summary(fname)
        ws.append(w)
        aucs.append(_mean_mlp_auc(summary))
        f1s.append(_mean_mlp_f1(summary))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ws, aucs, "o-", linewidth=2, label="ROC-AUC", color="#4C72B0")
    ax.plot(ws, f1s, "s--", linewidth=2, label="Macro-F1", color="#C44E52")
    best_idx = int(np.argmax(aucs))
    ax.scatter([ws[best_idx]], [aucs[best_idx]], s=120, c="gold", edgecolors="black", zorder=5)
    ax.annotate(
        f"最优 w={ws[best_idx]}",
        (ws[best_idx], aucs[best_idx]),
        textcoords="offset points",
        xytext=(10, 8),
        fontsize=9,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xlabel("输入窗口长度 (15m K 线根数)")
    ax.set_ylabel("LOSO 8 轮平均")
    ax.set_title("超参数敏感性：窗口长度 w (MLP, k=1.0)")
    ax.set_xticks(ws)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "sensitivity_window.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_baseline_comparison() -> None:
    summary = _load_summary(
        "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_summary.json"
    )
    mlp = summary["mean"]["mlp"]
    cnn = summary["mean"]["cnn"]
    names = ["多数类\nbaseline", "随机\n猜测", "MLP", "1D CNN"]
    auc = [0.50, 0.50, mlp["mean_test_roc_auc"], cnn["mean_test_roc_auc"]]
    f1 = [0.34, 0.50, mlp["mean_test_macro_f1"], cnn["mean_test_macro_f1"]]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, auc, width, label="ROC-AUC", color="#4C72B0")
    ax.bar(x + width / 2, f1, width, label="Macro-F1", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.75)
    ax.set_title("LOSO 主实验：模型与 Baseline 对比 (k=0.3, w=20)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "baseline_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ablation_summary() -> None:
    ablations = [
        ("Baseline", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_summary.json"),
        ("去 volume_z", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_no_volume_z_summary.json"),
        ("raw volume", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_raw_volume_summary.json"),
        ("去技术指标", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_no_indicators_summary.json"),
        ("固定标签", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_fixed_label_summary.json"),
    ]
    labels, aucs = [], []
    for label, fname in ablations:
        summary = _load_summary(fname)
        labels.append(label)
        aucs.append(_mean_mlp_auc(summary))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4C72B0" if i == 0 else "#DD8452" for i in range(len(labels))]
    bars = ax.bar(labels, aucs, color=colors)
    ax.axhline(aucs[0], color="#4C72B0", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("MLP mean ROC-AUC")
    ax.set_title("特征与标签消融实验 (LOSO 8 轮平均)")
    ax.set_ylim(0.555, 0.572)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005, f"{v:.4f}", ha="center", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "ablation_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pruned_kw_heatmap() -> None:
    """k×window pruned scan heatmap (MLP AUC)."""
    matrix = [
        (0.5, 40, "label_k05"),
        (0.5, 80, "label_k05"),
        (0.7, 40, "label_k07"),
        (0.7, 80, "label_k07"),
        (1.0, 40, "label_k10"),
        (1.0, 80, "label_k10"),
        (1.0, 96, "label_k10"),
        (1.2, 96, "label_k12"),
    ]
    ks = sorted({k for k, _, _ in matrix})
    ws = sorted({w for _, w, _ in matrix})
    grid = np.full((len(ks), len(ws)), np.nan)
    for k, w, tag in matrix:
        paths = sorted(METRICS_DIR.glob(f"meme*_w{w}_loso_{tag}_summary.json"))
        if not paths:
            continue
        summary = json.loads(paths[0].read_text(encoding="utf-8"))
        ki, wi = ks.index(k), ws.index(w)
        grid[ki, wi] = summary["mean"]["mlp"]["mean_test_roc_auc"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=0.55, vmax=0.75)
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("窗口长度 w (15m K 线根数)")
    ax.set_ylabel("标签阈值系数 k")
    ax.set_title("k×window 定向精扫：MLP LOSO 平均 ROC-AUC")
    for i in range(len(ks)):
        for j in range(len(ws)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="ROC-AUC")
    fig.tight_layout()
    out = PAPER_FIGURES / "pruned_kw_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_window_extended_scan() -> None:
    """Extended + fine window scan at k=1.2."""
    scans = [
        (96, "label_k12"),
        (128, "label_k12"),
        (160, "label_k12"),
        (176, "label_k12"),
        (184, "label_k12"),
        (192, "label_k12"),
        (200, "label_k12"),
        (208, "label_k12"),
    ]
    ws, aucs, stds = [], [], []
    for w, tag in scans:
        paths = sorted(METRICS_DIR.glob(f"meme*_w{w}_loso_{tag}_summary.json"))
        if not paths:
            continue
        summary = json.loads(paths[0].read_text(encoding="utf-8"))
        m = summary["mean"]["mlp"]
        ws.append(w)
        aucs.append(m["mean_test_roc_auc"])
        stds.append(m.get("std_test_roc_auc", 0))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(ws, aucs, yerr=stds, fmt="o-", linewidth=2, capsize=4, color="#4C72B0")
    best_idx = int(np.argmax(aucs))
    ax.scatter([ws[best_idx]], [aucs[best_idx]], s=140, c="gold", edgecolors="black", zorder=5)
    ax.annotate(
        f"峰值 w={ws[best_idx]}\nAUC={aucs[best_idx]:.4f}",
        (ws[best_idx], aucs[best_idx]),
        textcoords="offset points",
        xytext=(12, -18),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    ax.axhline(0.5, color="#C44E52", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("窗口长度 w（15m K 线根数）")
    ax.set_ylabel("LOSO 8 轮平均 ROC-AUC")
    ax.set_title("长窗口扩展扫描（k=1.2，含细扫 w=176–208）")
    ax.set_ylim(0.48, 0.78)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "window_extended_scan.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_confidence_threshold() -> None:
    path = METRICS_DIR / "confidence_threshold_w192_label_k12.json"
    if not path.exists():
        path = METRICS_DIR / "confidence_threshold_w96_label_k12.json"
    if not path.exists():
        print("Missing confidence threshold JSON; skip.")
        return
    window = 192 if "w192" in path.name else 96
    data = json.loads(path.read_text(encoding="utf-8"))
    pooled = data["pooled"]
    taus = [r["tau"] for r in pooled]
    mcc = [r["mcc"] for r in pooled]
    cov = [r["coverage"] * 100 for r in pooled]
    acc = [r["accuracy"] for r in pooled]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(taus, mcc, "o-", linewidth=2, color="#4C72B0", label="MCC")
    ax1.plot(taus, acc, "s--", linewidth=2, color="#55A868", label="Accuracy")
    ax1.axvline(0.70, color="gray", linestyle=":", linewidth=0.9, label="τ=0.70")
    ax1.set_xlabel("置信度阈值 τ")
    ax1.set_ylabel("分类指标")
    ax1.set_title(f"置信度过滤（k=1.2, w={window}）")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(taus, cov, width=0.03, alpha=0.25, color="#C44E52", label="Coverage (%)")
    ax2.set_ylabel("覆盖率 (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = PAPER_FIGURES / "confidence_threshold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_backtest_pnl() -> None:
    path = METRICS_DIR / "backtest_realistic_w192_label_k12.json"
    if not path.exists():
        path = METRICS_DIR / "backtest_realistic_w96_label_k12.json"
    if not path.exists():
        print("Missing backtest JSON; skip.")
        return
    window = 192 if "w192" in path.name else 96
    data = json.loads(path.read_text(encoding="utf-8"))
    pooled = data["pooled"]
    taus = [r["tau"] for r in pooled]
    ret = [r["total_return_net"] * 100 for r in pooled]
    sharpe = [r["sharpe_daily"] for r in pooled]
    dd = [abs(r["max_drawdown_net"]) * 100 for r in pooled]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(taus, ret, "o-", linewidth=2, color="#4C72B0")
    axes[0].axvline(0.70, color="gray", linestyle=":", linewidth=0.9)
    axes[0].set_xlabel("置信度阈值 τ")
    axes[0].set_ylabel("2 年总收益 (%)")
    axes[0].set_title(f"实盘约束 PnL（w={window}，30 bps 往返）")
    axes[0].grid(alpha=0.3)

    axes[1].plot(taus, sharpe, "s-", linewidth=2, color="#55A868", label="Sharpe(日)")
    axes[1].plot(taus, dd, "^--", linewidth=2, color="#C44E52", label="MaxDD (%)")
    axes[1].axvline(0.70, color="gray", linestyle=":", linewidth=0.9)
    axes[1].set_xlabel("置信度阈值 τ")
    axes[1].set_title("风险调整指标")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "backtest_pnl.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_milestone_evolution() -> None:
    milestones = [
        ("BTC\n固定标签", 0.5361),
        ("LOSO\nbaseline", 0.5664),
        ("k=0.5\nw=20", 0.5708),
        ("k=1.0\nw=96", 0.6372),
        ("k=1.2\nw=96", 0.7156),
        ("k=1.2\nw=192", 0.7434),
    ]
    labels = [m[0] for m in milestones]
    aucs = [m[1] for m in milestones]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#BBBBBB", "#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#C44E52"]
    bars = ax.bar(range(len(labels)), aucs, color=colors, zorder=2)
    ax.axhline(
        0.5,
        color="#C44E52",
        linestyle="--",
        linewidth=1.6,
        label="随机猜测 (AUC=0.5)",
        zorder=3,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MLP LOSO 平均 ROC-AUC")
    ax.set_title("性能演进：问题定义与标签—窗口协同优化")
    ax.set_ylim(0.48, 0.78)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "milestone_evolution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


    print(f"Saved {out}")


def plot_heldout_auc_bars() -> None:
    summary = _load_summary_by_pattern(192, "label_k12")
    if summary is None:
        print("Missing w192 summary; skip heldout bars.")
        return
    aucs = _held_out_auc(summary)
    syms = sorted(aucs, key=aucs.get, reverse=True)
    vals = [aucs[s] for s in syms]
    colors = ["#C44E52" if v < 0.65 else "#4C72B0" for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(syms))
    ax.barh(y, vals, color=colors, height=0.65)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.0, label="随机 0.5")
    ax.axvline(np.mean(vals), color="#55A868", linestyle="-.", linewidth=1.2,
               label=f"均值 {np.mean(vals):.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(syms)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("各 held-out 币种 AUC（k=1.2, w=192, MLP）")
    ax.set_xlim(0.48, 0.88)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "heldout_auc_w192.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_heldout_w96_vs_w192() -> None:
    s96 = _load_summary_by_pattern(96, "label_k12")
    s192 = _load_summary_by_pattern(192, "label_k12")
    if s96 is None or s192 is None:
        print("Missing w96/w192 summary; skip heldout compare.")
        return
    a96 = _held_out_auc(s96)
    a192 = _held_out_auc(s192)
    syms = sorted(set(a96) | set(a192), key=lambda s: a192.get(s, 0), reverse=True)
    x = np.arange(len(syms))
    bw = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - bw / 2, [a96.get(s, np.nan) for s in syms], bw, label="w=96 (24h)", color="#8172B3")
    ax.bar(x + bw / 2, [a192.get(s, np.nan) for s in syms], bw, label="w=192 (48h)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(syms, rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("held-out 币种：w=96 vs w=192 对比（k=1.2）")
    ax.set_ylim(0.48, 0.88)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "heldout_w96_vs_w192.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ablation_delta() -> None:
    ablations = [
        ("去 volume_z", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_no_volume_z_summary.json"),
        ("raw volume", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_raw_volume_summary.json"),
        ("去技术指标", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_no_indicators_summary.json"),
        ("固定标签", "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_fixed_label_summary.json"),
    ]
    baseline = _mean_mlp_auc(_load_summary(
        "meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso_summary.json"
    ))
    labels, deltas = [], []
    for name, fname in ablations:
        auc = _mean_mlp_auc(_load_summary(fname))
        labels.append(name)
        deltas.append((auc - baseline) * 100)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#C44E52" if d < -0.2 else "#DD8452" if d < 0 else "#55A868" for d in deltas]
    bars = ax.barh(labels, deltas, color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔAUC (pp) 相对 Baseline")
    ax.set_title("消融实验：相对 Baseline 的 AUC 变化")
    for bar, d in zip(bars, deltas):
        ax.text(d + (0.02 if d >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "ablation_delta.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_label_k_tradeoff() -> None:
    scans = [
        (0.1, "label_k01", 486511), (0.2, "label_k02", 448115), (0.3, None, 403804),
        (0.5, "label_k05", 316984), (0.7, "label_k07", 239838), (1.0, "label_k10", 152548),
    ]
    ks, aucs, ns = [], [], []
    for k, tag_suffix, n_fb in scans:
        fname = (
            f"meme8_DOGE-SHIB-PEPE-WIFU-BONK-FLOK-BOME-1000_15m_w20_loso"
            f"{'_' + tag_suffix if tag_suffix else ''}_summary.json"
        )
        ks.append(k)
        aucs.append(_mean_mlp_auc(_load_summary(fname)))
        ns.append(n_fb)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    ax1.plot(ks, aucs, "o-", linewidth=2, color="#4C72B0", label="ROC-AUC")
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    ax1.set_xlabel("标签阈值 k")
    ax1.set_ylabel("LOSO 平均 ROC-AUC", color="#4C72B0")
    ax1.set_xticks(ks)
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(ks, [n / 1000 for n in ns], width=0.08, alpha=0.25, color="#C44E52", label="样本量 (×10³)")
    ax2.set_ylabel("保留样本数 (×10³)", color="#C44E52")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=8)
    ax1.set_title("k 扫描：AUC 与样本保留率权衡（w=20）")
    fig.tight_layout()
    out = PAPER_FIGURES / "label_k_tradeoff.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mlp_vs_cnn_window() -> None:
    ws, mlp_auc, cnn_auc = [], [], []
    for w in (20, 40, 60, 96):
        summary = _load_summary_by_pattern(w, "label_k10")
        if summary is None:
            continue
        ws.append(w)
        mlp_auc.append(summary["mean"]["mlp"]["mean_test_roc_auc"])
        cnn_auc.append(summary["mean"]["cnn"]["mean_test_roc_auc"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ws, mlp_auc, "o-", linewidth=2, label="MLP", color="#4C72B0")
    ax.plot(ws, cnn_auc, "s--", linewidth=2, label="1D CNN", color="#C44E52")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xlabel("窗口长度 w")
    ax.set_ylabel("LOSO 平均 ROC-AUC")
    ax.set_title("MLP vs 1D CNN：长窗口下架构分化（k=1.0）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "mlp_vs_cnn_window.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_pnl_w96_vs_w192() -> None:
    paths = {
        "w=96 (24h)": METRICS_DIR / "backtest_realistic_w96_label_k12.json",
        "w=192 (48h)": METRICS_DIR / "backtest_realistic_w192_label_k12.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, path in paths.items():
        if not path.exists():
            continue
        pooled = json.loads(path.read_text())["pooled"]
        taus = [r["tau"] for r in pooled]
        ret = [r["total_return_net"] * 100 for r in pooled]
        dd = [abs(r["max_drawdown_net"]) * 100 for r in pooled]
        axes[0].plot(taus, ret, "o-", linewidth=2, label=label)
        axes[1].plot(taus, dd, "s--", linewidth=2, label=label)
    for ax in axes:
        ax.axvline(0.70, color="gray", linestyle=":", linewidth=0.9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("置信度阈值 τ")
    axes[0].set_ylabel("2 年总收益 (%)")
    axes[0].set_title("PnL：w=96 vs w=192")
    axes[1].set_xlabel("置信度阈值 τ")
    axes[1].set_ylabel("MaxDD (%)")
    axes[1].set_title("最大回撤对比")
    fig.tight_layout()
    out = PAPER_FIGURES / "pnl_w96_vs_w192.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_confidence_reliability() -> None:
    path = METRICS_DIR / "confidence_threshold_w192_label_k12.json"
    if not path.exists():
        return
    pooled = json.loads(path.read_text())["pooled"]
    taus = [r["tau"] for r in pooled]
    acc = [r["accuracy"] for r in pooled]
    cov = [r["coverage"] for r in pooled]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sc = ax.scatter(cov, acc, c=taus, cmap="viridis", s=80, edgecolors="black", linewidths=0.5)
    for t, c, a in zip(taus, cov, acc):
        if t in (0.5, 0.6, 0.7, 0.75, 0.85):
            ax.annotate(f"τ={t:.2f}", (c, a), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.plot(cov, acc, "--", color="gray", alpha=0.5, linewidth=1)
    ax.set_xlabel("Coverage（保留样本比例）")
    ax.set_ylabel("Accuracy")
    ax.set_title("置信度—准确率关系（k=1.2, w=192, pooled）")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046)
    cbar.set_label("τ")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGURES / "confidence_reliability.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_regime_analysis() -> None:
    src = METRICS_DIR / "regime_analysis_w192_label_k12.json"
    png_src = FIGURES_DIR / "regime_analysis_w192_label_k12.png"
    if png_src.exists():
        import shutil

        shutil.copy(png_src, PAPER_FIGURES / "regime_analysis_w192_label_k12.png")
        print(f"Copied regime figure -> {PAPER_FIGURES / 'regime_analysis_w192_label_k12.png'}")
    elif not src.exists():
        print("Skip regime figure; run scripts/regime_analysis.py first.")


def main() -> None:
    _ensure_dir()
    plot_data_exploration()
    plot_training_curve()
    plot_confusion_matrix()
    plot_label_k_sensitivity()
    plot_label_k_tradeoff()
    plot_window_sensitivity()
    plot_baseline_comparison()
    plot_mlp_vs_cnn_window()
    plot_ablation_summary()
    plot_ablation_delta()
    plot_pruned_kw_heatmap()
    plot_window_extended_scan()
    plot_heldout_auc_bars()
    plot_heldout_w96_vs_w192()
    plot_confidence_threshold()
    plot_confidence_reliability()
    plot_backtest_pnl()
    plot_pnl_w96_vs_w192()
    plot_milestone_evolution()
    plot_regime_analysis()
    import subprocess

    script = PROJECT_ROOT / "scripts" / "paper_mechanism_figures.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, check=False)
    print("All paper figures generated.")


if __name__ == "__main__":
    main()

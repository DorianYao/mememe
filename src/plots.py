from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from .config import ExperimentConfig


def plot_training_history(
    history: dict[str, list[float]],
    model_name: str,
    config: ExperimentConfig,
    run_tag: str | None = None,
) -> Path:
    config.ensure_dirs()
    tag = run_tag or config.run_tag(model_name)
    path = config.figures_dir / f"{tag}_training.png"
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"{model_name.upper()} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train Acc")
    axes[1].plot(epochs, history["val_accuracy"], label="Val Acc")
    axes[1].plot(epochs, history["train_macro_f1"], label="Train F1")
    axes[1].plot(epochs, history["val_macro_f1"], label="Val F1")
    axes[1].set_title(f"{model_name.upper()} Accuracy / Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    confusion: list[list[int]],
    model_name: str,
    config: ExperimentConfig,
    run_tag: str | None = None,
) -> Path:
    config.ensure_dirs()
    tag = run_tag or config.run_tag(model_name)
    path = config.figures_dir / f"{tag}_confusion.png"
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(confusion),
        display_labels=["Down (0)", "Up (1)"],
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name.upper()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_roc_curves(
    metrics_per_model: dict[str, dict[str, object]],
    config: ExperimentConfig,
    tag: str,
) -> Path:
    config.ensure_dirs()
    path = config.figures_dir / f"{tag}_roc.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    for name, metrics in metrics_per_model.items():
        roc = metrics["roc_curve"]
        auc_value = metrics["test_roc_auc"]
        ax.plot(roc["fpr"], roc["tpr"], label=f"{name.upper()} (AUC={auc_value:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {tag}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_model_comparison(
    metrics_per_model: dict[str, dict[str, object]],
    baseline: dict[str, float],
    config: ExperimentConfig,
    tag: str,
) -> Path:
    config.ensure_dirs()
    path = config.figures_dir / f"{tag}_comparison.png"
    model_names = [baseline["model"], *metrics_per_model.keys()]
    accuracy = [baseline["accuracy"], *[m["test_accuracy"] for m in metrics_per_model.values()]]
    macro_f1 = [baseline["macro_f1"], *[m["test_macro_f1"] for m in metrics_per_model.values()]]
    mcc = [baseline.get("mcc", 0.0), *[m["test_mcc"] for m in metrics_per_model.values()]]

    x = np.arange(len(model_names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, accuracy, width, label="Accuracy")
    ax.bar(x, macro_f1, width, label="Macro-F1")
    ax.bar(x + width, mcc, width, label="MCC")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-0.2, 1.0)
    ax.set_title(f"Model Comparison - {tag}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(accuracy):
        ax.text(x[i] - width, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
    for i, value in enumerate(macro_f1):
        ax.text(x[i], value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
    for i, value in enumerate(mcc):
        ax.text(x[i] + width, value + 0.01, f"{value:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_loso_summary(
    rounds: dict[str, dict[str, dict[str, object]]],
    baseline_per_round: dict[str, dict[str, float]],
    config: ExperimentConfig,
    metric: str = "test_roc_auc",
) -> Path:
    """画 LOSO 多轮的柱状图：每个 held-out 币种一组，每组 baseline + 各模型。"""
    config.ensure_dirs()
    held_out_symbols = sorted(rounds.keys())
    if not held_out_symbols:
        raise ValueError("No LOSO rounds to plot.")
    model_names = sorted({m for r in rounds.values() for m in r.keys()})

    n_groups = len(held_out_symbols)
    n_bars = len(model_names) + 1
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.2), 5))

    baseline_values = [
        baseline_per_round.get(sym, {}).get(
            "roc_auc" if metric == "test_roc_auc" else metric.replace("test_", ""),
            0.5,
        )
        for sym in held_out_symbols
    ]
    ax.bar(x - 0.4 + width / 2, baseline_values, width, label="majority_class")

    for i, name in enumerate(model_names):
        values = [rounds[sym][name][metric] for sym in held_out_symbols]
        ax.bar(x - 0.4 + width * (i + 1.5), values, width, label=name.upper())

    means = {
        name: float(np.nanmean([rounds[sym][name][metric] for sym in held_out_symbols]))
        for name in model_names
    }
    legend_extras = "  ".join(f"{n.upper()} mean={v:.3f}" for n, v in means.items())

    if metric == "test_roc_auc":
        ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", label="AUC=0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(held_out_symbols, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"LOSO {metric} per held-out symbol  ({legend_extras})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if metric in {"test_roc_auc", "test_accuracy", "test_macro_f1"}:
        ax.set_ylim(0, 1)

    fig.tight_layout()
    path = config.figures_dir / f"{config.universe_tag('loso')}_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .features import FeatureDataset, SplitDataset


def _resolve_device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def predict_probs(
    model: nn.Module,
    dataset: FeatureDataset,
    batch_size: int,
    device: str | None = None,
) -> np.ndarray:
    resolved_device = _resolve_device(device)
    x = torch.from_numpy(dataset.x)
    y = torch.from_numpy(dataset.y)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    model.to(resolved_device)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            logits = model(batch_x.to(resolved_device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": mcc,
        "roc_auc": auc,
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "samples": int(len(y_true)),
    }


def evaluate_model(
    model: nn.Module,
    splits: SplitDataset,
    model_name: str,
    config: ExperimentConfig,
    device: str | None = None,
    run_tag: str | None = None,
    metric_for_threshold: str = "macro_f1",
    prob_fn=None,
    skip_artifacts: bool = False,
) -> dict[str, object]:
    if prob_fn is not None:
        val_probs = prob_fn(model, splits.val)
        test_probs = prob_fn(model, splits.test)
    else:
        val_probs = predict_probs(model, splits.val, config.batch_size, device)
        test_probs = predict_probs(model, splits.test, config.batch_size, device)
    val_targets = splits.val.y.astype(int)
    test_targets = splits.test.y.astype(int)

    best_threshold = search_threshold(val_probs, val_targets, metric=metric_for_threshold)
    test_preds = (test_probs >= best_threshold).astype(int)

    try:
        fpr, tpr, _ = roc_curve(test_targets, test_probs)
    except ValueError:
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])

    base = _safe_metrics(test_targets, test_preds, test_probs)
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1]).tolist()
    report = classification_report(
        test_targets, test_preds, output_dict=True, zero_division=0
    )

    per_symbol = (
        per_symbol_breakdown(test_targets, test_preds, test_probs, splits.test.symbols)
        if splits.test.symbols is not None and len(splits.test.symbols) > 0
        else {}
    )

    metrics: dict[str, object] = {
        "model": model_name,
        "symbol": config.symbol.upper(),
        "interval": config.interval,
        "window_size": config.window_size,
        "split_mode": config.split_mode,
        "split_tag": config.split_tag,
        "universe": splits.universe,
        "held_out": splits.held_out,
        "best_threshold": float(best_threshold),
        **{f"test_{k}": v for k, v in base.items()},
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "per_symbol": per_symbol,
    }

    config.ensure_dirs()
    tag = run_tag or config.run_tag(model_name)
    metrics_path = config.metrics_dir / f"{tag}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if skip_artifacts:
        return metrics

    predictions_path = config.metrics_dir / f"{tag}_predictions.csv"
    have_syms = splits.test.symbols is not None and len(splits.test.symbols) > 0
    header = "timestamp,symbol,y_true,y_pred,prob_up" if have_syms else "timestamp,y_true,y_pred,prob_up"
    lines = [header]
    for i, (ts, y_true, y_pred, prob) in enumerate(
        zip(splits.test.timestamps, test_targets, test_preds, test_probs, strict=True)
    ):
        if have_syms:
            sym = splits.test.symbols[i]
            lines.append(f"{ts},{sym},{int(y_true)},{int(y_pred)},{prob:.8f}")
        else:
            lines.append(f"{ts},{int(y_true)},{int(y_pred)},{prob:.8f}")
    predictions_path.write_text("\n".join(lines), encoding="utf-8")

    val_preds = (val_probs >= best_threshold).astype(int)
    val_path = config.metrics_dir / f"{tag}_val_predictions.csv"
    val_header = (
        "timestamp,symbol,y_true,y_pred,prob_up"
        if splits.val.symbols is not None and len(splits.val.symbols) > 0
        else "timestamp,y_true,y_pred,prob_up"
    )
    val_lines = [val_header]
    for i, (ts, y_true, y_pred, prob) in enumerate(
        zip(splits.val.timestamps, val_targets, val_preds, val_probs, strict=True)
    ):
        if splits.val.symbols is not None and len(splits.val.symbols) > 0:
            sym = splits.val.symbols[i]
            val_lines.append(f"{ts},{sym},{int(y_true)},{int(y_pred)},{prob:.8f}")
        else:
            val_lines.append(f"{ts},{int(y_true)},{int(y_pred)},{prob:.8f}")
    val_path.write_text("\n".join(val_lines), encoding="utf-8")
    metrics["val_predictions_path"] = str(val_path)

    return metrics


def per_symbol_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    symbols: np.ndarray,
) -> dict[str, dict[str, float]]:
    """对每个币种单独算指标。"""
    out: dict[str, dict[str, float]] = {}
    if symbols is None or len(symbols) == 0:
        return out
    arr = np.asarray(symbols)
    for sym in np.unique(arr):
        mask = arr == sym
        if mask.sum() < 5:
            continue
        out[str(sym)] = _safe_metrics(y_true[mask], y_pred[mask], probs[mask])
    return out


def search_threshold(
    probs: np.ndarray, targets: np.ndarray, metric: str = "macro_f1"
) -> float:
    """在 [0.3, 0.7] 网格里搜最优阈值（按指定 metric）。"""
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.3, 0.7, 41):
        preds = (probs >= threshold).astype(int)
        if metric == "macro_f1":
            score = f1_score(targets, preds, average="macro", zero_division=0)
        elif metric == "mcc":
            try:
                score = matthews_corrcoef(targets, preds)
            except ValueError:
                score = 0.0
        elif metric == "precision":
            score = precision_score(targets, preds, zero_division=0)
        else:
            score = f1_score(targets, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def save_comparison(
    metrics_per_model: dict[str, dict[str, object]],
    baseline: dict[str, float],
    config: ExperimentConfig,
    tag: str,
) -> Path:
    config.ensure_dirs()
    header = "model,accuracy,macro_f1,mcc,roc_auc,precision,recall"
    lines = [header]
    lines.append(
        f"{baseline['model']},{baseline['accuracy']:.6f},"
        f"{baseline['macro_f1']:.6f},{baseline.get('mcc', 0.0):.6f},nan,nan,nan"
    )
    for name, m in metrics_per_model.items():
        lines.append(
            f"{name},"
            f"{m['test_accuracy']:.6f},"
            f"{m['test_macro_f1']:.6f},"
            f"{m['test_mcc']:.6f},"
            f"{m['test_roc_auc']:.6f},"
            f"{m['test_precision']:.6f},"
            f"{m['test_recall']:.6f}"
        )
    out_path = config.metrics_dir / f"{tag}_comparison.csv"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def aggregate_loso(
    rounds: dict[str, dict[str, dict[str, object]]],
    config: ExperimentConfig,
    universe: list[str],
) -> dict[str, object]:
    """聚合 LOSO 多轮结果。

    rounds: {held_out_symbol: {model_name: metrics}}
    返回每个模型在所有 held-out 下的 mean / std 指标。
    """
    model_names: set[str] = set()
    for r in rounds.values():
        model_names.update(r.keys())

    summary: dict[str, object] = {
        "universe": universe,
        "category_id": config.category_id,
        "n_rounds": len(rounds),
        "per_round": {
            symbol: {
                name: {
                    k: r[name][k]
                    for k in [
                        "test_accuracy",
                        "test_macro_f1",
                        "test_mcc",
                        "test_roc_auc",
                        "test_precision",
                        "test_recall",
                        "test_positive_rate",
                        "test_samples",
                    ]
                    if k in r[name]
                }
                for name in r
            }
            for symbol, r in rounds.items()
        },
    }

    aggregated: dict[str, dict[str, float]] = {}
    for name in sorted(model_names):
        rows = [r[name] for r in rounds.values() if name in r]
        if not rows:
            continue
        aggregated[name] = {}
        for metric_key in [
            "test_accuracy",
            "test_macro_f1",
            "test_mcc",
            "test_roc_auc",
            "test_precision",
            "test_recall",
        ]:
            values = [r[metric_key] for r in rows if isinstance(r.get(metric_key), (int, float))]
            if values:
                aggregated[name][f"mean_{metric_key}"] = float(np.mean(values))
                aggregated[name][f"std_{metric_key}"] = float(np.std(values))
    summary["mean"] = aggregated

    out_path = config.metrics_dir / f"{config.universe_tag('loso')}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(out_path)
    return summary

"""Tabular baselines (XGBoost) on flattened window features."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import ExperimentConfig
from .features import FeatureDataset, SplitDataset


def flatten_x(x: np.ndarray) -> np.ndarray:
    """(samples, window, features) -> (samples, window * features)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D input, got shape {x.shape}")
    return x.reshape(x.shape[0], -1).astype(np.float32)


def train_xgboost(
    splits: SplitDataset,
    config: ExperimentConfig,
    verbose: bool = True,
) -> tuple[object, dict[str, list[float]], dict[str, float]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required: pip install xgboost") from exc

    x_train = flatten_x(splits.train.x)
    x_val = flatten_x(splits.val.x)
    y_train = splits.train.y.astype(int)
    y_val = splits.val.y.astype(int)

    pos_rate = float(y_train.mean()) if len(y_train) else 0.5
    scale_pos_weight = float((1 - pos_rate) / max(pos_rate, 1e-4))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(max(scale_pos_weight, 0.5), 4.0),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=config.random_state,
        n_jobs=-1,
        early_stopping_rounds=config.early_stopping_patience,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=int(verbose),
    )

    train_probs = model.predict_proba(x_train)[:, 1]
    val_probs = model.predict_proba(x_val)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    val_preds = (val_probs >= 0.5).astype(int)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [float(accuracy_score(y_train, train_preds))],
        "train_macro_f1": [
            float(f1_score(y_train, train_preds, average="macro", zero_division=0))
        ],
        "val_loss": [],
        "val_accuracy": [float(accuracy_score(y_val, val_preds))],
        "val_macro_f1": [
            float(f1_score(y_val, val_preds, average="macro", zero_division=0))
        ],
    }

    best_iter = getattr(model, "best_iteration", model.n_estimators - 1)
    summary = {
        "best_epoch": int(best_iter),
        "best_val_macro_f1": float(history["val_macro_f1"][-1]),
        "parameter_count": int(model.get_booster().num_features()),
        "epochs_run": int(model.n_estimators),
        "model_type": "xgboost",
    }
    return model, history, summary


def predict_xgboost_probs(model, dataset: FeatureDataset) -> np.ndarray:
    x = flatten_x(dataset.x)
    return model.predict_proba(x)[:, 1].astype(np.float64)


def save_xgboost_model(
    model,
    history: dict[str, list[float]],
    summary: dict[str, float],
    config: ExperimentConfig,
    run_tag: str,
) -> Path:
    import joblib

    config.ensure_dirs()
    model_path = config.model_dir / f"{run_tag}.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": "xgboost",
            "window_size": config.window_size,
            "num_features": config.num_features,
            "feature_columns": list(config.feature_columns),
            "summary": summary,
        },
        model_path,
    )
    history_path = config.metrics_dir / f"{run_tag}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model_path


def load_xgboost_model(path: Path):
    import joblib

    payload = joblib.load(path)
    return payload["model"], payload

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .config import ExperimentConfig
from .features import FeatureDataset, SplitDataset
from .models import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_loader(dataset: FeatureDataset, batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.from_numpy(dataset.x)
    y = torch.from_numpy(dataset.y)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _resolve_device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    positive_rate = y.mean().clamp(min=1e-4, max=0.9999)
    return ((1 - positive_rate) / positive_rate).clamp(min=0.5, max=4.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=_pos_weight(y)
        )
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    preds = (probs >= 0.5).astype(int)
    n_samples = len(loader.dataset)  # type: ignore[arg-type]
    return {
        "loss": total_loss / max(n_samples, 1),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train_one_model(
    model_name: str,
    splits: SplitDataset,
    config: ExperimentConfig,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, float]]:
    set_seed(config.random_state)
    resolved_device = _resolve_device(device)

    train_loader = _to_loader(splits.train, config.batch_size, shuffle=True)
    val_loader = _to_loader(splits.val, config.batch_size, shuffle=False)

    window_size = splits.train.x.shape[1]
    num_features = splits.train.x.shape[2]

    model = build_model(model_name, window_size, num_features, config).to(resolved_device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    best_state = deepcopy(model.state_dict())
    best_val_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0

    iterator = range(config.epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {model_name.upper()}", leave=False)

    for epoch in iterator:
        train_metrics = run_epoch(model, train_loader, resolved_device, optimizer)
        val_metrics = run_epoch(model, val_loader, resolved_device)
        scheduler.step()

        for key, value in train_metrics.items():
            history[f"train_{key}"].append(float(value))
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(float(value))

        score = val_metrics["macro_f1"]
        if score > best_val_metric:
            best_val_metric = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(
                        f"[{model_name}] Early stopping at epoch {epoch + 1} "
                        f"(best epoch {best_epoch + 1}, val macro-F1 {best_val_metric:.4f})"
                    )
                break

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_metric),
        "parameter_count": int(count_parameters(model)),
        "epochs_run": len(history["train_loss"]),
    }
    return model, history, summary


def save_model(
    model: nn.Module,
    model_name: str,
    history: dict[str, list[float]],
    summary: dict[str, float],
    config: ExperimentConfig,
    run_tag: str | None = None,
) -> Path:
    config.ensure_dirs()
    tag = run_tag or config.run_tag(model_name)
    model_path = config.model_dir / f"{tag}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "window_size": config.window_size,
            "num_features": config.num_features,
            "feature_columns": list(config.feature_columns),
            "summary": summary,
        },
        model_path,
    )
    history_path = config.metrics_dir / f"{tag}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model_path


def majority_baseline(splits: SplitDataset) -> dict[str, float]:
    from sklearn.metrics import matthews_corrcoef

    majority = int(np.mean(splits.train.y) >= 0.5)
    preds = np.full(len(splits.test.y), majority, dtype=int)
    targets = splits.test.y.astype(int)
    try:
        mcc = float(matthews_corrcoef(targets, preds))
    except ValueError:
        mcc = 0.0
    return {
        "model": "majority_class",
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "mcc": mcc,
    }

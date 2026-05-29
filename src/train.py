from __future__ import annotations

import json
import os
import random
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


def _to_loader(
    dataset: FeatureDataset,
    batch_size: int,
    shuffle: bool,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    x = torch.from_numpy(dataset.x)
    y = torch.from_numpy(dataset.y)
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(TensorDataset(x, y), **kwargs)


def _resolve_device(device: str | None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    positive_rate = y.mean().clamp(min=1e-4, max=0.9999)
    return ((1 - positive_rate) / positive_rate).clamp(min=0.5, max=4.0)


def _estimate_split_bytes(splits: SplitDataset) -> int:
    total = 0
    for ds in (splits.train, splits.val):
        total += int(ds.x.nbytes + ds.y.nbytes)
    return total


def _should_preload_to_gpu(splits: SplitDataset, device: torch.device) -> bool:
    """Keep train+val on GPU when they fit — avoids per-batch H2D and DataLoader overhead."""
    if device.type != "cuda":
        return False
    need = _estimate_split_bytes(splits)
    parallel = max(1, int(os.environ.get("PARALLEL_GPU_JOBS", "1")))
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return False
    return need < int(free * 0.85 / parallel)


def _run_epoch_preloaded(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    pos_weight: torch.Tensor | None = None,
    shuffle: bool = True,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    n = y.size(0)
    order = torch.randperm(n, device=device) if shuffle and training else torch.arange(n, device=device)

    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        pw = pos_weight if training and pos_weight is not None else _pos_weight(yb)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * yb.size(0)
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        pw = pos_weight if training and pos_weight is not None else _pos_weight(y)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw
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
    use_cuda = resolved_device.type == "cuda"
    workers = config.dataloader_workers if config.dataloader_workers > 0 else 0
    if workers > 0:
        workers = min(workers, max(1, (os.cpu_count() or 2) - 1))

    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": use_cuda,
    }
    train_pos_weight = _pos_weight(torch.from_numpy(splits.train.y)).to(resolved_device)

    window_size = splits.train.x.shape[1]
    num_features = splits.train.x.shape[2]

    model = build_model(model_name, window_size, num_features, config).to(resolved_device)

    preload = _should_preload_to_gpu(splits, resolved_device)
    if preload:
        train_x = torch.from_numpy(splits.train.x).to(resolved_device)
        train_y = torch.from_numpy(splits.train.y).to(resolved_device)
        val_x = torch.from_numpy(splits.val.x).to(resolved_device)
        val_y = torch.from_numpy(splits.val.y).to(resolved_device)
        train_loader = val_loader = None
    else:
        train_x = train_y = val_x = val_y = None
        train_loader = _to_loader(
            splits.train, config.batch_size, shuffle=True, **loader_kwargs
        )
        val_loader = _to_loader(
            splits.val, config.batch_size, shuffle=False, **loader_kwargs
        )

    if verbose:
        mode = "gpu-preload" if preload else f"loader(workers={workers})"
        print(
            f"[{model_name}] device={resolved_device} train_n={len(splits.train.y)} "
            f"batch={config.batch_size} mode={mode}",
            flush=True,
        )

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

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0

    iterator = range(config.epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {model_name.upper()}", leave=False)

    for epoch in iterator:
        if preload:
            train_metrics = _run_epoch_preloaded(
                model,
                train_x,
                train_y,
                config.batch_size,
                resolved_device,
                optimizer,
                pos_weight=train_pos_weight,
                shuffle=True,
            )
            with torch.inference_mode():
                val_metrics = _run_epoch_preloaded(
                    model,
                    val_x,
                    val_y,
                    config.batch_size,
                    resolved_device,
                    shuffle=False,
                )
        else:
            train_metrics = run_epoch(
                model,
                train_loader,
                resolved_device,
                optimizer,
                pos_weight=train_pos_weight,
            )
            with torch.inference_mode():
                val_metrics = run_epoch(model, val_loader, resolved_device)
        scheduler.step()

        for key, value in train_metrics.items():
            history[f"train_{key}"].append(float(value))
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(float(value))

        score = val_metrics["macro_f1"]
        if score > best_val_metric:
            best_val_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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

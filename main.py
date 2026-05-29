from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.categories import CATEGORY_IDS, apply_category, get_legacy_meme8_symbols, list_categories
from src.config import ExperimentConfig, config_from_yaml
from src.data import download_multi, download_ohlcv, load_ohlcv
from src.evaluate import (
    aggregate_loso,
    evaluate_model,
    save_comparison,
)
from src.features import build_and_save, load_processed
from src.multi import (
    build_and_save_loso,
    build_and_save_pooled,
    build_per_symbol,
    loso_processed_path,
    pooled_processed_path,
)
from src.plots import (
    plot_confusion_matrix,
    plot_loso_summary,
    plot_model_comparison,
    plot_roc_curves,
    plot_training_history,
)
from src.train import majority_baseline, save_model, train_one_model


MODEL_CHOICES = ["mlp", "cnn", "xgboost", "both"]
STAGE_CHOICES = ["download", "features", "train", "evaluate", "all"]
MODE_CHOICES = ["single", "pooled", "loso"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="15m K-line next-bar up/down classifier "
        "(MLP + 1D CNN, single / pooled / LOSO)."
    )
    parser.add_argument("--stage", default="all", choices=STAGE_CHOICES)
    parser.add_argument("--model", default="both", choices=MODEL_CHOICES)
    parser.add_argument(
        "--mode",
        default="loso",
        choices=MODE_CHOICES,
        help=(
            "single: legacy single-symbol training; "
            "pooled: multi-symbol pooled with time-aligned split; "
            "loso: leave-one-symbol-out (default, the headline experiment)."
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--symbol", default=None, help="Override single-mode symbol.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated override of the meme universe (pooled / loso).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Research category id (bluechip, midcap, solana_fast, base_eco, micro_cap, "
            "or meme8 for legacy 8-coin universe)."
        ),
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Run loso mode sequentially for all five research categories.",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of categories (with --all-categories or loso).",
    )
    parser.add_argument(
        "--held-out",
        default=None,
        help="If set in loso mode, only run that one held-out symbol round.",
    )
    parser.add_argument("--interval", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-mode", default=None, choices=["fixed", "volatility"])
    parser.add_argument("--label-k", type=float, default=None)
    parser.add_argument("--label-epsilon", type=float, default=None)
    parser.add_argument(
        "--exclude-features",
        default=None,
        help="Comma-separated feature column names to drop from feature_columns "
        "(ablation studies).",
    )
    parser.add_argument(
        "--include-features",
        default=None,
        help="Comma-separated feature column names to use exclusively (whitelist; "
        "overrides default feature_columns).",
    )
    parser.add_argument(
        "--ablation-tag",
        default=None,
        help="Suffix appended to all output filenames; use to keep ablation runs "
        "separate from baseline.",
    )
    parser.add_argument(
        "--split-mode",
        default=None,
        choices=["ratio", "calendar"],
        help="LOSO split: ratio (legacy 85/15 pool) or calendar walk-forward.",
    )
    parser.add_argument(
        "--time-train-fraction",
        type=float,
        default=None,
        help="Calendar mode: global train end quantile (default 0.70).",
    )
    parser.add_argument(
        "--time-val-fraction",
        type=float,
        default=None,
        help="Calendar mode: global val end quantile (default 0.85).",
    )
    parser.add_argument(
        "--label-volatility-window",
        type=int,
        default=None,
        help="Rolling window for label sigma (decouple from feature volatility_50).",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=None,
        help="Rolling window for feature volatility_50 column.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib figures (faster for hyperparameter scans).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip LOSO held-out rounds whose metrics JSON already exists.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Scan mode: --skip-plots, skip prediction CSVs, batch_size=1024 if unset.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default from config, usually 4).",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=None,
        help="Parallel workers for per-symbol window building (default 1).",
    )
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides: dict[str, object] = {}
    direct_keys = [
        "symbol",
        "interval",
        "window_size",
        "lookback_days",
        "epochs",
        "batch_size",
        "learning_rate",
        "dropout",
        "label_mode",
        "label_k",
        "label_epsilon",
        "split_mode",
        "time_train_fraction",
        "time_val_fraction",
        "label_volatility_window",
        "volatility_window",
        "dataloader_workers",
        "feature_workers",
    ]
    for key in direct_keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "fast", False):
        if getattr(args, "batch_size", None) is None:
            overrides["batch_size"] = 1024
        if getattr(args, "dataloader_workers", None) is None:
            overrides["dataloader_workers"] = 4
    if args.symbols:
        overrides["symbols"] = tuple(
            s.strip().upper() for s in args.symbols.split(",") if s.strip()
        )
    if args.category:
        cat = args.category.strip().lower()
        if cat == "meme8":
            overrides["symbols"] = get_legacy_meme8_symbols()
            overrides["category_id"] = None
        else:
            tmp = apply_category(config, cat)
            overrides["symbols"] = tmp.symbols
            overrides["category_id"] = tmp.category_id
    if args.ablation_tag is not None:
        overrides["ablation_tag"] = args.ablation_tag

    # 消融实验：feature 列表覆盖
    base_features = list(config.feature_columns)
    if args.include_features:
        whitelist = [f.strip() for f in args.include_features.split(",") if f.strip()]
        unknown = [f for f in whitelist if f not in base_features and f not in {
            "volume_log", "volume_change"
        }]
        if unknown:
            raise ValueError(
                f"--include-features contains unknown columns: {unknown}. "
                "Valid columns must exist in features.add_indicators output."
            )
        overrides["feature_columns"] = tuple(whitelist)
    elif args.exclude_features:
        drop = {f.strip() for f in args.exclude_features.split(",") if f.strip()}
        remaining = [f for f in base_features if f not in drop]
        if not remaining:
            raise ValueError("Cannot exclude all features.")
        overrides["feature_columns"] = tuple(remaining)

    if not overrides:
        return config
    return ExperimentConfig(**{**config.__dict__, **overrides})


def resolve_models(model_arg: str) -> list[str]:
    if model_arg == "both":
        return ["mlp", "cnn"]
    return [model_arg]


def run_single(args: argparse.Namespace, config: ExperimentConfig) -> dict[str, object]:
    summary: dict[str, object] = {"mode": "single", "config": _config_summary(config)}

    if args.stage in {"download", "all"}:
        path = download_ohlcv(config, refresh=args.refresh)
        summary["download"] = {"raw_csv": str(path)}

    if args.stage in {"features", "all"}:
        ohlcv = load_ohlcv(config)
        splits, npz_path = build_and_save(ohlcv, config)
        summary["features"] = _split_summary(splits, str(npz_path))

    if args.stage in {"train", "evaluate", "all"}:
        splits = load_processed(config.processed_npz_path)
        run_tag = config.run_tag  # function reference for callers below
        baseline = majority_baseline(splits)
        summary["baseline"] = baseline
        metrics_per_model = _train_eval_models(args, splits, config, run_tag)
        comparison_tag = f"{config.symbol.upper()}_{config.interval}_w{config.window_size}"
        _emit_comparison(metrics_per_model, baseline, config, comparison_tag, summary)

    return summary


def run_pooled(args: argparse.Namespace, config: ExperimentConfig) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": "pooled",
        "config": _config_summary(config),
        "universe": list(config.symbols),
    }

    if args.stage in {"download", "all"}:
        saved = download_multi(config, refresh=args.refresh)
        summary["download"] = {sym: str(path) for sym, path in saved.items()}

    if args.stage in {"features", "all"}:
        splits, npz_path = build_and_save_pooled(config)
        summary["features"] = _split_summary(splits, str(npz_path))

    if args.stage in {"train", "evaluate", "all"}:
        splits = load_processed(pooled_processed_path(config))
        baseline = majority_baseline(splits)
        summary["baseline"] = baseline
        tag_base = config.universe_tag("pooled")

        def run_tag(model_name: str) -> str:
            return f"{tag_base}_{model_name}"

        metrics_per_model = _train_eval_models(args, splits, config, run_tag)
        _emit_comparison(metrics_per_model, baseline, config, tag_base, summary)

    return summary


def run_loso(args: argparse.Namespace, config: ExperimentConfig) -> dict[str, object]:
    summary: dict[str, object] = {
        "mode": "loso",
        "config": _config_summary(config),
        "universe": list(config.symbols),
    }

    if args.stage in {"download", "all"}:
        saved = download_multi(config, refresh=args.refresh)
        summary["download"] = {sym: str(path) for sym, path in saved.items()}

    if args.held_out:
        held_outs = [args.held_out.upper()]
    else:
        held_outs = list(config.symbols)

    rounds: dict[str, dict[str, dict[str, object]]] = {}
    baseline_per_round: dict[str, dict[str, float]] = {}
    features_summary: dict[str, dict[str, object]] = {}

    # 关键优化：一次性把所有币的窗口建好，多轮 LOSO 共用。
    per_symbol_cache: dict | None = None
    need_feature_build = _loso_need_feature_build(args, config, held_outs)
    if need_feature_build:
        print("[loso] building per-symbol window cache (one-time)...")
        per_symbol_cache = build_per_symbol(list(config.symbols), config)

    skip_artifacts = getattr(args, "fast", False)
    skip_plots = getattr(args, "skip_plots", False) or getattr(args, "fast", False)

    for held in held_outs:
        print(f"\n========== LOSO  held-out: {held} ==========")
        path = loso_processed_path(config, held)
        tag_base = config.universe_tag("loso", held)

        if getattr(args, "skip_existing", False) and args.stage == "features":
            if path.exists() and not args.refresh:
                print(f"[skip-existing] {held} npz already at {path.name}")
                continue

        if getattr(args, "skip_existing", False) and args.stage in {"train", "evaluate", "all"}:
            metrics_path = config.metrics_dir / f"{tag_base}_mlp_metrics.json"
            if metrics_path.exists() and path.exists():
                print(f"[skip-existing] {held} metrics already at {metrics_path.name}")
                with metrics_path.open("r", encoding="utf-8") as fh:
                    rounds[held] = {"mlp": json.load(fh)}
                continue

        if path.exists() and not args.refresh:
            splits = load_processed(path)
        elif args.stage in {"features", "all", "train", "evaluate"} or not path.exists():
            if per_symbol_cache is None:
                print("[loso] building per-symbol window cache (one-time)...")
                per_symbol_cache = build_per_symbol(list(config.symbols), config)
            splits, _ = build_and_save_loso(
                config, held, per_symbol_cache=per_symbol_cache
            )
        else:
            splits = load_processed(path)
        features_summary[held] = _split_summary(splits, str(path))

        if args.stage in {"train", "evaluate", "all"}:
            baseline = majority_baseline(splits)
            baseline_per_round[held] = baseline

            def run_tag(model_name: str, _tag_base=tag_base) -> str:
                return f"{_tag_base}_{model_name}"

            metrics_per_model = _train_eval_models(
                args,
                splits,
                config,
                run_tag,
                skip_plots=skip_plots,
                skip_artifacts=skip_artifacts,
            )
            rounds[held] = metrics_per_model
            _emit_comparison(
                metrics_per_model,
                baseline,
                config,
                tag_base,
                summary,
                key=held,
                skip_plots=skip_plots,
            )

    summary["features"] = features_summary

    if rounds:
        aggregation = aggregate_loso(rounds, config, list(config.symbols))
        summary["loso_aggregate"] = aggregation["mean"]
        summary["loso_summary_path"] = aggregation["summary_path"]

        if not skip_plots:
            for metric in ["test_roc_auc", "test_macro_f1", "test_mcc"]:
                plot_path = plot_loso_summary(
                    rounds, baseline_per_round, config, metric=metric
                )
                summary.setdefault("loso_figures", {})[metric] = str(plot_path)

    return summary


def _loso_need_feature_build(
    args: argparse.Namespace,
    config: ExperimentConfig,
    held_outs: list[str],
) -> bool:
    """True when any held-out round still needs fresh feature NPZ files."""
    if args.stage == "features":
        return True
    if args.stage not in {"all", "train"}:
        return False
    if args.refresh:
        return True
    return any(not loso_processed_path(config, held).exists() for held in held_outs)


def _train_eval_models(
    args: argparse.Namespace,
    splits,
    config: ExperimentConfig,
    run_tag_fn,
    *,
    skip_plots: bool = False,
    skip_artifacts: bool = False,
) -> dict[str, dict[str, object]]:
    from src.tabular import (
        load_xgboost_model,
        predict_xgboost_probs,
        save_xgboost_model,
        train_xgboost,
    )

    metrics_per_model: dict[str, dict[str, object]] = {}
    for model_name in resolve_models(args.model):
        tag = run_tag_fn(model_name)
        if model_name == "xgboost":
            joblib_path = config.model_dir / f"{tag}.joblib"
            if args.stage in {"train", "all"}:
                model, history, train_summary = train_xgboost(
                    splits, config, verbose=not args.quiet
                )
                save_xgboost_model(model, history, train_summary, config, tag)
            else:
                if not joblib_path.exists():
                    raise FileNotFoundError(f"XGBoost model not found: {joblib_path}")
                model, _ = load_xgboost_model(joblib_path)

            prob_fn = lambda m, ds: predict_xgboost_probs(m, ds)
            metrics = evaluate_model(
                model,
                splits,
                model_name,
                config,
                run_tag=tag,
                prob_fn=prob_fn,
            )
        else:
            if args.stage in {"train", "all"}:
                model, history, train_summary = train_one_model(
                    model_name,
                    splits,
                    config,
                    device=args.device,
                    verbose=not args.quiet,
                )
                save_model(model, model_name, history, train_summary, config, run_tag=tag)
                if not skip_plots:
                    plot_training_history(history, model_name, config, run_tag=tag)
            else:
                model = _load_model(model_name, config, tag, device=args.device)

            metrics = evaluate_model(
                model,
                splits,
                model_name,
                config,
                device=args.device,
                run_tag=tag,
                skip_artifacts=skip_artifacts,
            )
        if not skip_plots:
            plot_confusion_matrix(metrics["confusion_matrix"], model_name, config, run_tag=tag)
        metrics_per_model[model_name] = metrics
    return metrics_per_model


def _emit_comparison(
    metrics_per_model: dict[str, dict[str, object]],
    baseline: dict[str, float],
    config: ExperimentConfig,
    tag: str,
    summary: dict[str, object],
    key: str | None = None,
    *,
    skip_plots: bool = False,
) -> None:
    if not metrics_per_model:
        return
    comparison_csv = save_comparison(metrics_per_model, baseline, config, tag)
    artifacts = {"comparison_csv": str(comparison_csv)}
    if not skip_plots:
        comparison_plot = plot_model_comparison(metrics_per_model, baseline, config, tag)
        roc_plot = plot_roc_curves(metrics_per_model, config, tag)
        artifacts["comparison_plot"] = str(comparison_plot)
        artifacts["roc_plot"] = str(roc_plot)
    if key:
        summary.setdefault("artifacts", {})[key] = artifacts
        summary.setdefault("evaluate", {})[key] = {
            name: _strip_metrics(m) for name, m in metrics_per_model.items()
        }
    else:
        summary["artifacts"] = artifacts
        summary["evaluate"] = {
            name: _strip_metrics(m) for name, m in metrics_per_model.items()
        }


def _strip_metrics(metrics: dict[str, object]) -> dict[str, object]:
    drop = {"classification_report", "roc_curve"}
    out = {k: v for k, v in metrics.items() if k not in drop}
    return out


def _split_summary(splits, npz_path: str) -> dict[str, object]:
    return {
        "processed_npz": npz_path,
        "train_samples": int(len(splits.train.y)),
        "val_samples": int(len(splits.val.y)),
        "test_samples": int(len(splits.test.y)),
        "positive_rate_train": float(splits.train.y.mean()) if len(splits.train.y) else 0.0,
        "positive_rate_val": float(splits.val.y.mean()) if len(splits.val.y) else 0.0,
        "positive_rate_test": float(splits.test.y.mean()) if len(splits.test.y) else 0.0,
        "universe": splits.universe,
        "held_out": splits.held_out,
    }


def _config_summary(config: ExperimentConfig) -> dict[str, object]:
    return {
        "symbol": config.symbol,
        "interval": config.interval,
        "window_size": config.window_size,
        "lookback_days": config.lookback_days,
        "label_mode": config.label_mode,
        "label_k": config.label_k,
        "label_epsilon": config.label_epsilon,
        "feature_columns": list(config.feature_columns),
        "universe": list(config.symbols),
        "category_id": config.category_id,
        "ablation_tag": config.ablation_tag,
        "num_features": config.num_features,
        "split_mode": config.split_mode,
        "split_tag": config.split_tag,
        "time_train_fraction": config.time_train_fraction,
        "time_val_fraction": config.time_val_fraction,
        "label_volatility_window": config.label_volatility_window,
    }


def _load_model(
    model_name: str,
    config: ExperimentConfig,
    tag: str,
    device: str | None,
):
    import torch

    from src.models import build_model

    path = config.model_dir / f"{tag}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. Run --stage train first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    window_size = checkpoint.get("window_size", config.window_size)
    num_features = checkpoint.get("num_features", config.num_features)
    model = build_model(model_name, window_size, num_features, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    return model


def _resolve_category_run_list(args: argparse.Namespace) -> list[str]:
    if args.categories:
        return [c.strip().lower() for c in args.categories.split(",") if c.strip()]
    if args.all_categories:
        return list(CATEGORY_IDS)
    return []


def run_all_categories_loso(
    args: argparse.Namespace, base_config: ExperimentConfig
) -> dict[str, object]:
    categories = _resolve_category_run_list(args)
    if not categories:
        raise ValueError("--all-categories or --categories requires category ids")
    unknown = [c for c in categories if c not in CATEGORY_IDS and c != "meme8"]
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}. Valid: {list(CATEGORY_IDS)}")

    aggregate: dict[str, object] = {"mode": "loso", "categories": {}}
    for cat in categories:
        print(f"\n################## CATEGORY: {cat} ##################")
        cat_args = argparse.Namespace(**{**vars(args), "category": cat, "all_categories": False})
        if cat == "meme8":
            config = apply_overrides(
                replace_config_category(base_config, None, get_legacy_meme8_symbols()),
                cat_args,
            )
        else:
            config = apply_overrides(apply_category(base_config, cat), cat_args)
        config.ensure_dirs()
        aggregate["categories"][cat] = run_loso(cat_args, config)

    return aggregate


def replace_config_category(
    config: ExperimentConfig,
    category_id: str | None,
    symbols: tuple[str, ...],
) -> ExperimentConfig:
    from dataclasses import replace

    return replace(config, category_id=category_id, symbols=symbols)


def main() -> None:
    args = parse_args()
    base_config = config_from_yaml(args.config)
    base_config.ensure_dirs()

    if args.all_categories or args.categories:
        if args.mode != "loso":
            raise ValueError("--all-categories/--categories only supported with --mode loso")
        summary = run_all_categories_loso(args, base_config)
    elif args.mode == "single":
        config = apply_overrides(base_config, args)
        summary = run_single(args, config)
    elif args.mode == "pooled":
        config = apply_overrides(base_config, args)
        summary = run_pooled(args, config)
    elif args.mode == "loso":
        config = apply_overrides(base_config, args)
        summary = run_loso(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

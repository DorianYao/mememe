#!/usr/bin/env python3
"""Run one LOSO experiment with parallel feature build + parallel held-out training.

Phase 1: per-symbol windows + all LOSO .npz (parallel threads, shared cache).
Phase 2: train+eval each held-out in parallel (--jobs workers).
Phase 3: aggregate summary.json from per-round metrics on disk.

Usage:
  LOSO_JOBS=4 python3 scripts/loso_parallel_train.py -- \\
    python3 main.py --mode loso --category bluechip --stage all --model mlp ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, config_from_yaml
from src.evaluate import aggregate_loso
from src.multi import build_all_loso_npz, loso_processed_path


def _parse_main_argv(argv: list[str]) -> tuple[argparse.Namespace, ExperimentConfig]:
    """Lightweight re-parse of main.py flags needed for symbol list + paths."""
    from main import apply_overrides, parse_args

    saved = sys.argv
    try:
        sys.argv = ["main.py", *argv]
        args = parse_args()
    finally:
        sys.argv = saved

    config = config_from_yaml(args.config) if args.config else ExperimentConfig()
    config = apply_overrides(config, args)
    return args, config


def _train_one(
    base_cmd: list[str],
    held: str,
    env: dict[str, str],
) -> tuple[str, int]:
    stage = env.get("LOSO_PARALLEL_STAGE", "train")
    cmd = [
        *base_cmd,
        "--held-out",
        held,
        "--stage",
        stage,
    ]
    if env.get("LOSO_FORCE_REEVAL") != "1" and stage == "train":
        cmd.append("--skip-existing")
    print(f"[parallel] train held-out={held}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return held, proc.returncode


def aggregate_from_disk(config: ExperimentConfig) -> dict[str, object]:
    rounds: dict[str, dict[str, dict[str, object]]] = {}
    for sym in config.symbols:
        tag_base = config.universe_tag("loso", sym)
        metrics_path = config.metrics_dir / f"{tag_base}_mlp_metrics.json"
        if metrics_path.exists():
            rounds[sym] = {"mlp": json.loads(metrics_path.read_text(encoding="utf-8"))}
    if not rounds:
        raise RuntimeError("No per-round metrics found; nothing to aggregate.")
    return aggregate_loso(rounds, config, list(config.symbols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel LOSO held-out trainer")
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("LOSO_JOBS", "1")),
        help="Max concurrent held-out training jobs (default: LOSO_JOBS env or 1)",
    )
    parser.add_argument(
        "main_cmd",
        nargs=argparse.REMAINDER,
        help="Command after --, e.g. python3 main.py --mode loso ...",
    )
    args = parser.parse_args()
    cmd = [c for c in args.main_cmd if c != "--"]
    if len(cmd) < 2 or "main.py" not in cmd:
        raise SystemExit("Usage: loso_parallel_train.py [--jobs N] -- python3 main.py --mode loso ...")

    main_idx = cmd.index("main.py")
    main_argv: list[str] = []
    skip = False
    for tok in cmd[main_idx + 1 :]:
        if skip:
            skip = False
            continue
        if tok in {"--held-out", "--stage"}:
            skip = True
            continue
        main_argv.append(tok)
    base_cmd = cmd[: main_idx + 1] + main_argv

    main_args, config = _parse_main_argv(main_argv)
    jobs = max(1, args.jobs)
    symbols = list(config.symbols)
    npz_default = int(os.environ.get("NPZ_JOBS", os.environ.get("FEATURE_WORKERS", str(jobs))))
    npz_workers = max(1, min(npz_default, 16, len(symbols)))

    env = os.environ.copy()
    env["PARALLEL_GPU_JOBS"] = str(jobs)

    print(
        f"[parallel] category={config.category_id} w={config.window_size} "
        f"k={config.label_k} train_jobs={jobs} npz_threads={npz_workers} symbols={len(symbols)}",
        flush=True,
    )

    build_all_loso_npz(
        config,
        max_workers=npz_workers,
        refresh=bool(main_args.refresh),
    )

    pending = [
        sym
        for sym in symbols
        if not (
            main_args.skip_existing
            and loso_processed_path(config, sym).exists()
            and (
                config.metrics_dir / f"{config.universe_tag('loso', sym)}_mlp_metrics.json"
            ).exists()
        )
    ]
    if not pending:
        print("[parallel] all held-out rounds already complete", flush=True)
    elif jobs == 1:
        for sym in pending:
            code = _train_one(base_cmd, sym, env)[1]
            if code != 0:
                raise SystemExit(code)
    else:
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_train_one, base_cmd, sym, env) for sym in pending]
            for fut in as_completed(futures):
                held, code = fut.result()
                if code != 0:
                    failures.append(held)
        if failures:
            raise SystemExit(f"Failed held-outs: {', '.join(sorted(failures))}")

    agg = aggregate_from_disk(config)
    print(f"[parallel] summary -> {agg['summary_path']}", flush=True)


if __name__ == "__main__":
    main()

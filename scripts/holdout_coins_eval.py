"""Download and evaluate external holdout meme coins (frozen k/w)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HOLDOUT = (
    "TURBOUSDT",
    "MOGUSDT",
    "POPCATUSDT",
    "NEIROUSDT",
)

DEVELOPMENT_UNIVERSE = (
    "DOGEUSDT",
    "SHIBUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "BONKUSDT",
    "FLOKIUSDT",
    "BOMEUSDT",
    "1000SATSUSDT",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="External holdout coin LOSO evaluation")
    parser.add_argument(
        "--holdout-symbols",
        default=",".join(DEFAULT_HOLDOUT),
        help="Comma-separated external symbols",
    )
    parser.add_argument("--window", type=int, default=192)
    parser.add_argument("--label-k", type=float, default=1.2)
    parser.add_argument("--split-mode", default="calendar", choices=["ratio", "calendar"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    holdouts = [s.strip().upper() for s in args.holdout_symbols.split(",") if s.strip()]
    dev = list(DEVELOPMENT_UNIVERSE)
    tag = "holdout_confirm_k12"

    for held in holdouts:
        if not args.skip_download and not args.dry_run:
            # Use single-symbol download; loso --stage download still builds features if npz missing.
            dl_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--mode",
                "single",
                "--symbol",
                held,
                "--stage",
                "download",
                "--refresh",
            ]
            subprocess.run(dl_cmd, cwd=PROJECT_ROOT, check=False)

        universe = ",".join(list(dev) + [held])
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--mode",
            "loso",
            "--stage",
            "all" if not args.skip_download else "train",
            "--model",
            args.model,
            "--symbols",
            universe,
            "--held-out",
            held,
            "--window-size",
            str(args.window),
            "--label-k",
            str(args.label_k),
            "--split-mode",
            args.split_mode,
            "--ablation-tag",
            tag,
            "--epochs",
            str(args.epochs),
        ]
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)

    summary = {
        "holdout_symbols": holdouts,
        "development_universe": dev,
        "window": args.window,
        "label_k": args.label_k,
        "split_mode": args.split_mode,
        "ablation_tag": tag,
        "frozen_hyperparams": True,
    }
    out = PROJECT_ROOT / "outputs" / "metrics" / "holdout_coins_eval.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()

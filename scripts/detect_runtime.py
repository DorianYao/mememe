#!/usr/bin/env python3
"""Recommend LOSO / feature / batch concurrency from GPU VRAM and CPU count.

Print shell assignments for detect_train_args.sh, or JSON for debugging.

Usage:
  python3 scripts/detect_runtime.py --shell
  python3 scripts/detect_runtime.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from effective_cpus import effective_cpus


def _gpu_info() -> tuple[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return "", 0.0
        props = torch.cuda.get_device_properties(0)
        name = props.name
        vram_gb = props.total_memory / (1024**3)
        return name, vram_gb
    except Exception:
        return "", 0.0


def recommend(*, ncpu: int | None = None) -> dict[str, int | str | float]:
    ncpu = ncpu if ncpu is not None else effective_cpus()
    gpu_name, vram_gb = _gpu_info()
    feat = min(max(1, ncpu - 1), 16)

    if vram_gb <= 0:
        loso = min(max(2, ncpu // 2), 8)
        batch = 1024
        npz = min(feat, 8)
        tier = "cpu"
    elif vram_gb >= 70:
        # A100 80GB, H100, etc.
        loso = 8
        batch = 65536
        npz = min(feat, 12)
        tier = "high"
    elif vram_gb >= 40:
        # A40 48GB, A6000 48GB, L40S
        loso = 6
        batch = 32768
        npz = min(feat, 10)
        tier = "mid"
    elif vram_gb >= 20:
        # L4 24GB, RTX 4090 24GB
        loso = 4
        batch = 16384
        npz = min(feat, 8)
        tier = "entry"
    else:
        loso = 3
        batch = 8192
        npz = min(feat, 6)
        tier = "small"

    return {
        "tier": tier,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "ncpu": ncpu,
        "loso_jobs": loso,
        "batch_size": batch,
        "feature_workers": feat,
        "npz_jobs": npz,
    }


def _shell(cfg: dict[str, object]) -> str:
    lines = [
        f'export LOSO_JOBS="${{LOSO_JOBS:-{cfg["loso_jobs"]}}}"',
        f'export FEATURE_WORKERS="${{FEATURE_WORKERS:-{cfg["feature_workers"]}}}"',
        f'export NPZ_JOBS="${{NPZ_JOBS:-{cfg["npz_jobs"]}}}"',
        f'BATCH_DEFAULT={cfg["batch_size"]}',
        f'RUNTIME_TIER="{cfg["tier"]}"',
        f'RUNTIME_GPU="{cfg["gpu_name"]}"',
        f'RUNTIME_VRAM_GB="{cfg["vram_gb"]}"',
        f'RUNTIME_NCPU="{cfg["ncpu"]}"',
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", action="store_true", help="Print bash eval lines")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    cfg = recommend()
    if args.json:
        print(json.dumps(cfg, indent=2))
    elif args.shell:
        print(_shell(cfg))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

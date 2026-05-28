"""Run label-feature decoupling ablations under walk-forward LOSO."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    {
        "tag": "decouple_sigma200",
        "extra": ["--label-volatility-window", "200", "--volatility-window", "50"],
    },
    {
        "tag": "decouple_no_vol_feat",
        "extra": [
            "--exclude-features",
            "volatility_50,atr_14_ratio,bb_position,macd_hist",
        ],
    },
    {
        "tag": "decouple_price_taker",
        "extra": [
            "--include-features",
            "log_return,upper_wick_ratio,lower_wick_ratio,body_ratio,body_abs_ratio,"
            "taker_buy_ratio,taker_buy_ratio_change",
        ],
    },
]


def main() -> None:
    base = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--mode",
        "loso",
        "--stage",
        "all",
        "--model",
        "mlp",
        "--window-size",
        "192",
        "--label-k",
        "1.2",
        "--split-mode",
        "calendar",
        "--ablation-tag",
    ]
    for run in RUNS:
        cmd = base + [run["tag"]] + run["extra"]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()

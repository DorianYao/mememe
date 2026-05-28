#!/usr/bin/env bash
# Validate w=192: confidence threshold + realistic PnL (no retraining).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/outputs/metrics/validate_w192.log"
mkdir -p "$(dirname "$LOG")"

echo "=== validate k=1.2 w=192 $(date -Iseconds) ===" | tee "$LOG"

echo "" | tee -a "$LOG"
echo ">>> confidence threshold" | tee -a "$LOG"
python3 scripts/confidence_threshold_scan.py --tag label_k12 --window 192 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> realistic PnL" | tee -a "$LOG"
python3 scripts/backtest_pnl_realistic.py --tag label_k12 --window 192 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo ">>> compare w=96 vs w=192 (realistic PnL @ tau=0.70)" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

metrics = Path("outputs/metrics")
for w in (96, 192):
    path = metrics / f"backtest_realistic_w96_label_k12.json".replace("_w96_", f"_w{w}_")
    if not path.exists():
        print(f"w={w}: {path.name} missing")
        continue
    data = json.loads(path.read_text())
    row = next(r for r in data["pooled"] if r["tau"] == 0.7)
    print(
        f"w={w}: AUC summary see window compare; "
        f"tau=0.70 net_bps={row['mean_net_return_bps']:.1f} "
        f"win={row['win_rate_net']:.3f} sharpe={row['sharpe_daily']:.2f} "
        f"maxDD={row['max_drawdown_net']:.3f} total_ret={row['total_return_net']:.3f}"
    )
PY

echo "=== validate finished $(date -Iseconds) ===" | tee -a "$LOG"

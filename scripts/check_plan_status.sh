#!/usr/bin/env bash
# Quick status for remote experiments (safe to run after SSH reconnect).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
M="$ROOT/outputs/metrics"

echo "=== tmux ==="
tmux ls 2>/dev/null || echo "(no tmux server)"

echo ""
echo "=== processes ==="
pgrep -af 'run_base_eco_validation|loso_parallel|main.py.*loso' 2>/dev/null | head -6 || echo "(none)"

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true

echo ""
echo "=== P0 strict base_eco k=1.5 w=224 ==="
if [[ -f "$M/base_eco16_15m_w224_loso_wf7085_label_k15_summary.json" ]]; then
  python3 -c "
import json
d=json.load(open('$M/base_eco16_15m_w224_loso_wf7085_label_k15_summary.json'))
print('  rounds:', d.get('n_rounds'), 'mean_auc:', round(d['mean']['mlp']['mean_test_roc_auc'],4))
"
else
  echo "  (no summary yet)"
fi
echo "  metrics: $(ls "$M"/*wf7085_label_k15*mlp_metrics.json 2>/dev/null | wc -l)/16"

echo ""
echo "=== P0 external holdout (holdout_k15_w224) ==="
for s in NEIROUSDT PNUTUSDT TURBOUSDT; do
  if compgen -G "$M"/*"${s}"*holdout_k15*mlp_metrics.json >/dev/null 2>&1; then
    python3 - "$M" "$s" <<'PY'
import json, glob, sys
mdir, sym = sys.argv[1], sys.argv[2]
p = sorted(glob.glob(f"{mdir}/*{sym}*holdout_k15*mlp_metrics.json"))[-1]
auc = json.load(open(p))["test_roc_auc"]
print(f"  {sym}: AUC {auc:.4f}")
PY
  else
    echo "  ${s}: pending"
  fi
done

echo ""
echo "=== P1 strict k=1.2 w=208 ==="
if compgen -G "$M"/*w208*wf7085*label_k12*summary.json >/dev/null 2>&1; then
  echo "  summary: $(ls "$M"/*w208*wf7085*label_k12*summary.json | tail -1)"
else
  echo "  metrics: $(ls "$M"/*w208*wf7085*label_k12*mlp_metrics.json 2>/dev/null | wc -l)/16"
fi

echo ""
echo "=== logs (tail) ==="
for f in base_eco_validation_resume.log base_eco_validation_plan.log; do
  if [[ -f "$M/$f" ]]; then
    echo "--- $f ---"
    tail -3 "$M/$f"
  fi
done

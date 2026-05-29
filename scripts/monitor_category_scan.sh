#!/usr/bin/env bash
# Poll category k×w scan progress; print REPORT lines for notifications.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CATEGORY="${1:-bluechip}"
SPLIT="${2:-ratio}"
POLL="${POLL_SEC:-90}"
LOG="${ROOT}/outputs/metrics/${CATEGORY}_kw_scan_${SPLIT}.log"

mapfile -t COMBOS < <(python3 "${ROOT}/scripts/category_kw_matrix.py" base)

seen_file="${ROOT}/outputs/metrics/.${CATEGORY}_kw_${SPLIT}_seen"
touch "$seen_file"

while true; do
  for combo in "${COMBOS[@]}"; do
    IFS=: read -r k w tag <<< "$combo"
    key="${k}_${w}_${tag}"
    grep -qx "$key" "$seen_file" 2>/dev/null && continue
    summary=$(compgen -G "${ROOT}/outputs/metrics/${CATEGORY}16_"*"_w${w}_loso_${SPLIT}_${tag}_summary.json" || true)
    if [[ -n "$summary" ]]; then
      n=$(python3 -c "import json; d=json.load(open('${summary}')); print(d.get('n_rounds',0))" 2>/dev/null || echo 0)
      if [[ "$n" -ge 16 ]]; then
        auc=$(python3 -c "import json; d=json.load(open('${summary}')); print(d.get('mean',{}).get('mlp',{}).get('mean_test_roc_auc',''))" 2>/dev/null || echo "")
        echo "REPORT: [DONE] ${CATEGORY} k=${k} w=${w} tag=${tag} AUC=${auc} file=$(basename "$summary") $(date -Iseconds)"
        echo "$key" >> "$seen_file"
      fi
    fi
  done
  if [[ -f "$LOG" ]] && grep -q "k×window scan finished" "$LOG" 2>/dev/null; then
    if ! grep -qx "SCAN_FINISHED" "$seen_file" 2>/dev/null; then
      echo "REPORT: [ALL DONE] ${CATEGORY} Phase A (${SPLIT}) scan finished $(date -Iseconds)"
      echo "SCAN_FINISHED" >> "$seen_file"
      exit 0
    fi
  fi
  sleep "$POLL"
done

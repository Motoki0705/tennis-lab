set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo 'usage: bash knowledge/runs/run-court-d2-probe-l4b-20261024/repro.sh RECOVERED_REVIEW_ROOT' >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURT_PROBE_RECOVERED="$1"
COURT_PROBE_REPRO=$(mktemp -d tmp/court-d2-l4b-repro.XXXXXX)
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
COURT_PROBE_REF=$(.venv/bin/python - "$SCRIPT_DIR/run.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["pushed_commit"])
PY
)
.venv/bin/python "$SCRIPT_DIR/analyze_saved.py" \
  --source-commit "$COURT_PROBE_REF" \
  --recovered "$COURT_PROBE_RECOVERED" \
  --geometry knowledge/runs/run-court-d2-probe-l4-20261024/evidence/selected-geometry.json \
  --output "$COURT_PROBE_REPRO/analysis.json" --review "$COURT_PROBE_REPRO/review"
cmp "$COURT_PROBE_REPRO/analysis.json" "$SCRIPT_DIR/evidence/analysis.json"
.venv/bin/python "$SCRIPT_DIR/reprice_costs.py" \
  --analysis "$COURT_PROBE_REPRO/analysis.json" \
  --setup "$COURT_PROBE_RECOVERED/setup" --output "$COURT_PROBE_REPRO/costs.json"
cmp "$COURT_PROBE_REPRO/costs.json" "$SCRIPT_DIR/evidence/costs.json"
.venv/bin/python "$SCRIPT_DIR/summarize_timing.py" \
  --recovered "$COURT_PROBE_RECOVERED" \
  --render-status "$SCRIPT_DIR/evidence/render-probe-status-saved.json" \
  --collect-status "$SCRIPT_DIR/evidence/collect-review-status-final.json" \
  --drive-inspect "$SCRIPT_DIR/evidence/drive-output-inspect.json" \
  --output "$COURT_PROBE_REPRO/timing.json"
cmp "$COURT_PROBE_REPRO/timing.json" "$SCRIPT_DIR/evidence/timing.json"
.venv/bin/python knowledge/runs/run-court-d2-probe-l4-20261024/verify_probe_paths.py \
  --scratch "$COURT_PROBE_REPRO/path-contract" \
  --output "$COURT_PROBE_REPRO/path-contract.json"
cmp "$COURT_PROBE_REPRO/path-contract.json" "$SCRIPT_DIR/evidence/path-contract.json"
echo "CPU evidence replay passed: $COURT_PROBE_REPRO"

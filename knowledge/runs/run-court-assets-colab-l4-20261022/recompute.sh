#!/usr/bin/env bash
# Recompute CPU-only comparisons from verified Drive downloads; run at repo root.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
if [[ $# != 4 ]]; then
  echo "Usage: bash recompute.sh ASSET_ROOT LOCAL_RAW_ROOT L4_RAW_ROOT NEW_OUTPUT_DIR" >&2
  exit 2
fi
PROBE_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ASSETS="$1"
PROBE_LOCAL="$2"
PROBE_COLAB="$3"
PROBE_OUTPUT="$4"
if [[ -e "$PROBE_OUTPUT" ]]; then
  echo "Output already exists: $PROBE_OUTPUT" >&2
  exit 2
fi
mkdir -p -- "$PROBE_OUTPUT"
.venv/bin/python "$PROBE_BUNDLE/court_assets_visibility.py" --scenes "$PROBE_ASSETS/scenes" --renders "$PROBE_LOCAL" --output "$PROBE_OUTPUT/local-visibility.json"
.venv/bin/python "$PROBE_BUNDLE/court_assets_visibility.py" --scenes "$PROBE_ASSETS/scenes" --renders "$PROBE_COLAB" --output "$PROBE_OUTPUT/colab-visibility.json"
compare() {
  local label="$1" reference="$2" candidate="$3" reference_visibility="$4" candidate_visibility="$5" reference_repeat="$6" candidate_repeat="$7"
  .venv/bin/python "$PROBE_BUNDLE/court_assets_compare.py" \
    --scenes "$PROBE_ASSETS/scenes" --reference "$reference" --candidate "$candidate" \
    --reference-visibility "$reference_visibility" --candidate-visibility "$candidate_visibility" \
    --reference-repeat "$reference_repeat" --candidate-repeat "$candidate_repeat" \
    --output "$PROBE_OUTPUT/$label-comparison.json" > "$PROBE_OUTPUT/$label-summary.json"
}
compare cross-gpu "$PROBE_LOCAL" "$PROBE_COLAB" "$PROBE_OUTPUT/local-visibility.json" "$PROBE_OUTPUT/colab-visibility.json" repeat-0 repeat-0
compare cross-gpu-repeat "$PROBE_LOCAL" "$PROBE_COLAB" "$PROBE_OUTPUT/local-visibility.json" "$PROBE_OUTPUT/colab-visibility.json" repeat-1 repeat-1
compare local-repeat "$PROBE_LOCAL" "$PROBE_LOCAL" "$PROBE_OUTPUT/local-visibility.json" "$PROBE_OUTPUT/local-visibility.json" repeat-0 repeat-1
compare colab-repeat "$PROBE_COLAB" "$PROBE_COLAB" "$PROBE_OUTPUT/colab-visibility.json" "$PROBE_OUTPUT/colab-visibility.json" repeat-0 repeat-1
.venv/bin/python "$PROBE_BUNDLE/court_assets_outliers.py" --reference "$PROBE_LOCAL" --candidate "$PROBE_COLAB" --comparison "$PROBE_OUTPUT/cross-gpu-comparison.json" --visibility "$PROBE_OUTPUT/local-visibility.json" --output "$PROBE_OUTPUT/outlier-audit.json" > "$PROBE_OUTPUT/outlier-summary.txt"
.venv/bin/python "$PROBE_BUNDLE/court_assets_figure.py" --reference "$PROBE_LOCAL" --candidate "$PROBE_COLAB" --comparison "$PROBE_OUTPUT/cross-gpu-comparison.json" --visibility "$PROBE_OUTPUT/local-visibility.json" --output "$PROBE_OUTPUT/comparison.png"

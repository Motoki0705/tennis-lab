#!/usr/bin/env bash
set -euo pipefail

# Preprocess tennis JSON scenes into npz/memmap arrays.
#
# This wrapper calls src/cli/preprocess_tennis_memmap.py using the
# dataset_root/name from configs/datasets/tennis_pose_sim.yaml by default.
#
# Usage:
#   ./scripts/preprocess_tennis_memmap.sh
#   ./scripts/preprocess_tennis_memmap.sh --overwrite
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_CFG="${DATASET_CFG:-configs/datasets/tennis_pose_sim.yaml}"

cd "${ROOT_DIR}"

DATASET_ROOT=$(python - << 'PY'
import sys
from pathlib import Path

try:
    import yaml  # type: ignore[import]
except Exception:
    sys.exit("ERROR: pyyaml is required to parse dataset config (pip install pyyaml)")

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

root = cfg.get("root", "data/tennis_autogen")
name = cfg.get("name")
if not name:
    sys.exit("ERROR: 'name' must be set in dataset config")

print(root)
print(name, file=sys.stderr)
PY "${DATASET_CFG}")

DATASET_NAME=$(python - << 'PY'
import sys
from pathlib import Path

try:
    import yaml  # type: ignore[import]
except Exception:
    sys.exit(1)

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

name = cfg.get("name")
print(name or "")
PY "${DATASET_CFG}")

if [[ -z "${DATASET_NAME}" ]]; then
  echo "ERROR: failed to read dataset name from ${DATASET_CFG}" >&2
  exit 1
fi

python src/cli/preprocess_tennis_memmap.py \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  "$@"


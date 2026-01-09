#!/usr/bin/env bash
set -euo pipefail

: <<'DOC'
Run a BLCS hyperparameter sweep with Hydra-managed training.

Example:
  UV_CACHE_DIR=agents_workspace/tmp_cache/uv_cache \
  RUN_ROOT=outputs/blcs/sweep_$(date +%Y-%m-%d_%H-%M-%S) \
  bash src/blcs/scripts/run_hparam_sweep.sh

Config entry point: src/blcs/configs/train.yaml
DOC

BATCH_SIZES=${BATCH_SIZES:-"16 32 64 128"}
HIDDEN_DIMS=${HIDDEN_DIMS:-"128 256 384 512"}
NUM_LAYERS=${NUM_LAYERS:-"2 4 6 8"}
SEEDS=${SEEDS:-"42"}

BASE_BATCH_SIZE=${BASE_BATCH_SIZE:-32}
BASE_MAX_EPOCHS=${BASE_MAX_EPOCHS:-200}
BASE_WARMUP_STEPS=${BASE_WARMUP_STEPS:-2000}
NUM_TRAIN_SCENES=${NUM_TRAIN_SCENES:-80000}

NUM_HEADS=${NUM_HEADS:-8}
NUM_KV_HEADS=${NUM_KV_HEADS:-2}
GPUS=${GPUS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

RUN_ROOT=${RUN_ROOT:-"outputs/blcs/sweep_$(date +%Y-%m-%d_%H-%M-%S)"}
RESULTS_DIR="${RUN_ROOT}/results"
RESULTS_JSONL="${RESULTS_DIR}/metrics.jsonl"
RESULTS_CSV="${RESULTS_DIR}/summary.csv"

USE_UV=${USE_UV:-true}
UV_BIN=${UV_BIN:-uv}
UV_CACHE_DIR=${UV_CACHE_DIR:-agents_workspace/tmp_cache/uv_cache}

PYTHON_BIN=${PYTHON_BIN:-python}

if [[ "${USE_UV}" == "true" ]]; then
  PYTHON_RUN=("${UV_BIN}" --cache-dir "${UV_CACHE_DIR}" run --no-sync python)
else
  PYTHON_RUN=("${PYTHON_BIN}")
fi

mkdir -p "${RESULTS_DIR}"

if (( NUM_HEADS % NUM_KV_HEADS != 0 )); then
  echo "num_heads (${NUM_HEADS}) must be divisible by num_kv_heads (${NUM_KV_HEADS})." >&2
  exit 1
fi

compute_schedule() {
  local batch_size="$1"
  "${PYTHON_BIN}" - "${BASE_MAX_EPOCHS}" "${BASE_BATCH_SIZE}" "${BASE_WARMUP_STEPS}" \
    "${NUM_TRAIN_SCENES}" "${batch_size}" <<'PY'
import math
import sys

base_max_epochs = int(sys.argv[1])
base_batch = int(sys.argv[2])
base_warmup = int(sys.argv[3])
num_train = int(sys.argv[4])
batch = int(sys.argv[5])

base_steps_per_epoch = max(1, num_train // base_batch)
base_total_steps = base_steps_per_epoch * base_max_epochs

steps_per_epoch = max(1, num_train // batch)
max_epochs = max(1, int(round(base_total_steps / steps_per_epoch)))
total_steps = steps_per_epoch * max_epochs

warmup_ratio = base_warmup / base_total_steps if base_total_steps > 0 else 0.0
warmup_steps = max(1, int(round(warmup_ratio * total_steps)))

print(max_epochs, warmup_steps, total_steps, steps_per_epoch)
PY
}

BATCH_SIZES=${BATCH_SIZES//,/ }
HIDDEN_DIMS=${HIDDEN_DIMS//,/ }
NUM_LAYERS=${NUM_LAYERS//,/ }
SEEDS=${SEEDS//,/ }

read -r -a BATCH_SIZES_ARR <<< "${BATCH_SIZES}"
read -r -a HIDDEN_DIMS_ARR <<< "${HIDDEN_DIMS}"
read -r -a NUM_LAYERS_ARR <<< "${NUM_LAYERS}"
read -r -a SEEDS_ARR <<< "${SEEDS}"

echo "Sweep output: ${RUN_ROOT}"
echo "Results JSONL: ${RESULTS_JSONL}"

for batch_size in "${BATCH_SIZES_ARR[@]}"; do
  read -r max_epochs warmup_steps total_steps steps_per_epoch < <(compute_schedule "${batch_size}")

  for hidden_dim in "${HIDDEN_DIMS_ARR[@]}"; do
    if (( hidden_dim % NUM_HEADS != 0 )); then
      echo "Skipping hidden_dim=${hidden_dim} (not divisible by num_heads=${NUM_HEADS})." >&2
      continue
    fi

    for num_layers in "${NUM_LAYERS_ARR[@]}"; do
      for seed in "${SEEDS_ARR[@]}"; do
        run_name="bs${batch_size}_hd${hidden_dim}_l${num_layers}_seed${seed}"
        output_dir="${RUN_ROOT}/${run_name}"

        echo "Running ${run_name} (epochs=${max_epochs}, warmup=${warmup_steps})"

        "${PYTHON_RUN[@]}" -m src.blcs.scripts.train \
          "run.output_dir=${output_dir}" \
          "run.seed=${seed}" \
          "run.gpus=${GPUS}" \
          "run.fast_dev_run=false" \
          "data.batch_size=${batch_size}" \
          "data.num_workers=${NUM_WORKERS}" \
          "model.hidden_dim=${hidden_dim}" \
          "model.num_layers=${num_layers}" \
          "model.num_heads=${NUM_HEADS}" \
          "model.num_kv_heads=${NUM_KV_HEADS}" \
          "training.max_epochs=${max_epochs}" \
          "training.warmup_steps=${warmup_steps}"

        RUN_OUTPUT_DIR="${output_dir}" \
        RUN_NAME="${run_name}" \
        RUN_BATCH_SIZE="${batch_size}" \
        RUN_HIDDEN_DIM="${hidden_dim}" \
        RUN_NUM_LAYERS="${num_layers}" \
        RUN_SEED="${seed}" \
        RUN_MAX_EPOCHS="${max_epochs}" \
        RUN_WARMUP_STEPS="${warmup_steps}" \
        RUN_TOTAL_STEPS="${total_steps}" \
        RUN_STEPS_PER_EPOCH="${steps_per_epoch}" \
        RUN_NUM_HEADS="${NUM_HEADS}" \
        RUN_NUM_KV_HEADS="${NUM_KV_HEADS}" \
        RUN_NUM_TRAIN_SCENES="${NUM_TRAIN_SCENES}" \
        RESULTS_JSONL="${RESULTS_JSONL}" \
        "${PYTHON_RUN[@]}" - <<'PY'
import json
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
log_root = output_dir / "logs"
version_dirs = sorted(log_root.glob("version_*"), key=lambda p: p.stat().st_mtime)
if not version_dirs:
    raise SystemExit(f"No TensorBoard logs found in {log_root}")
latest_version = version_dirs[-1]

acc = EventAccumulator(str(latest_version))
acc.Reload()

scalars = {}
for tag in acc.Tags().get("scalars", []):
    events = acc.Scalars(tag)
    if events:
        scalars[tag] = events[-1].value

record = {
    "run": {
        "run_name": os.environ["RUN_NAME"],
        "output_dir": str(output_dir),
        "batch_size": int(os.environ["RUN_BATCH_SIZE"]),
        "hidden_dim": int(os.environ["RUN_HIDDEN_DIM"]),
        "num_layers": int(os.environ["RUN_NUM_LAYERS"]),
        "seed": int(os.environ["RUN_SEED"]),
        "max_epochs": int(os.environ["RUN_MAX_EPOCHS"]),
        "warmup_steps": int(os.environ["RUN_WARMUP_STEPS"]),
        "total_steps": int(os.environ["RUN_TOTAL_STEPS"]),
        "steps_per_epoch": int(os.environ["RUN_STEPS_PER_EPOCH"]),
        "num_heads": int(os.environ["RUN_NUM_HEADS"]),
        "num_kv_heads": int(os.environ["RUN_NUM_KV_HEADS"]),
        "num_train_scenes": int(os.environ["RUN_NUM_TRAIN_SCENES"]),
    },
    "scalars": scalars,
    "test_metrics": {k: v for k, v in scalars.items() if k.startswith("test/")},
}

with open(os.environ["RESULTS_JSONL"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
      done
    done
  done
done

"${PYTHON_RUN[@]}" - "${RESULTS_JSONL}" "${RESULTS_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

rows = []
scalar_keys = set()

with jsonl_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        rows.append(record)
        scalar_keys.update(record.get("scalars", {}).keys())

scalar_keys = sorted(scalar_keys)
run_keys = [
    "run_name",
    "output_dir",
    "batch_size",
    "hidden_dim",
    "num_layers",
    "seed",
    "max_epochs",
    "warmup_steps",
    "total_steps",
    "steps_per_epoch",
    "num_heads",
    "num_kv_heads",
    "num_train_scenes",
]

fieldnames = run_keys + scalar_keys

with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for record in rows:
        row = {}
        row.update(record.get("run", {}))
        scalars = record.get("scalars", {})
        for key in scalar_keys:
            row[key] = scalars.get(key)
        writer.writerow(row)
PY

echo "Sweep complete."
echo "Aggregated results: ${RESULTS_CSV}"

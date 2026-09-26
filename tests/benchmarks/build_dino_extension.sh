#!/usr/bin/env bash
# Build DINO's MultiScaleDeformableAttention extension into a run-owned directory.
#
# Usage: build_dino_extension.sh <asset_root> <output_dir>
#   <asset_root>  root whose third_party/DINO submodule is initialized
#   <output_dir>  receives lib/ (put it on PYTHONPATH), temp/ and cache/
#
# Nothing is written into the repository; the build records its inputs in
# <output_dir>/build.json. CUDA_HOME and TORCH_CUDA_ARCH_LIST default to the
# local RTX 5060 Ti toolchain and can be overridden.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <asset_root> <output_dir>" >&2
    exit 2
fi
code_root="$(cd "$(dirname "$0")/../.." && pwd)"
asset_root="$(cd "$1" && pwd)"
mkdir -p "$2"
output="$(cd "$2" && pwd)"
python="${code_root}/.venv/bin/python"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
config="$("$python" - "$code_root" "$asset_root" "$output" <<'PY'
import json
import sys

code_root, asset_root, output = sys.argv[1:]
print(json.dumps({
    "paths": {
        "project_root": code_root,
        "data_root": f"{asset_root}/data",
        "checkpoint_root": f"{asset_root}/ckpt",
        "artifact_root": output,
        "output_root": output,
        "cache_root": f"{output}/cache",
        "external_asset_root": f"{asset_root}/third_party",
    },
    "source_role": "external_asset",
    "source": "DINO/models/dino/ops/src",
    "destination_role": "cache",
    "destination": "dino_ops_sources",
    "compressed_time_local_bindings": "src/utils/models/components/ops/compressed_time_local/bindings.cpp",
    "compressed_time_local_kernels": "src/utils/models/components/ops/compressed_time_local/kernels.cu",
}, sort_keys=True))
PY
)"
"$python" - "$output/build.json" "$config" <<'PY'
import json
import os
import sys

path, config = sys.argv[1:]
keys = ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS")
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"config": json.loads(config), "environment": {key: os.environ[key] for key in keys}}, handle, indent=2, sort_keys=True)
PY
cd "$code_root"
CUDA_VISIBLE_DEVICES="" TENNIS_LAB_BUILD_CUDA_OPS=1 TENNIS_LAB_CUDA_OPS_BUILD_TARGET=all \
    TENNIS_LAB_DINO_OPS_BUILD_CONFIG="$config" \
    "$python" setup.py build_ext --build-lib "$output/lib" --build-temp "$output/temp" --force
test -f "$output/lib/MultiScaleDeformableAttention"*.so
echo "[build_dino_extension] built $(ls "$output/lib"/MultiScaleDeformableAttention*.so)"

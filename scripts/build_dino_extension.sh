#!/usr/bin/env bash
# Build DINO's MultiScaleDeformableAttention extension into a run-owned directory.
#
# Usage (from the root of the checkout to build): build_dino_extension.sh <asset_root> <output_dir>
#   <asset_root>  root whose third_party/DINO submodule is initialized
#   <output_dir>  receives lib/ (put it on PYTHONPATH), temp/ and cache/
#
# The code (setup.py and the ops build contract) is the git checkout of the
# current directory, so knowledge run bundles can carry a copy of this script.
# All generated files live in <output_dir>; inputs are recorded in build.json.
# Toolchain/architecture selection is explicit. See docs/installation/README.md.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <asset_root> <output_dir>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
asset_root="$(cd "$1" && pwd)"
mkdir -p "$2"
output="$(cd "$2" && pwd)"
python="${code_root}/.venv/bin/python"
: "${CUDA_HOME:?Set CUDA_HOME to your CUDA 13 toolkit (see docs/installation/README.md)}"
: "${TORCH_CUDA_ARCH_LIST:?Set TORCH_CUDA_ARCH_LIST for your GPU (see docs/installation/README.md)}"
export CUDA_HOME TORCH_CUDA_ARCH_LIST
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "nvcc is missing: $CUDA_HOME/bin/nvcc" >&2
    exit 1
fi
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
import subprocess
import sys
import torch

path, config = sys.argv[1:]
keys = ("CUDA_HOME", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS")
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"config": json.loads(config), "environment": {key: os.environ[key] for key in keys},
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "nvcc": subprocess.check_output([os.environ["CUDA_HOME"] + "/bin/nvcc", "--version"], text=True)},
              handle, indent=2, sort_keys=True)
PY
cd "$code_root"
CUDA_VISIBLE_DEVICES="" TENNIS_LAB_BUILD_CUDA_OPS=1 TENNIS_LAB_CUDA_OPS_BUILD_TARGET=all \
    TENNIS_LAB_DINO_OPS_BUILD_CONFIG="$config" \
    "$python" setup.py build_ext --build-lib "$output/lib" --build-temp "$output/temp" --force \
    egg_info --egg-base "$output"
test -f "$output/lib/MultiScaleDeformableAttention"*.so
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$output/lib" "$python" -c \
    'from src.submodules.models import validate_dino_extension; print(validate_dino_extension())'
echo "[build_dino_extension] built $(ls "$output/lib"/MultiScaleDeformableAttention*.so)"

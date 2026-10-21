#!/usr/bin/env bash
# Preserve the benchmark's historical local defaults; the implementation and
# public installation instructions live in scripts/ and docs/installation/.
set -euo pipefail
code_root="$(git rev-parse --show-toplevel)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
    exec bash "$code_root/scripts/build_dino_extension.sh" "$@"

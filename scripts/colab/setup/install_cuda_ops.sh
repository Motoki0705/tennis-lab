#!/usr/bin/env bash

# Source-only module for building the CUDA operation required by track-query models.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "[install_cuda_ops] This file is a setup module and must be sourced by a train script." >&2
    exit 2
fi

_compressed_time_local_import_probe() {
    local repo_root="$1"

    (
        cd "${repo_root}"
        python -c \
            "from src.utils.models.components.ops.loader import require_compressed_time_local_cuda_extension; require_compressed_time_local_cuda_extension()"
    )
}

_colab_cuda_build_config() {
    local repo_root="$1"

    python - "${repo_root}" <<'PY'
import json
import sys

repository_root = sys.argv[1]
print(
    json.dumps(
        {
            "paths": {
                "project_root": repository_root,
                "data_root": "data",
                "checkpoint_root": "ckpt",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": ".cache",
                "external_asset_root": "third_party",
            },
            "source_role": "external_asset",
            "source": "DINO/ops/src",
            "destination_role": "cache",
            "destination": "dino_ops/src",
            "moe_bindings": "src/utils/models/components/ops/moe/csrc/moe.cpp",
            "moe_kernels": "src/utils/models/components/ops/moe/csrc/moe_cuda.cu",
            "time_local_bindings": "src/utils/models/components/ops/time_local/csrc/time_local.cpp",
            "time_local_kernels": "src/utils/models/components/ops/time_local/csrc/time_local_cuda.cu",
            "compressed_time_local_bindings": "src/utils/models/components/ops/compressed_time_local/bindings.cpp",
            "compressed_time_local_kernels": "src/utils/models/components/ops/compressed_time_local/kernels.cu",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
}

_colab_cuda_build_signature() {
    local repository_revision="$1"
    local cuda_home="$2"

    python - "${repository_revision}" "${cuda_home}" <<'PY'
import json
import platform
import sys

import torch

device_capability = None
if torch.cuda.is_available():
    device_capability = list(torch.cuda.get_device_capability())
print(
    json.dumps(
        {
            "schema": 1,
            "repository_revision": sys.argv[1],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_home": sys.argv[2],
            "device_capability": device_capability,
            "build_target": "compressed_time_local",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
}

install_colab_cuda_ops() {
    if [[ "$#" -ne 1 ]]; then
        echo "[install_cuda_ops] usage: install_colab_cuda_ops <repo_root>" >&2
        return 2
    fi

    local requested_root="$1"
    local max_jobs="${CUDA_OPS_MAX_JOBS-2}"
    if [[ ! "${max_jobs}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[install_cuda_ops] CUDA_OPS_MAX_JOBS must be a positive integer; got '${max_jobs}'." >&2
        return 2
    fi
    if [[ ! -d "${requested_root}" ]]; then
        echo "[install_cuda_ops] repository root does not exist: ${requested_root}" >&2
        return 2
    fi

    local repo_root
    repo_root="$(cd "${requested_root}" && pwd -P)"
    if [[ "${repo_root}" == "/" || ! -f "${repo_root}/setup.py" ]]; then
        echo "[install_cuda_ops] repository root must contain setup.py: ${repo_root}" >&2
        return 2
    fi
    if ! command -v python >/dev/null 2>&1; then
        echo "[install_cuda_ops] python is required to build the CUDA operation." >&2
        return 1
    fi

    local cuda_home
    if ! cuda_home="$(python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

if torch.version.cuda is None:
    raise RuntimeError("The installed PyTorch build does not include CUDA support.")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access a CUDA device in this Colab runtime.")
if CUDA_HOME is None:
    raise RuntimeError("PyTorch could not resolve CUDA_HOME for extension compilation.")
print(CUDA_HOME)
PY
    )"; then
        echo "[install_cuda_ops] CUDA toolchain validation failed." >&2
        return 1
    fi
    if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
        echo "[install_cuda_ops] nvcc is missing or not executable: ${cuda_home}/bin/nvcc" >&2
        return 1
    fi
    if ! command -v g++ >/dev/null 2>&1; then
        echo "[install_cuda_ops] g++ is required to compile the CUDA operation." >&2
        return 1
    fi
    if ! command -v git >/dev/null 2>&1; then
        echo "[install_cuda_ops] git is required to identify the repository revision." >&2
        return 1
    fi

    local repository_revision
    if ! repository_revision="$(git -C "${repo_root}" rev-parse HEAD)"; then
        echo "[install_cuda_ops] failed to resolve the repository revision." >&2
        return 1
    fi
    local signature
    if ! signature="$(_colab_cuda_build_signature "${repository_revision}" "${cuda_home}")"; then
        echo "[install_cuda_ops] failed to describe the active CUDA runtime." >&2
        return 1
    fi
    local cache_dir="${repo_root}/.cache/colab_cuda_ops"
    local signature_path="${cache_dir}/build_signature.json"
    if ! mkdir -p "${cache_dir}"; then
        echo "[install_cuda_ops] failed to create build cache: ${cache_dir}" >&2
        return 1
    fi

    if [[ -f "${signature_path}" && "$(<"${signature_path}")" == "${signature}" ]]; then
        if _compressed_time_local_import_probe "${repo_root}"; then
            echo "[install_cuda_ops] compressed_time_local CUDA operation is already built for this runtime."
            return 0
        fi
        echo "[install_cuda_ops] cached build failed its import probe; rebuilding."
    fi

    rm -f "${signature_path}"
    echo "[install_cuda_ops] installing the CUDA extension build dependency..."
    if ! python -m pip install ninja; then
        echo "[install_cuda_ops] failed to install ninja." >&2
        return 1
    fi

    local build_config
    if ! build_config="$(_colab_cuda_build_config "${repo_root}")"; then
        echo "[install_cuda_ops] failed to serialize the CUDA build contract." >&2
        return 1
    fi
    echo "[install_cuda_ops] building compressed_time_local with MAX_JOBS=${max_jobs}..."
    if ! (
        cd "${repo_root}"
        MAX_JOBS="${max_jobs}" \
        TENNIS_LAB_BUILD_CUDA_OPS=1 \
        TENNIS_LAB_CUDA_OPS_BUILD_TARGET=compressed_time_local \
        TENNIS_LAB_DINO_OPS_BUILD_CONFIG="${build_config}" \
        python setup.py build_ext --inplace --force
    ); then
        echo "[install_cuda_ops] compressed_time_local CUDA operation build failed." >&2
        return 1
    fi
    if ! _compressed_time_local_import_probe "${repo_root}"; then
        echo "[install_cuda_ops] built extension failed its import probe." >&2
        return 1
    fi

    local signature_tmp="${signature_path}.tmp"
    if ! printf '%s\n' "${signature}" > "${signature_tmp}" \
        || ! mv "${signature_tmp}" "${signature_path}"; then
        echo "[install_cuda_ops] failed to record the successful build signature." >&2
        return 1
    fi
    echo "[install_cuda_ops] compressed_time_local CUDA operation is ready."
}

#!/usr/bin/env bash
set -euo pipefail

readonly NHT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly NHT_UPSTREAM="${NHT_ROOT}/upstream"
readonly NHT_PYTHON="${NHT_ROOT}/.venv/bin/python"

# Runtime directories stay local; pins and launch behavior stay reviewable.
# shellcheck disable=SC1091
source "${NHT_ROOT}/pins.env"

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Required command is unavailable: %s\n' "${command_name}" >&2
    exit 1
  fi
}

verify_checkout() {
  if [[ ! -d "${NHT_UPSTREAM}/.git" ]]; then
    printf 'NHT checkout is absent: %s\n' "${NHT_UPSTREAM}" >&2
    exit 1
  fi

  test "$(git -C "${NHT_UPSTREAM}" rev-parse HEAD)" = "${NHT_COMMIT}"
  test "$(git -C "${NHT_UPSTREAM}/gsplat" rev-parse HEAD)" = "${GSPLAT_COMMIT}"
  test \
    "$(git -C "${NHT_UPSTREAM}/gsplat/gsplat/cuda/csrc/third_party/glm" rev-parse HEAD)" \
    = "${GLM_COMMIT}"

  if [[ -n "$(git -C "${NHT_UPSTREAM}" status --short)" ]]; then
    printf 'Refusing to use a modified NHT checkout\n' >&2
    git -C "${NHT_UPSTREAM}" status --short >&2
    exit 1
  fi
}

configure_cuda_build() {
  require_command gcc
  require_command g++

  local nvcc_bin="${CUDA_TOOLKIT_ROOT}/bin/nvcc"
  if [[ ! -x "${nvcc_bin}" ]]; then
    printf 'Pinned CUDA compiler is unavailable: %s\n' "${nvcc_bin}" >&2
    exit 1
  fi
  if ! "${nvcc_bin}" --version | grep -F 'release 13.0' >/dev/null; then
    printf 'Pinned CUDA compiler is not release 13.0: %s\n' "${nvcc_bin}" >&2
    exit 1
  fi
  export CUDA_HOME="${CUDA_TOOLKIT_ROOT}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CC
  CC="$(command -v gcc)"
  export CXX
  CXX="$(command -v g++)"
  export TORCH_CUDA_ARCH_LIST
  export TCNN_CUDA_ARCHITECTURES
  export MAX_JOBS="${MAX_JOBS:-2}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"

  local cuda_driver_link_dir=""
  local candidate
  for candidate in \
    /usr/lib/wsl/lib \
    "${CUDA_HOME}/lib64/stubs" \
    "${CUDA_HOME}/targets/x86_64-linux/lib/stubs"; do
    if [[ -f "${candidate}/libcuda.so" ]]; then
      cuda_driver_link_dir="${candidate}"
      break
    fi
  done
  if [[ -z "${cuda_driver_link_dir}" ]]; then
    printf 'Could not locate an unversioned libcuda.so for extension linking\n' >&2
    exit 1
  fi
  export LIBRARY_PATH="${cuda_driver_link_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
}

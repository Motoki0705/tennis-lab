#!/usr/bin/env bash
set -euo pipefail

# Fail closed if the pinned runtime or checkout is unavailable or modified.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python3 "${NHT_ROOT}/train.py" "$@"
fi

verify_checkout
configure_cuda_build
if [[ ! -x "${NHT_PYTHON}" ]]; then
  printf 'Run %s/setup.sh before training\n' "${NHT_ROOT}" >&2
  exit 1
fi

exec "${NHT_PYTHON}" "${NHT_ROOT}/train.py" "$@"

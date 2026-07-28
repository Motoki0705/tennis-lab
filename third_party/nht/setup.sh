#!/usr/bin/env bash
set -euo pipefail

# Build an isolated, content-pinned NHT runtime without touching tennis-lab's venv.
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

require_command git
require_command uv
configure_cuda_build

new_checkout=false
if [[ ! -e "${NHT_UPSTREAM}" ]]; then
  clone_args=()
  if [[ -n "${NHT_SEED_REPOSITORY:-}" ]]; then
    if [[ ! -d "${NHT_SEED_REPOSITORY}/.git" ]]; then
      printf 'NHT_SEED_REPOSITORY is not a git checkout: %s\n' \
        "${NHT_SEED_REPOSITORY}" >&2
      exit 1
    fi
    clone_args+=(--reference-if-able "${NHT_SEED_REPOSITORY}" --dissociate)
  fi
  git clone "${clone_args[@]}" --no-checkout \
    "${NHT_REPOSITORY}" "${NHT_UPSTREAM}"
  new_checkout=true
elif [[ ! -d "${NHT_UPSTREAM}/.git" ]]; then
  printf 'Refusing to overwrite non-git path: %s\n' "${NHT_UPSTREAM}" >&2
  exit 1
fi

actual_remote="$(git -C "${NHT_UPSTREAM}" remote get-url origin)"
if [[ "${actual_remote}" != "${NHT_REPOSITORY}" ]]; then
  printf 'Unexpected NHT origin: %s\n' "${actual_remote}" >&2
  exit 1
fi
if [[ "${new_checkout}" == false ]] &&
  [[ -n "$(git -C "${NHT_UPSTREAM}" status --short)" ]]; then
  printf 'Refusing setup with a modified NHT checkout\n' >&2
  exit 1
fi

git -C "${NHT_UPSTREAM}" fetch origin "${NHT_COMMIT}"
git -C "${NHT_UPSTREAM}" checkout --detach "${NHT_COMMIT}"
git -C "${NHT_UPSTREAM}" submodule sync --recursive
git -C "${NHT_UPSTREAM}" submodule update --init --recursive
verify_checkout

if [[ ! -x "${NHT_PYTHON}" ]]; then
  uv venv --python "${PYTHON_VERSION}" --prompt tennis-lab-nht \
    "${NHT_ROOT}/.venv"
fi

uv pip install --python "${NHT_PYTHON}" \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
uv pip install --python "${NHT_PYTHON}" \
  "setuptools>64,<80" wheel ninja "numpy<2.0.0" rich
uv pip install --python "${NHT_PYTHON}" --no-build-isolation \
  -e "${NHT_UPSTREAM}"

if "${NHT_PYTHON}" - "${NHT_UPSTREAM}/gsplat/gsplat/csrc.so" <<'PY'
from pathlib import Path
import sys

from gsplat import csrc

expected = Path(sys.argv[1]).resolve()
actual = Path(csrc.__file__).resolve()
assert actual == expected, (actual, expected)
print("Reusing pinned NHT gsplat CUDA module:", actual)
PY
then
  printf 'Pinned gsplat CUDA module is already usable; skipping rebuild\n'
else
  uv pip install --python "${NHT_PYTHON}" --no-build-isolation \
    -e "${NHT_UPSTREAM}/gsplat"
fi

uv pip install --python "${NHT_PYTHON}" --no-build-isolation \
  -r "${NHT_ROOT}/requirements.in"

mkdir -p "${NHT_ROOT}/artifacts"
"${NHT_PYTHON}" "${NHT_ROOT}/smoke.py" \
  --output "${NHT_ROOT}/artifacts/smoke.json"
uv pip freeze --python "${NHT_PYTHON}" \
  > "${NHT_ROOT}/artifacts/requirements-resolved.lock"

printf 'NHT environment is ready: %s\n' "${NHT_PYTHON}"

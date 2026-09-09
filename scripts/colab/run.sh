#!/usr/bin/env bash

# Thin local entry point for the standard-library Colab workflow CLI.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python_executable="${TENNIS_LAB_COLAB_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_executable}" ]]; then
    printf 'error: Colab workflow Python is not executable: %s\n' "${python_executable}" >&2
    printf 'create the repository environment or set TENNIS_LAB_COLAB_PYTHON explicitly\n' >&2
    exit 3
fi
cd "${repo_root}"
exec "${python_executable}" -m scripts.colab.workflow.cli --repo-root "${repo_root}" "$@"

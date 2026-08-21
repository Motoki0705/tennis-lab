#!/usr/bin/env bash

# Run a repository tool from the main checkout's shared virtual environment.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <tool> [args...]" >&2
  exit 2
fi

git_common_dir="$(git rev-parse --path-format=absolute --git-common-dir)" || {
  echo "Failed to locate the repository's shared Git directory." >&2
  exit 1
}
repo_root="$(cd "${git_common_dir}/.." && pwd -P)"
venv_bin="${repo_root}/.venv/bin"
tool_path="${venv_bin}/$1"

if [[ ! -x "${tool_path}" ]]; then
  echo "Repository virtualenv tool not found or not executable: ${tool_path}" >&2
  echo "Run 'uv sync --locked' in the main checkout: ${repo_root}" >&2
  exit 1
fi

shift
export PATH="${venv_bin}${PATH:+:${PATH}}"
exec "${tool_path}" "$@"

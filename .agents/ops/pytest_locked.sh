#!/usr/bin/env bash
# Historical entrypoint: run pytest without a host-wide lock.
set -euo pipefail

pytest_workers=(-n 4)
for argument in "$@"; do
  case "$argument" in
    -n*|--numprocesses|--numprocesses=*)
      pytest_workers=()
      ;;
  esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "pytest_locked: repository .venv/bin/python is missing or not executable" >&2
  exit 2
fi
cd -- "$repo_root"
exec "$repo_root/.venv/bin/python" -m pytest "${pytest_workers[@]}" "$@"

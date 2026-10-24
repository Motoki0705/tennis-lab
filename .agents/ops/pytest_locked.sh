#!/usr/bin/env bash
# Serialize pytest across worktrees; keep implementation agents independent.
set -euo pipefail

for argument in "$@"; do
  case "$argument" in
    -n*|--numprocesses|--numprocesses=*)
      echo "pytest_locked: worker count is fixed at -n 4; omit -n/--numprocesses" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
pytest_lock="${PYTEST_LOCK:-${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/pytest.lock}"
if [[ "$pytest_lock" != /* ]]; then
  echo "pytest_locked: PYTEST_LOCK / XDG_STATE_HOME must resolve to an absolute path" >&2
  exit 2
fi
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
  echo "pytest_locked: repository .venv/bin/python is missing or not executable" >&2
  exit 2
fi
umask 077
mkdir -p -- "$(dirname -- "$pytest_lock")"
cd -- "$repo_root"
exec flock --no-fork "$pytest_lock" "$repo_root/.venv/bin/python" -m pytest -n 4 "$@"

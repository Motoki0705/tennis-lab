#!/usr/bin/env bash
set -euo pipefail

runner_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$runner_root/.venv/bin/python" -c \
  'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("tools.dsh.runner", run_name="__main__")' \
  "$runner_root" "$@"
